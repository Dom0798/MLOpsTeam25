# usage:
# UNIT TESTS: pytest -m "not integtest" -v -s -x
# loading data: pytest -k load -v -s
# preprocessing data: pytest -k preprocess -v -s
# model params: pytest -k train -v -s
# model evaluation: pytest -k evaluation -v -s
# model input (necessary to change input array): pytest -k input -v -s
# INTEGRATION TESTS: pytest -m integtest -v -s

import warnings

import numpy as np
import onnx
import onnxruntime
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from mlops.ApartmentRent import (ApartmentPriceModel,
                                 ApartmentPriceModelwithParams)
from mlops.utils import load_config

warnings.filterwarnings("ignore")


@pytest.fixture
def model():
    model = ApartmentPriceModel('../data/raw')
    model.load_data()
    return model


@pytest.fixture
def model_w_params():
    model = ApartmentPriceModelwithParams('../data/raw', '../params.yaml')
    model.load_data().preprocess_data().split_data()
    return model


@pytest.fixture
def model_trained():
    model = ApartmentPriceModelwithParams('../data/raw', '../params.yaml')
    model.load_data().preprocess_data().split_data().train_model()
    return model


@pytest.fixture
def params_only():
    return load_config('../params.yaml')


## Test data loading
def test_load_data(model):
    print("Data shape:", model.data.shape)
    print("Expected shape:", (10000 + 99492, 22))
    assert model.data.shape == (10000 + 99492, 22)    # known valus from 10k and 100k datasets


## Test preprocess_data method
def test_preprocess_data_columns(model):
    original_columns = model.data.columns
    model.preprocess_data()
    preprocessed_columns = model.data.columns
    print("Original columns: ", original_columns.tolist())
    print("Preprocessed columns: ", preprocessed_columns.tolist())
    assert original_columns.equals(preprocessed_columns) == False, "Columns are not preprocessed"


def test_preprocess_data_normalization_mean(model):
    model.preprocess_data().split_data()
    print("Original mean: ", model.X_train.mean().tolist())
    print("Normalized mean: ", model.X_train_scaled.mean().tolist())
    assert np.allclose(model.X_train_scaled.mean(), 0, atol=1e-5), "Data mean not close to 0"


def test_preprocess_data_normalization_std(model):
    model.preprocess_data().split_data()
    print("Original std: ", model.X_train.std().tolist())
    print("Normalized std: ", model.X_train_scaled.std().tolist())
    assert np.allclose(model.X_train_scaled.std(), 1, atol=1e-5), "Data std not close to 1"


## Test model parameters
def test_train_parameters(model_w_params):
    assert model_w_params.config['train']['model_name'] in ['XGBRegressor', 'RandomForestRegressor'], f"Model name {model_w_params.config['train']['model_name']} is not supported"
    print(f"Model parameters to change for {model_w_params.config['train']['model_name']}:")
    if model_w_params.config['train']['model_name'] == 'XGBRegressor':
        print(model_w_params.config['train']['models'][model_w_params.config['train']['model_name']])
        assert model_w_params.config['train']['models'][model_w_params.config['train']['model_name']].keys() <= XGBRegressor().get_params().keys(), \
        "One or more model parameters are not subset of XGBRegressor parameters"
    elif model_w_params.config['train']['model_name'] == 'RandomForestRegressor':
        print(model_w_params.config['train']['models'][model_w_params.config['train']['model_name']])
        assert model_w_params.config['train']['models'][model_w_params.config['train']['model_name']].keys() <= RandomForestRegressor().get_params().keys(), \
        "One or more model parameters are not subset of RandomForestRegressor parameters"


## Test model evaluation
def test_evaluation_w_benchmark(model_trained):
    print("Benchmark scores in evaluation:")
    benchmark_predictions = [model_trained.y_test.median()] * len(model_trained.y_test)
    b_mse = mean_squared_error(model_trained.y_test, benchmark_predictions)
    b_rmse = np.sqrt(b_mse)
    b_r2 = r2_score(model_trained.y_test, benchmark_predictions)
    print(f"RMSE: {b_rmse:.4f} \nMSE: {b_mse:.4f} \nR2: {b_r2:.4f}")
    print(f"Model {model_trained.config['train']['model_name']} evaluation:")
    y_pred = model_trained.model.predict(model_trained.X_test_scaled)
    m_mse = mean_squared_error(model_trained.y_test, y_pred)
    m_rmse = np.sqrt(m_mse)
    m_r2 = r2_score(model_trained.y_test, y_pred)
    print(f"RMSE: {m_rmse:.4f} \nMSE: {m_mse:.4f} \nR2: {m_r2:.4f}")
    assert m_mse < b_mse, "Model MSE is not better than benchmark MSE"
    assert m_rmse < b_rmse, "Model RMSE is not better than benchmark RMSE"
    assert m_r2 > b_r2, "Model R2 is not better than benchmark R2"


## Test model input
def test_input_ranges_n_inference(model_trained, input_data=[1, 1, 120, 25, -70, 1, 0, 1, 1, 0, 1]):
    input_data = np.array(input_data).reshape(1, -1)
    org_data = model_trained.X_train
    for i, feature in enumerate(org_data.columns):
        print(f"{feature}: {org_data[feature].min()} - {input_data[0][i]} - {org_data[feature].max()}")
        assert org_data[feature].min() <= input_data[0][i] <= org_data[feature].max(), \
            f"Input data for {feature} is out of range: {org_data[feature].min()} - {input_data[0][i]} - {org_data[feature].max()}"
    prediction = model_trained.predict(input_data)
    # print(prediction)
    print(f"Predicted price: ${np.exp(prediction):.3f}")


def test_onnx_inference(params_only, input_data=[1, 1, 120, 25, -70, 1, 0, 1, 1, 0, 1]):
    input_data = np.array(input_data).reshape(1, -1)
    # print("1st type", input_data.dtype)
    onnx_model = onnx.load('../models/model.onnx')
    onnx.checker.check_model(onnx_model)
    onnx_model = onnxruntime.InferenceSession('../models/model.onnx')
    input_name = onnx_model.get_inputs()[0].name
    scaler = StandardScaler()
    scaler.mean_ = np.array(list(params_only['train']['dataset_mean'].values()))
    scaler.scale_ = np.array(list(params_only['train']['dataset_std'].values()))
    input_data = scaler.transform(input_data)
    input_data = input_data.astype(np.float32)
    # print("mean", scaler.mean_)
    # print("input", input_data)
    # print("input type", input_data.dtype)
    pred_onnx = onnx_model.run(None, {input_name: input_data})[0]
    # print(pred_onnx[0][0])
    print(f"Predicted price: ${np.exp(pred_onnx[0][0]):.3f}")


## Test whole integration
@pytest.mark.integtest
def test_integration(model_w_params):
    model = model_w_params.load_data()
    test_load_data(model)
    test_preprocess_data_columns(model)
    model.split_data()
    model.train_model()
    test_evaluation_w_benchmark(model)
    model.evaluate_model()
    test_input_ranges_n_inference(model, [1, 1, 120, 25, -70, 1, 0, 1, 1, 0, 1])
