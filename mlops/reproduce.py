# usage: python mlops/reproduce.py --config params.yaml --run_id 3f0409dddc2a4a3cad322b38e88bbe5e
# usage: python mlops/reproduce.py --config params.yaml

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_config

import mlflow

from evaluate import plot_regression, plot_xgb_feature_importance, plot_rfr_feature_importance


def reproduce(run_id: str, model_name: str, X_test_path: str, y_test_path: str, repro_path: str, tracking_uri: str) -> None:
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    y_test = y_test.squeeze()
    experiment = mlflow.set_tracking_uri(tracking_uri)
    logged_model = f'runs:/{run_id}/{model_name}_model'
    print(f'Loading model {logged_model}')
    try:
        if model_name == 'XGBRegressor':
            model = mlflow.xgboost.load_model(logged_model)
            
        elif model_name == 'RandomForestRegressor':
            model = mlflow.sklearn.load_model(logged_model)
        else:
            raise ValueError(f"Model {model_name} not supported")
    except Exception as e:
        print(f"Error loading model. Check run_id or model_name in params.yaml.")
        return
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("Model evaluation in test set")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    plot_regression(y_test, y_pred, rmse, mse, r2, repro_path, f'{run_id}regression_plot.png')
    if model_name == 'XGBRegressor':
        plot_xgb_feature_importance(model, repro_path, f'{run_id}feature_importance.png')
    elif model_name == 'RandomForestRegressor':
        plot_rfr_feature_importance(model, X_test.columns, repro_path, f'{run_id}feature_importance.png')
    

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', dest='config', required=True)
    argparse.add_argument('--run_id', dest='run_id', required=False)
    args = argparse.parse_args()
    config = load_config(args.config)
    if args.run_id:
        reproduce(args.run_id, config['train']['model_name'], config['data']['X_test_path'], config['data']['y_test_path'], 
                  config['reproduction']['reproduction_path'], config['mlflow']['tracking_uri'])
    else:
        reproduce(config['mlflow']['last_run_id'], config['train']['model_name'],
                  config['data']['X_test_path'], config['data']['y_test_path'], 
                  config['reproduction']['reproduction_path'], config['mlflow']['tracking_uri'])