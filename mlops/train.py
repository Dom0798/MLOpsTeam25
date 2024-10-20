# usage: python train.py --config config.yaml
import argparse
import json
import os

import joblib
import pandas as pd
import xgboost
from catboost import CatBoostRegressor

from utils import load_config, rewrite_yaml
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import mlflow


def train(X_train_path: str, y_train_path: str, model_name: str, model_path: str, train_means: dict, train_std: dict, config: dict = None, **model_params) -> XGBRegressor:
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    if model_name == 'XGBRegressor':
        model = XGBRegressor(**model_params)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**model_params)
    elif model_name == 'CatBoostRegressor':
        model = CatBoostRegressor(**model_params)
    else:
        raise ValueError(f"Model {model_name} not supported")
    print(model)
    model.fit(X_train, y_train.squeeze())
    print("Score in train set: ", model.score(X_train, y_train))
    print("Model trained successfully")

    if config:
        if config['mlflow']['log_experiment']:
            experiment = mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
            mlflow.set_experiment(config['mlflow']['experiment_name'])
            pd_dataset = mlflow.data.from_pandas(pd.concat([X_train, y_train], axis=1), name='Train', targets='price')
            print("mlflow tracking uri:", mlflow.tracking.get_tracking_uri())
            print("experiment:", config['mlflow']['experiment_name'])

            with mlflow.start_run() as run:
                print("Logging model training...")
                mlflow.log_input(context='Train', dataset=pd_dataset)
                signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
                mlflow.log_metric('train_score', model.score(X_train, y_train))
                mlflow.log_dict(train_std, 'train_dataset_std.json')
                mlflow.log_dict(train_means, 'train_dataset_mean.json')
                if model_name == 'XGBRegressor':
                    mlflow.log_params(model.get_params())
                    mlflow.xgboost.log_model(model, f"{model_name}_model", signature=signature)
                    mlflow.log_dict(model.get_booster().get_score(importance_type='weight'), 'train_feature_importance.json')
                    fig = xgboost.plot_importance(model.get_booster()).figure
                    fig.tight_layout()
                    mlflow.log_figure(fig, 'train_feature_importance.png')
                elif model_name == 'RandomForestRegressor':
                    mlflow.log_params(model.get_params())
                    mlflow.sklearn.log_model(model, f"{model_name}_model", signature=signature)
                    mlflow.log_dict(model.feature_importances_, 'train_feature_importance.json')
                elif model_name == 'CatBoostRegressor':
                    model: CatBoostRegressor = model
                    mlflow.log_params(model.get_params())
                    mlflow.catboost.log_model(model, f"{model_name}_model", signature=signature)
                    mlflow.log_dict(model.feature_importances_, 'train_feature_importance.json')
                config['mlflow']['last_run_id'] = run.info.run_id
                rewrite_yaml(config, 'params.yaml')
            
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    return model

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', dest='config', required=True)
    args = argparse.parse_args()
    config = load_config(args.config)
    train(config['data']['X_train_path'], config['data']['y_train_path'], config['train']['model_name'], 
          #config['train']['model_path'].format(model_name=config['train']['model_name']), 
          config['train']['model_path'], config['train']['dataset_mean'], config['train']['dataset_std'],
          config, **config['train']['models'][config['train']['model_name']])