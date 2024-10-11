# usage: python train.py --config config.yaml
import argparse

import joblib
# import mlflow
import pandas as pd
from utils import load_config
from xgboost import XGBRegressor


def train(X_train_path: str, y_train_path: str, model_name: str, model_path: str, **model_params) -> XGBRegressor:
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    print("Training", model_name, "model")
    if model_name == 'XGBRegressor':
        model = XGBRegressor(**model_params)
        print(model)
    model.fit(X_train, y_train)
    print("Score in train set: ", model.score(X_train, y_train))
    print("Model trained successfully")

    # # mlflow.set_experiment(config['mlflow']['experiment_name'])
    # # mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # # with mlflow.start_run():
    # #     model.fit(X_train, y_train)
    # #     mlflow.sklearn.log_model(model, f"{model_name}_model")

    joblib.dump(model, model_path)
    return model

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', dest='config', required=True)
    args = argparse.parse_args()
    config = load_config(args.config)
    train(config['data']['X_train_path'], config['data']['y_train_path'], config['train']['model_name'], 
          #config['train']['model_path'].format(model_name=config['train']['model_name']), 
          config['train']['model_path'],
          **config['train']['models'][config['train']['model_name']])