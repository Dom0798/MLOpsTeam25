# usage python preprocess_data.py --config params.yaml
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import load_config, rewrite_yaml


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    return data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # fill missing values with median
    for col in ['bathrooms', 'bedrooms', 'latitude', 'longitude']:
        data[col].fillna(data[col].median(), inplace=True)

    # fill missing values with new category
    data['pets_allowed'].fillna('unknown', inplace=True)
    return data

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    # unnecessary columns
    data.drop(columns =["id", "title", "body", "time", "price_display"], inplace=True)
    # constant columns
    data.drop(columns=["category", "currency", "fee", "price_type"], inplace=True)
    # high cardinality columns
    data.drop(columns=["amenities", "address", "cityname", "state", "source"], inplace=True)
    return data

def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    # Remove rows with missing target
    data.dropna(subset=["price"], inplace=True)
    # Remove no bedrooms
    data = data[data["bedrooms"] > 0]
    return data

def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if config:
        config['train']['dataset_mean'] = X_train.mean().round(8).to_dict()
        config['train']['dataset_std'] = X_train.std().round(8).to_dict()
        rewrite_yaml(config, 'params.yaml')
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def split_data(data: pd.DataFrame, target: str, 
               X_train_path: str, X_test_path: str, y_train_path: str, y_test_path: str, config: dict = None,
               test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=[target])
    y = data[target]
    y = np.log(y)   # log transform the target to normalize it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test = scale_features(X_train, X_test, config)
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print("Data split successfully")
    return X_train, X_test, y_train, y_test

def preprocess_data(data_path: str, data_preprocessed_path: str, 
                    X_train_path: str, X_test_path: str, y_train_path: str, y_test_path: str, config: dict = None,
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(data_path)
    print(f"Raw data shape: {data.shape}")
    # Drop duplicates
    data = drop_duplicates(data)
    # Fill missing values
    data = fill_missing_values(data)
    # Drop columns
    data = drop_columns(data)
    # Remove nans
    data = remove_nans(data)
    # Encode categorical columns
    data = encode_categorical(data)
    print(f"Preprocessed data shape: {data.shape}")
    data.to_csv(data_preprocessed_path, index=False)
    X_train, X_test, y_train, y_test = split_data(data, "price", X_train_path, X_test_path, y_train_path, y_test_path, config, test_size, random_state)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = load_config(args.config)
    preprocess_data(config['data']['raw_data_path'], config['data']['interim_data_path'], 
                    config['data']['X_train_path'], config['data']['X_test_path'], 
                    config['data']['y_train_path'], config['data']['y_test_path'], config,
                    config['data']['test_size'], config['base']['random_state'])