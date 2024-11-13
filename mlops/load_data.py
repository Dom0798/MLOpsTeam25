# usage: python load_data.py --config params.yaml
import argparse
import os
import sys

import pandas as pd
import yaml
from utils import load_config

def load_data(filepath: str, out_file: str) -> pd.DataFrame:
    data = pd.DataFrame()
    for file in os.listdir(filepath):
        if file.endswith(".csv"):
            parcial_data = pd.read_csv(os.path.join(filepath, file), sep=";", encoding='cp1252')
            data = pd.concat([data, parcial_data], axis=0)
            print(f"Shape of the {file} is: {parcial_data.shape}")
    data.to_csv(out_file, index=False)
    print("Data loaded successfully")
    return data

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = load_config(args.config)
    load_data(config['data']['external_data_path'], config['data']['raw_data_path'])