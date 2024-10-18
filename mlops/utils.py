import json
import os

import matplotlib
import yaml

import mlflow


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def rewrite_yaml(changed_file, config_path: str) -> None:
    with open(config_path, 'w') as file:
        yaml.safe_dump(changed_file, file, sort_keys=False)