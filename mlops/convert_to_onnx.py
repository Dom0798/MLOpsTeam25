# usage: python mlops/convert_to_onnx.py --config params.yaml
import argparse

import joblib
import numpy as np
import skl2onnx
from skl2onnx import to_onnx
from utils import load_config
import pandas as pd


def convert_to_onnx(model_path: str, X_train_path: str, onnx_model_path: str) -> None:
    model = joblib.load(model_path)
    X_train = pd.read_csv(X_train_path)
    onnx_model = to_onnx(model, X_train.to_numpy().astype(np.float32))

    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Model exported to ONNX format.")


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = load_config(args.config)
    convert_to_onnx(config['train']['model_path'], config['data']['X_train_path'], config['onnx']['onnx_model_path'])
