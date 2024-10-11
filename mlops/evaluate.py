# usage python evaluate.py --config params.yaml
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_config


def plot_regression(y_test: pd.Series, y_pred: np.ndarray, 
                    rmse: float, mse: float, r2: float, save_path: str) -> None:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted price (scaled)")
    plt.text(min_val, max_val, f"RMSE: {rmse:.4f}", va='top')
    plt.text(min_val, max_val-(max_val/24), f"MSE: {mse:.4f}", va='top')
    plt.text(min_val, max_val-(max_val/12), f"R2: {r2:.4f}", va='top')
    plt.savefig(os.path.join(save_path, 'regression_plot.png'))
    plt.close()
    print('Regression plot saved')

def plot_feature_importance(model: 'Model', X: pd.DataFrame, save_path: str) -> None:
    features = X.columns
    importances = model.feature_importances_
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Feature importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()
    print('Feature importance plot saved')

def eval(model_path: str, X_test_path: str, y_test_path: str, report_path: str) -> None:
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    y_test = y_test.squeeze()
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("Model evaluation in test set")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    plot_regression(y_test, y_pred, rmse, mse, r2, report_path)
    plot_feature_importance(model, X_test, report_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = load_config(args.config)
    eval(config['train']['model_path'].format(model_name=config['train']['model_name']),
         config['data']['X_test_path'], config['data']['y_test_path'], config['reports']['images_path'])