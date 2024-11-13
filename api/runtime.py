import onnxruntime as ort
import numpy as np

import pickle
from sklearn.preprocessing import StandardScaler
from api.models import HomeFeatures

with open('models/scaler.pkl', 'rb') as f:
    scaler: StandardScaler = pickle.load(f)

onnx_session = ort.InferenceSession('models/model.onnx')


def scale_inputs(array: np.array) -> np.array:
    return scaler.transform(array)


def predict_array(array: np.array) -> float:
    array = np.array([array])
    array = scale_inputs(array)
    model_output = onnx_session.run(None, {
        'X': np.array(array)
    })
    model_output = np.exp(model_output[0][0])
    model_output: np.float32 = model_output[0]
    return model_output.astype(float)


def predict(features: HomeFeatures) -> float:
    arr = np.array([
        features.bathrooms,
        features.bedrooms,
        features.square_feet,
        features.latitude,
        features.longitude,
        1.0 if features.has_thumbnail else 0.0,
        1.0 if features.has_photo else 0.0,
        1.0 if features.allows_pets else 0.0,
        1.0 if features.allows_pets else 0.0,
        1.0 if features.allows_pets else 0.0,
        1.0 if features.allows_pets is None else 0.0
    ]).astype(np.float32)
    return predict_array(arr)
