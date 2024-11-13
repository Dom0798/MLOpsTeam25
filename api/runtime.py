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
        features.has_photo_Thumbnail,
        features.has_photo_Yes,
        features.pets_allowed_CatsDogs,
        features.pets_allowed_CatsDogsNone,
        features.pets_allowed_Dogs,
        features.pets_allowed_unknown
    ]).astype(np.float32)
    return predict_array(arr)
