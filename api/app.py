from fastapi import FastAPI
from api.models import HomeFeatures
from api.runtime import predict

app = FastAPI()


@app.post("/predict-price")
def predict_home_price(features: HomeFeatures):
    return {'price': predict(features)}
