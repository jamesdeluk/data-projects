from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Union, List

from model.model_inference import predict

app = FastAPI()

class PredictPayload(BaseModel):
    carat: float
    cut: int
    color: int
    clarity: int
    x: float
    y: float
    z: float

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict-named")
def predict_price(payload: PredictPayload):
    features = [[
        payload.carat,
        payload.cut,
        payload.color,
        payload.clarity,
        payload.x,
        payload.y,
        payload.z,
    ]]
    price = predict(features)
    return {"price": price}

@app.post("/predict-list")
def predict_price(features: List[float]):
    price = predict([features])
    return {"price": price}

@app.post("/predict")
def predict_price(input_data: Union[list, dict] = Body(...)):
    if isinstance(input_data, list):
        if len(input_data) != 7:
            raise HTTPException(status_code=400, detail="Expected a list of 7 features.")
        features = input_data
    elif isinstance(input_data, dict):
        expected_keys = {"carat", "cut", "color", "clarity", "x", "y", "z"}
        if not expected_keys.issubset(input_data.keys()):
            raise HTTPException(status_code=400, detail=f"Missing one or more required keys: {expected_keys}")
        features = [
            input_data["carat"],
            input_data["cut"],
            input_data["color"],
            input_data["clarity"],
            input_data["x"],
            input_data["y"],
            input_data["z"],
        ]
    else:
        raise HTTPException(status_code=400, detail="Input must be a list or a dictionary.")
    price = predict([features])
    return {"price": price}