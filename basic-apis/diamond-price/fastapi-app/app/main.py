from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"welcome_message": "hi!"}

@app.post("/predict")
def predict_price(payload: PredictPayload):
    criteria = [[
        payload.carat,
        payload.cut,
        payload.color,
        payload.clarity,
        payload.x,
        payload.y,
        payload.z,
    ]]
    price = predict(criteria)
    return {"input": payload, "price": price}