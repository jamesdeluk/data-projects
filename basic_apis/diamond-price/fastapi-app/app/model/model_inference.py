import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_NAME = "diamonds_rf_model.pkl"

with open(f"{BASE_DIR}/{MODEL_NAME}", "rb") as f:
    model = pickle.load(f)

def predict(d):
    return model.predict(d)[0]

# if __name__ == '__main__':
#     details = [[1,3,3,3,3,3,3]]
#     price_prediction = predict(details)
#     print(f"Prediction: £{price_prediction}")