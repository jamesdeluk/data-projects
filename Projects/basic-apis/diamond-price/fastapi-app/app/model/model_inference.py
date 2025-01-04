import pickle
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/diamonds_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(d):
    return model.predict(d)[0]

if __name__ == '__main__':
    details = [[1,3,3,3,3,3,3]]
    price_prediction = predict(details)
    print('Prediction: £', price_prediction)