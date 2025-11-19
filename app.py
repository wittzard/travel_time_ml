from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
model_folder = os.path.join(cwd, 'model')  

encoder_file_name = 'encoder.pkl'
encoder_file_path = os.path.join(model_folder, encoder_file_name)
encoder = joblib.load(encoder_file_path)

model_file_name = 'best_model.pkl'
model_file_path = os.path.join(model_folder, model_file_name)
model = joblib.load(model_file_path)

categorical_cols = ['weather','traffic_level','time_of_day','vehicle_type']
numeric_cols = ['distance_km','preparation_time_min','courier_experience_yrs']

# NOTE valid categories for check input payload
validate = {
    'weather': ['windy', 'clear', 'foggy', 'rainy', 'snowy'], 
    'traffic_level': ['low', 'medium', 'high'], 
    'time_of_day': ['afternoon', 'evening', 'night', 'morning'], 
    'vehicle_type': ['scooter', 'bike', 'car']
}

class InputData(BaseModel):
    weather: str
    traffic_level: str
    time_of_day: str
    vehicle_type: str
    distance_km: float
    preparation_time_min: float
    courier_experience_yrs: float

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):

    # NOTE Logic check input  ==============================
    errors = []

    for col in categorical_cols:
        val = getattr(data, col)
        if val not in validate[col]:
            errors.append(f"{col}='{val}' not in allowed categories: {validate[col]}")
    

    if data.distance_km < 0:
        errors.append("distance_km must be >= 0")
    if data.preparation_time_min < 0:
        errors.append("preparation_time_min must be >= 0")
    if data.courier_experience_yrs < 0:
        errors.append("courier_experience_yrs must be >= 0")

    if errors:
        return {"error": errors}
    # ======================================================
    
    df = pd.DataFrame([data.dict()])

    X_cat = encoder.transform(df[categorical_cols])
    X_num = df[numeric_cols].values
    X = np.hstack([X_num, X_cat])

    pred = model.predict(X)[0]

    return {"predicted_delivery_time_min": round(float(pred),2)}

# health check
@app.get("/health")
def health():
    if encoder is not None and model is not None:
        return {"status": "ok"}
    else:
        return {"status": "error"}
