import joblib
import pandas as pd
from src.features import prepare_text

MODEL_PATH_RF = "models/model_rf.pkl"

def predict_ticket(data: dict):
    df = pd.DataFrame([data])
    df = prepare_text(df)

    drop_cols = ["Subject", "Body", "text"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    model1 = joblib.load(MODEL_PATH_RF)
    pred = model1.predict(X)[0]
    return {"predicted_priority": pred}