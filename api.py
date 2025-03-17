from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

# Charger le modèle
model = joblib.load("best_lightgbm_model.pkl")

# Définir l'application FastAPI
app = FastAPI()

# Définir la structure des données en entrée
class ClientData(BaseModel):
    features: List[float]  # Une liste de valeurs numériques

# Endpoint de prédiction
@app.post("/predict")
def predict(data: ClientData):
    features_array = np.array(data.features).reshape(1, -1)
    score = model.predict_proba(features_array)[:, 1][0]  # Probabilité d'être en défaut
    return {"score_credit": score}

