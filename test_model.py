import joblib
import numpy as np

# Charger le modèle (uniquement)
model = joblib.load("best_lightgbm_model.pkl")

# Générer des données aléatoires avec le bon nombre de features
test_data = np.random.rand(1, 42)  # Remplace 42 par le vrai nombre de features attendues

# Prédiction
prediction = model.predict_proba(test_data)[:, 1]

# Afficher le résultat
print(f"Prédiction test : {prediction[0]:.4f}")

