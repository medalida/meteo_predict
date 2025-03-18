from fastapi import FastAPI
import pickle
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle entraîné
with open("../artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

class WeatherInput(BaseModel):
    description: str

@app.post("/predict")
def predict_weather(input: WeatherInput):
    # Vectoriser la description avant la prédiction
    vectorized_input = model['vectorizer'].transform([input.description])
    prediction = model['classifier'].predict(vectorized_input)[0]
    return {"prediction": "Beau temps" if prediction == 1 else "Mauvais temps"}
