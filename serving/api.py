from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import os
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string

app = FastAPI()

# Déplacer la définition de preprocess_text AVANT le chargement du modèle
stemmer = SnowballStemmer("french")
stop_words = set(stopwords.words("french"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Ajouter cette ligne cruciale pour enregistrer la fonction dans le module principal
import __main__
__main__.preprocess_text = preprocess_text

# Charger le modèle APRÈS avoir défini la fonction
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl"))
with open(model_path, "rb") as f:
    model = pickle.load(f)

class WeatherInput(BaseModel):
    description: str

class FeedbackInput(BaseModel):
    description: str
    prediction: int
    feedback: str    

@app.post("/predict")
def predict_weather(input: WeatherInput):
    # Vérification de la connexion
    print(f"Requête reçue : {input.description[:50]}...")  # Log de débogage
    vectorized_input = model['vectorizer'].transform([input.description])
    prediction = model['classifier'].predict(vectorized_input)[0]
    return {"prediction": "Beau temps" if prediction == 1 else "Mauvais temps"}


@app.post("/feedback")
def feedback(input: FeedbackInput):
    line = ""
    if (input.feedback == "Like" and input.prediction == "Beau temps") or (input.feedback == "Dislike" and input.prediction == "Mauvais temps"):
        line = input.description + ",1\n"
    else:
        line = input.description + ",0\n"
    with open("../data/weather_dataset.txt", "a", encoding="utf-8") as f:
        f.write(line)    

# Ajouter ce bloc pour l'exécution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
