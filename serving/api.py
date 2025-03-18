from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = FastAPI()

# Charger le modèle BERT et le tokenizer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "bert_model"))
tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "bert_tokenizer"))

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

class WeatherInput(BaseModel):
    description: str

class FeedbackInput(BaseModel):
    description: str
    prediction: int
    feedback: str    

@app.post("/predict")
def predict_weather(input: WeatherInput):
    # Tokenization et prédiction
    inputs = tokenizer(
        input.description,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities).item()
    
    return {"prediction": "beau temps" if prediction == 1 else "mauvais temps"}


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
