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

# Ajouter ce bloc pour l'exécution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
