import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Nouvelle fonction de prétraitement
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Chargement des données (adapté pour BERT)
with open("data/weather_dataset.txt", "r", encoding="utf-8") as f:
    texts, labels = [], []
    for line in f:
        try:
            desc, label = line.strip().split(',')
            texts.append(desc.strip())
            labels.append(int(label))
        except:
            continue

# Création du dataset HuggingFace
full_dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})
dataset = full_dataset.train_test_split(test_size=0.2)

# Tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Modèle BERT
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Métriques d'évaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds)
    }

# Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

# Sauvegarde du modèle
model.save_pretrained("artifacts/bert_model")
tokenizer.save_pretrained("artifacts/bert_tokenizer")

# Métadonnées
model_metadata = {
    'framework': 'transformers',
    'model_type': 'bert-base-uncased',
    'max_length': 128
}

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(model_metadata, f)
