from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import json

# Initialiser les outils NLP
stemmer = SnowballStemmer("french")
stop_words = set(stopwords.words("french"))

def preprocess_text(text):
    # 1. Minuscules
    text = text.lower()
    # 2. Suppression ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Suppression stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 4. Racines des mots (stemming)
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Chargement des données
with open("data/weather_dataset.txt", "r", encoding="utf-8") as f:
    train_data = []
    for line in f:
        try:
            # Séparation description/label
            desc, label = line.strip().split(',')
            # Conversion en tuple (description, label)
            train_data.append((desc.strip(), int(label)))
        except ValueError as e:
            print(f"Avertissement: ligne ignorée - {line.strip()} ({str(e)})")

print(f"{len(train_data)} exemples chargés avec succès")

# Après le chargement des données
# Vérification de la qualité des données
texts = [x[0] for x in train_data]
labels = [x[1] for x in train_data]
print(f"Distribution des classes : {sum(labels)/len(labels):.1%} positifs")

# Vérification des doublons
unique_descriptions = len(set(texts))
print(f"{len(texts) - unique_descriptions} doublons détectés")

# Avant l'entraînement
if len(np.unique(labels)) < 2:
    raise ValueError("Les données ne contiennent qu'une seule classe")

# Création du pipeline ML
model = Pipeline([
    ('vectorizer', TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=1000,
        ngram_range=(1, 2)
    )),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])

# Entraînement
# Séparation entraînement/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Recherche d'hyperparamètres
param_grid = {
    'vectorizer__max_features': [500, 1000, 1500],
    'classifier__C': [0.1, 1, 10]
}

# Création du modèle optimisé
model = GridSearchCV(
    estimator=Pipeline([
        ('vectorizer', TfidfVectorizer(preprocessor=preprocess_text)),
        ('classifier', LogisticRegression())
    ]),
    param_grid=param_grid,
    cv=5  # Validation croisée 5 folds
)

# Entraînement avec optimisation
model.fit(X_train, y_train)

# Évaluation
print(f"Meilleurs paramètres: {model.best_params_}")
print(f"Score test: {model.score(X_test, y_test):.2%}")

# Évaluation détaillée
y_pred = model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump({
        'vectorizer': model.best_estimator_.named_steps['vectorizer'],
        'classifier': model.best_estimator_.named_steps['classifier']
    }, f)

# Sauvegarde des noms de features
with open("artifacts/features.pkl", "wb") as f:
    pickle.dump(model.best_estimator_.named_steps['vectorizer'].get_feature_names_out(), f)

# Après la sauvegarde
# Ajout de metadata
model_metadata = {
    'accuracy': model.score(X_test, y_test),
    'dataset_size': len(train_data),
    'features_version': '1.0'
}

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(model_metadata, f)
