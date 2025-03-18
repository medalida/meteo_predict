# Utiliser l'image officielle Python
FROM python:3.9

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt requirements.txt
COPY project.py project.py

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exécuter Streamlit au démarrage du conteneur
CMD ["streamlit", "run", "project.py", "--server.port=8501", "--server.address=0.0.0.0"]
