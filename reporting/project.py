import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Prédiction Météo", layout="centered")

# Titre de l'application
st.title("🌦️ Application de Prédiction Météo")

# Zone de texte pour entrer la description de la météo
user_input = st.text_area("Décrivez la météo actuelle :", 
                          placeholder="Exemple : Le ciel est nuageux avec un vent léger.")

# Bouton pour prédire la météo
if st.button("Prédire"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une description avant de prédire.")
    else:
        # Envoi de la requête à l'API de prédiction
        api_url = "http://localhost:8000/predict"  # Nouvelle URL corrigée
        response = requests.post(api_url, json={"description": user_input})  # Modifier aussi le nom de la clé JSON

        # Vérification de la réponse de l'API
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Erreur dans la réponse")
            st.success(f"🌡️ Prédiction météo : {prediction}")
        else:
            st.error("Erreur lors de la requête à l'API de prédiction.")

