import streamlit as st
import requests

api_url = "http://localhost:8000"  # URL de l'API de prédiction

# Configuration de la page
st.set_page_config(page_title="Prédiction Météo", layout="centered")

# Titre de l'application
st.title("🌦️ Application de Prédiction Météo")

# Zone de texte pour entrer la description de la météo
user_input = st.text_area("Décrivez la météo actuelle :", 
                          placeholder="Exemple : Le ciel est nuageux avec un vent léger.")

def send_feedback(feedback):
    feedback_data = {"description": user_input, "prediction": prediction, "feedback": feedback}
    feedback_url = api_url+"/feedback"
    requests.post(feedback_url, json=feedback_data)

# Bouton pour prédire la météo
if st.button("Prédire"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une description avant de prédire.")
    else:
        # Envoi de la requête à l'API de prédiction
        response = requests.post(api_url+"/predict", json={"description": user_input})  # Modifier aussi le nom de la clé JSON

        # Vérification de la réponse de l'API
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Erreur dans la réponse")
            st.success(f"🌡️ Prédiction météo : {prediction}")
            
            # Ajout des boutons de feedback
            feedback = st.radio("Avez-vous trouvé la prédiction utile ?", ("Sélectionner", "Like", "Dislike"))
            
            if feedback == "Like":
                st.write("Merci pour votre retour positif ! 👍")
                send_feedback(feedback)
            elif feedback == "Dislike":
                st.write("Désolé pour cette prédiction. Nous travaillons pour nous améliorer. 👎")
                send_feedback(feedback)         
        
        else:
            st.error("Erreur lors de la requête à l'API de prédiction.")


