# Chatbot & Vision Web App

Une application web interactive de chatbot doté de capacités de génération et d’analyse d’images, basée sur Flask, Socket.IO, Ollama et Stable Diffusion.

---

## 📦 Structure du projet

.
├── app.py
├── uploads/ # Dossier où sont stockées les images générées et uploadées
├── templates/
│ └── index.html # Interface utilisateur
└── README.md # Documentation du projet
yaml

---

## 🚀 Fonctionnalités

- **Chat texte** en streaming via Ollama (llm)
- **Génération d’images** Stable Diffusion (`sd-turbo` ou autre) avec retour asynchrone  
- **Upload & analyse d’images** par le vllm Ollama  
- **Liste & téléchargement** de toutes les images générées ou uploadées  
- **Contrôle d’interruption** des générations (texte et image)  

---

## 🔧 Prérequis

- Python 3.8+  
- CUDA (optionnel, pour accélération GPU)  
- Un environnement virtuel (possible)  

---

## ⚙️ Installation

1. **installer ollama et un llm/vllm**  
    https://ollama.com/
2. **Installer les dépendances**  
    pip install eventlet flask flask-socketio pillow diffusers[torch] transformers accelerate safetensors torch
---

## ▶️ Utilisation

1. **Lancer le serveur**  
    ```bash
    python app.py
    ```
2. **Ouvrir** `http://localhost:5000` dans votre navigateur  
3. **Interactions**  
   - **Chat texte** : tapez un message et cliquez « Envoyer »  
   - **Génération image** : saisissez un prompt et cliquez « Générer Image »  
   - **Upload/Analyse** : sélectionnez un fichier puis « Importer & Analyser »  
   - **Voir images** : cliquez « Voir les images générées » pour afficher/télécharger  

---

## 🎨 Personnalisation

- **Modèles IA**  
  - Chat : changez `CHAT_MODEL` dans `app.py` (nom du modèle ou paramètres)  
  - Diffusion : modifiez le pipeline Diffusers (chemin du modèle ou paramètres)  
- **Taille du chat** : ajustez le CSS dans `templates/index.html`  

---

## 📝 package requirements

flask
flask-socketio[eventlet]
torch
diffusers
ollama
Pillow
eventlet
 