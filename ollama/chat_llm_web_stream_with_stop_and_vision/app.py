import eventlet
# Pour permettre l'asynchrone avec eventlet
eventlet.monkey_patch()
import os
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import ollama


app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')
model="gemma3"
temperature = 0.125
#top_k = 4
#num_predict=8000
num_ctx=4000

# Dossier d'upload (créé s'il n'existe pas)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionnaire pour gérer l'interruption par session
interrupt_flags = {}

# Historique des messages (incluant un message système initial)
messages = [
    {
        'role': 'system',
        "content": "tu es un programme utile fiable et serviable.",
    },
]

def convert_to_png(image_path):
    """
    Vérifie si l'image est au format PNG, sinon la convertit en PNG.
    Renvoie le chemin de l'image au format PNG.
    """
    try:
        img = Image.open(image_path)
        if img.format != 'PNG':
            png_path = os.path.splitext(image_path)[0] + ".png"
            img.save(png_path, format="PNG")
            print(f"L'image {image_path} a été convertie en {png_path}")
            return png_path
        return image_path
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image {image_path}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("Aucun fichier fourni dans la requête")
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    file = request.files['file']
    if file.filename == '':
        print("Aucun fichier sélectionné")
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Fichier reçu et sauvegardé : {filepath}")
    return jsonify({'filepath': filename}), 200

@socketio.on('send_message')
def handle_send_message_event(json):
    sid = request.sid
    # Réinitialiser le flag d'interruption pour ce client
    interrupt_flags[sid] = False

    user_input = json.get('message')
    image_filename = json.get('image')  # facultatif

    # Créer le message de l'utilisateur avec le rôle 'user'
    user_message = {'role': 'user', 'content': user_input}

    # S'il y a une image, la traiter et l'ajouter dans le champ "images"
    if image_filename:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        compatible_image_path = convert_to_png(image_path)
        if compatible_image_path:
            user_message['images'] = [compatible_image_path]

    # Ajouter le message utilisateur (contenant potentiellement l'image) à l'historique
    messages.append(user_message)

    # Appel de l'IA en mode streaming
    try:
        response_stream = ollama.chat(
            model,
        options={"temperature":temperature, "num_ctx":num_ctx},                             
            messages=messages,
            stream=True
        )
    except Exception as e:
        emit('receive_message', {'response': f"Erreur lors de l'appel de l'IA: {e}"})
        return

    assistant_responses = []
    for part in response_stream:
        if interrupt_flags.get(sid, False):
            print("Génération interrompue pour le client :", sid)
            break
        assistant_response_content = part['message']['content']
        emit('receive_message', {'response': assistant_response_content}, broadcast=False)
        socketio.sleep(0)
        assistant_responses.append(assistant_response_content)
    full_response = ''.join(assistant_responses)
    messages.append({'role': 'assistant', 'content': full_response})
    interrupt_flags[sid] = False

@socketio.on('stop_generation')
def handle_stop_generation():
    sid = request.sid
    interrupt_flags[sid] = True
    print("Arrêt demandé par le client :", sid)

if __name__ == '__main__':
    socketio.run(app, debug=True)
