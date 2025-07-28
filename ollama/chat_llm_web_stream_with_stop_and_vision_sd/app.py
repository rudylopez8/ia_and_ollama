import os
import time
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import ollama
from diffusers import StableDiffusionPipeline
import torch

# Configuration
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, async_mode='eventlet')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modèles
CHAT_MODEL = "gemma3" #mistral-small3.2
temperature = 0.125
#top_k = 4
#num_predict=8000
num_ctx=8000
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
#    "runwayml/stable-diffusion-v1-5",

#    "stabilityai/stable-diffusion-xl-base-1.0",
    safety_checker=None
)
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipeline = sd_pipeline.to(device)
sd_pipeline.enable_attention_slicing()

# État par session
interrupt_flags = {}
messages = [
    {'role': 'system', 'content': "tu es un programme utile fiable et serviable qui exécute les ordre doné par l'utilisateur."}
]

# Routes HTTP
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return jsonify({'filepath': filename}), 200

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Tâche de fond image
def background_image_task(sid, prompt):
    try:
        img = sd_pipeline(prompt, num_inference_steps=256, guidance_scale=7.5).images[0]
        fname = f"sd_{int(time.time())}.png"
        img.save(os.path.join(UPLOAD_FOLDER, fname))
        socketio.emit('receive_image', {'filepath': fname}, room=sid)
    except Exception as e:
        socketio.emit('receive_image', {'error': str(e)}, room=sid)

# Chat streaming
@socketio.on('send_message')
def handle_send_message(data):
    sid = request.sid
    interrupt_flags[sid] = False

    user_msg = {'role': 'user', 'content': data.get('message', '')}
    if data.get('image'):
        user_msg['images'] = [os.path.join(UPLOAD_FOLDER, data['image'])]
    messages.append(user_msg)

    try:
        stream = ollama.chat(CHAT_MODEL,
        options={"temperature":temperature, "num_ctx":num_ctx},                             
                              messages=messages, stream=True)
    except Exception as e:
        emit('receive_message', {'response': f"Erreur IA : {e}"})
        return

    acc = []
    for part in stream:
        if interrupt_flags.get(sid):
            break
        chunk = part['message']['content']
        acc.append(chunk)
        emit('receive_message', {'response': chunk})
        socketio.sleep(0)
    full = ''.join(acc)
    messages.append({'role': 'assistant', 'content': full})
    interrupt_flags[sid] = False

# Génération image
@socketio.on('generate_image')
def handle_generate_image(data):
    sid = request.sid
    interrupt_flags[sid] = False
    prompt = data.get('prompt', '').strip()
    if not prompt:
        emit('receive_image', {'error': 'Prompt vide.'})
        return
    socketio.start_background_task(background_image_task, sid, prompt)

# Stop
@socketio.on('stop_generation')
def handle_stop():
    interrupt_flags[request.sid] = True

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
