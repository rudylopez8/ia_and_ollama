import eventlet
eventlet.monkey_patch()
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, request
from ollama import chat  # module IA

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Dictionnaire pour gérer l'interruption par session
interrupt_flags = {}
# Paramètres globaux
model="gemma3:12b"
temperature = 0.25
#top_k = 4
num_predict=8000
num_ctx=8120

messages = [
    {
        'role': 'system',
        "content": "tu es un programme, utile fiable et serviable sincère, avec de très forte compétences en informatique biologie et mathématique.",
    },
]

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('send_message')
def handle_send_message_event(json):
    sid = request.sid
    # Réinitialiser le flag d'interruption pour ce client
    interrupt_flags[sid] = False

    user_input = json['message']
    messages.append({'role': 'user', 'content': user_input})

    # Génération de la réponse en mode streaming
    response_stream = chat(
        model,
        options={"temperature":temperature, "num_predict":num_predict, "num_ctx":num_ctx},
        messages=messages,
        stream=True
    )

    assistant_responses = []
    for part in response_stream:
        # Vérifier régulièrement si une interruption a été demandée
        if interrupt_flags.get(sid, False):
            print("Génération interrompue pour le client :", sid)
            break

        assistant_response_content = part['message']['content']
        emit('receive_message', {'response': assistant_response_content}, broadcast=False)
        socketio.sleep(0)  # Cède le contrôle pour l'envoi immédiat
        assistant_responses.append(assistant_response_content)

    full_response = ''.join(assistant_responses)
    # Sauvegarder la partie générée même en cas d'interruption
    messages.append({'role': 'assistant', 'content': full_response})

    # Réinitialiser le flag pour la prochaine génération
    interrupt_flags[sid] = False

@socketio.on('stop_generation')
def handle_stop_generation():
    sid = request.sid
    interrupt_flags[sid] = True
    print("Arrêt demandé par le client :", sid)

if __name__ == '__main__':
    socketio.run(app, debug=True)
