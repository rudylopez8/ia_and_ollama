from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()
from ollama import chat  # module IA

# Utilisez le mode eventlet pour permettre le streaming asynchrone
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

messages = [
    {
        'role': 'system',
        "content": "tu es un programme utile fiable et serviable.",
    },
]

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('send_message')
def handle_send_message_event(json):
    user_input = json['message']
    messages.append({'role': 'user', 'content': user_input})

    # Génération de la réponse en mode streaming
    response_stream = chat(
        'llama3.2',
        messages=messages,
        stream=True
    )

    assistant_responses = []
    for part in response_stream:
        assistant_response_content = part['message']['content']
        # Émettre immédiatement chaque chunk
        emit('receive_message', {'response': assistant_response_content}, broadcast=False)
        # Céder le contrôle pour permettre l'envoi immédiat
        socketio.sleep(0)
        assistant_responses.append(assistant_response_content)
    
    full_response = ''.join(assistant_responses)
    messages.append({'role': 'assistant', 'content': full_response})

if __name__ == '__main__':
    socketio.run(app, debug=True)
