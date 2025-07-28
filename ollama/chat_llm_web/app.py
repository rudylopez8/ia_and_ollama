from flask import Flask, request, jsonify, render_template

# Importez l'outil de chat
from ollama import chat

app = Flask(__name__)

messages = [
{
    'role': 'system',
    "content": "Tu es un programme fiable et utile.",
},
]

@app.route('/')
def home():
    # Retourne le fichier index.html lorsqu'on accède à la racine
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get('message')
    messages.append({'role': 'user', 'content': user_input})

    response_stream = chat(
        'llama3.2',
        messages=messages,
        stream=True
    )

    assistant_responses = []

    for part in response_stream:
        assistant_response_content = part['message']['content']
        print(assistant_response_content, end='',flush=True)
        assistant_responses.append(assistant_response_content)

    full_response = ''.join(assistant_responses)

    if assistant_responses:
        messages.append({'role': 'assistant', 'content': assistant_responses[-1]})

    return jsonify({'response': full_response})

if __name__ == '__main__':
    app.run(debug=True)