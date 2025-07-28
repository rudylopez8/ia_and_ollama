import time
import json
from ollama import chat
model="llama3.2"

def get_response(messages):
    response_stream = chat(
        model,
        messages=messages,
        stream=True  # Active le mode streaming
    )

    # Stockez les parties de la réponse pour maintenir l'historique
    assistant_responses = []

    # Affichez chaque partie de la réponse mot par mot
    for part in response_stream:
        print(part['message']['content'], end='', flush=True)
        assistant_responses.append(part['message']['content'])

    if assistant_responses:
        complete_response = ''.join(assistant_responses)
    return complete_response

def save_to_file_with_time(content, time_taken, filename="data.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
        file.write(f"\n\nTime taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    # Lire la question à partir du fichier question.txt
    question_file = "question.txt"
    with open(question_file, "r", encoding="utf-8") as file:
        question = "\n".join(line.strip() for line in file)
    messages = [
        {
            "role": "system",
            "content": "Si l'utilisateur écrit en français tu répond en français. Tu es un programme compétant fiable et utile. ",
        },
        #{"role": "control", "content": "thinking"},
        {
            "role": "user",
            "content": question,
        },
    ]

    # Mesurer le temps pris pour obtenir la réponse
    start_time = time.time()
    response=get_response(messages)
    end_time = time.time()
    time_taken = end_time - start_time

    # Définir le nom du fichier
    filename = "data.txt"
            
    # Sauvegarder la réponse dans un fichier avec le temps pris
    save_to_file_with_time(response, time_taken, filename)
            
    print(f"\nLa réponse a été sauvegardée dans {filename} avec le temps pris de {time_taken:.2f} secondes")
