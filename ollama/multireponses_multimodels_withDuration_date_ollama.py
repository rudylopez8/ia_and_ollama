import re
import time
from ollama import chat

# Paramètres globaux
message_system="Tu réponds en français sans limitation avec sincérité logique et inventivitée."
#message_system=""

#temperature = 0.25
#top_k = 4
#num_predict=8000
#num_ctx=8000

def get_response(model,question):
    messages = [
    {
        "role": "system",
        "content": message_system,
    },
    #{"role": "control", "content": "thinking"},
    {
        "role": "user",
        "content": question,
    },
    ]


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

def save_to_file_with_time(content, time_taken, model_name):
    # Nettoyer le nom du modèle pour qu'il soit compatible avec les noms de fichiers
    clean_model_name = re.sub(r'[^\w]', '', model_name)

    # Construire le nom du fichier de réponse avec le temps pris et le nom du modèle
    filename = f"reponse_{clean_model_name}_think_date_{date}.txt"
    
    with open(filename, "a", encoding="utf-8") as file:  # Utiliser 'a' pour append
        file.write(content)
        file.write(f"\n\nTime taken: {time_taken:.2f} seconds\n\n")

if __name__ == "__main__":
    #list_model = ["llama3.2:1b", "granite3.1-moe:1b"]
    #list_model = ["llama3.2:1b", "granite3.1-dense:2b", "granite3.1:moe", "llama3.2", "gemma2:2b", "phi3.5", "deepscaler", "mistral", "mistral-nemo", "gemma2", "phi4", "deepseek-r1:14b", "mistral-small", "smollm2:135m", "smollm2"]
    #"hf.co/croissantllm/CroissantLLMChat-v0.1-GGUF"
    #list_model = ["openthinker", "marco-o1", "smallthinker", "falcon3", "falcon3:1b", "llava", "llama3.2-vision"]
    #list_model = ["granite3.2:2b", "granite3.2"]
    #list_model = ["llama3.2-vision", "phi4-mini"]
    list_model = ["gemma3:1b", "gemma3", "gemma3:12b"]
    # Liste des modèles


    for model_name in list_model:
        # Durée totale pour chaque modèle
        total_duration = 0
        date=time.strftime('%Y%m%d%H%M%S')

        for i in range(1, 17):  # Boucle sur une plage de nombres pour générer les noms de fichiers
            question_file = f"p{i}.txt"
            
            try:
                # Lire la question à partir du fichier
                with open(question_file, "r", encoding="utf-8") as file:
                    question = "\n".join(line.strip() for line in file)

                # Mesurer le temps pris pour obtenir la réponse
                start_time = time.time()
                response = get_response(model_name,question)
                end_time = time.time()
                time_taken = end_time - start_time
                total_duration += time_taken

                # Sauvegarder la réponse dans un fichier avec le temps pris et le nom du modèle
                save_to_file_with_time(f"Question from {question_file}:\n\nResponse:\n{response}\n", time_taken, model_name)

                print(f"\nLa réponse à la question dans {question_file} a été sauvegardée dans reponse_{model_name}.txt avec un temps pris de {time_taken:.2f} secondes")
            
            except FileNotFoundError:
                print(f"Le fichier {question_file} n'existe pas.")

        # Écrire la durée totale pour chaque modèle dans le fichier de réponse
        save_to_file_with_time(f"Durée totale: {total_duration:.2f} seconds\n", total_duration, model_name)
