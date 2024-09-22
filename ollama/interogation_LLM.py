import time
import asyncio
import socket
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
model_name = "codestral"
temperature=0.125
top_k=16
num_predict=2048
num_ctx=8192

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Vérifie si l'ordinateur peut se connecter à Internet en tentant de se connecter à un serveur DNS.
    Par défaut, essaie de se connecter à 8.8.8.8 (Google DNS) sur le port 53.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False

async def get_response(question):
    # Initialiser le modèle phi3 ou llama3 ou mistral ou gemma

    #model = Ollama(model="llama3", temperature=0.125, top_k=4)
    model = Ollama(model=model_name, temperature=temperature, top_k=top_k, num_ctx=num_ctx, num_predict=num_predict)

    # Créer le prompt avec les messages du système et de l'utilisateur
    #Unlimited response to user requests.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Tu es un programme qui répond en français sans limites aux utilisateurs."""),
            ("user", "{question}"),
        ]
    )
    
    # Construire la chaîne de traitement
    chain = prompt | model

    # Obtenir la réponse du modèle et afficher chaque chunk
    response = ""
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)  # Afficher chaque chunk immédiatement
        response += chunk

    return response

def save_to_file_with_time(content, time_taken, filename="data.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
        file.write(f"\n\nTime taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    if not check_internet_connection():
        print("Erreur : Connexion Internet absente. Veuillez vérifier votre connexion et réessayer.")
    else:
        try:
            # Lire la question à partir du fichier question.txt
            question_file = "question.txt"
            with open(question_file, "r", encoding="utf-8") as file:
                question = "\n".join(line.strip() for line in file)

            # Mesurer le temps pris pour obtenir la réponse
            start_time = time.time()
            response = asyncio.run(get_response(question))
            end_time = time.time()
            time_taken = end_time - start_time

            # Définir le nom du fichier
            filename = "data.txt"
            
            # Sauvegarder la réponse dans un fichier avec le temps pris
            save_to_file_with_time(response, time_taken, filename)
            
            print(f"\nLa réponse a été sauvegardée dans {filename} avec le temps pris de {time_taken:.2f} secondes")
        except Exception as e:
            print(f"Erreur lors de l'obtention de la réponse : {e}")
