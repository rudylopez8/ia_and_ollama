import time
import asyncio
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import re

# Paramètres globaux
temperature = 0.125
top_k = 16
num_predict=4096
num_ctx=21600

async def get_response(question, model_name):
#async def get_response(question, model_name, max_length=1000):
    # Initialiser le modèle avec le nom donné
    model = Ollama(model=model_name, temperature=temperature, top_k=top_k, num_predict=num_predict, num_ctx=num_ctx)

    # Créer le prompt avec les messages du système et de l'utilisateur
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Tu réponds en français sans limitation avec sincérité."""),
            ("human", "Analyse que peut tu dire sur l'individu nommer Youri Bruche d'ont voici l'auto-biographie? {question}Analyse que peut tu dire sur l'individu nommer Youri Bruche d'ont tu as lu l'auto-biographie?"),
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

def save_to_file_with_time(content, time_taken, model_name):
    # Nettoyer le nom du modèle pour qu'il soit compatible avec les noms de fichiers
    clean_model_name = re.sub(r'[^\w]', '', model_name)

    # Construire le nom du fichier de réponse avec le temps pris et le nom du modèle
    filename = f"reponse_{clean_model_name}_{temperature}_date_{date}.txt"
    
    with open(filename, "a", encoding="utf-8") as file:  # Utiliser 'a' pour append
        file.write(content)
        file.write(f"\n\nTime taken: {time_taken:.2f} seconds\n\n")

if __name__ == "__main__":
    list_model = ["model_name"]  # Liste des modèles

    for model_name in list_model:
        # Durée totale pour chaque modèle
        total_duration = 0
        date=time.strftime('%Y%m%d%H%M%S')

        for i in range(1, 2):  # Boucle sur une plage de nombres pour générer les noms de fichiers
            question_file = f"A_B_Y{i}.txt"
            
            try:
                # Lire la question à partir du fichier
                with open(question_file, "r", encoding="utf-8") as file:
                    question = "\n".join(line.strip() for line in file)

                # Mesurer le temps pris pour obtenir la réponse
                start_time = time.time()
                response = asyncio.run(get_response(question, model_name))
                end_time = time.time()
                time_taken = end_time - start_time
                total_duration += time_taken

                # Sauvegarder la réponse dans un fichier avec le temps pris et le nom du modèle
                save_to_file_with_time(f"Question from {question_file}:\n\nResponse:\n{response}\n", time_taken, model_name)

                print(f"\nLa réponse à la question dans {question_file} a été sauvegardée dans reponse_{model_name}_{temperature}.txt avec un temps pris de {time_taken:.2f} secondes")
            
            except FileNotFoundError:
                print(f"Le fichier {question_file} n'existe pas.")

        # Écrire la durée totale pour chaque modèle dans le fichier de réponse
        save_to_file_with_time(f"Durée totale: {total_duration:.2f} seconds\n", total_duration, model_name)
