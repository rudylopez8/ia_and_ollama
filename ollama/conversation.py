import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

conversation_history = []  # Variable globale pour stocker l'historique de la conversation
num_predict = 2048
num_ctx = 4096

async def get_response(question, model_name, temperature, top_k, stop_event):
    # Initialiser le modèle avec les paramètres spécifiés
    model = Ollama(model=model_name, temperature=temperature, top_k=top_k, num_predict=num_predict, num_ctx=num_ctx)

    # Créer le prompt avec les messages du système et de l'utilisateur
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu réponds en français. Réponds avec logique et rigueur en français."),
        ("human", "{question}"),
    ])

    # Construire la chaîne de traitement
    chain = prompt | model

    response = ""
    async for chunk in chain.astream({"question": question}):
        if stop_event.is_set():
            print("\n\nInterruption détectée. Arrêt de la génération.\n")
            return response  # Retourner la réponse partielle obtenue avant l'interruption
        print(chunk, end="", flush=True)  # Afficher chaque chunk immédiatement
        response += chunk

    return response

async def check_for_interruption(stop_event):
    # Fonction d'écoute pour l'interruption
    with ThreadPoolExecutor() as executor:
        await asyncio.get_event_loop().run_in_executor(executor, input, "\nAppuyez sur 'Enter' pour interrompre la génération de la réponse...\n")
    stop_event.set()

async def ask_question_and_get_response(model_name, temperature, top_k):
    while True:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if question.lower() == "exit":
            break

        # Ajouter la question à l'historique de la conversation
        conversation_history.append(f"User: {question}")

        # Créer un événement de type asyncio pour signaler l'interruption
        stop_event = asyncio.Event()

        # Exécuter les tâches de génération de réponse et de vérification d'interruption en parallèle
        response_task = asyncio.create_task(get_response("\n".join(conversation_history), model_name, temperature, top_k, stop_event))
        interruption_task = asyncio.create_task(check_for_interruption(stop_event))

        # Attendre la fin de l'une ou l'autre des tâches
        done, pending = await asyncio.wait([response_task, interruption_task], return_when=asyncio.FIRST_COMPLETED)

        # Si la tâche de réponse est terminée, afficher la réponse complète
        if response_task in done:
            response = response_task.result()
            print("\n\nRéponse :\n", response)

            # Ajouter la réponse de l'IA à l'historique de la conversation
            conversation_history.append(f"AI: {response}")

        # Si la génération a été interrompue, annuler la tâche restante
        for task in pending:
            task.cancel()

        # Marquer la fin du cycle et s'assurer que la prochaine question peut être posée
        print("\n\n--- Prêt pour la prochaine question ---\n")

async def main():
    # Liste des modèles disponibles
    list_model = [
        "gemma2:2b", "phi3.5", "deepseek-v2", "llama3.1", "mistral", "mistral-nemo", "codestral"]

    # Afficher les modèles disponibles avec leurs indices
    print("Modèles disponibles :")
    for i, model in enumerate(list_model, start=1):
        print(f"{i}. {model}")

    # Demander à l'utilisateur de choisir un modèle par son numéro
    while True:
        try:
            model_index = int(input("Entrez le numéro du modèle souhaité : ")) - 1
            if 0 <= model_index < len(list_model):
                break
            else:
                print("Numéro invalide. Veuillez entrer un nombre correspondant à un modèle.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")

    # Obtenir le nom du modèle choisi
    model_name = list_model[model_index]

    # Demander les paramètres temperature et top_k
    temperature = float(input("Entrez la valeur de la température (0-1) : "))
    top_k = int(input("Entrez la valeur de top_k : "))

    # Lancer le cycle de questions et réponses
    await ask_question_and_get_response(model_name, temperature, top_k)

if __name__ == "__main__":
    asyncio.run(main())
