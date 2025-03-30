import cv2
import ollama

def capture_image():
    """
    Capture une image à partir de la caméra intégrée et l'enregistre au format JPG.
    """
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Impossible d'accéder à la caméra.")
        return None

    print("Appuyez sur 'Espace' pour capturer une image ou 'Échap' pour quitter.")

    filename = "captured_image.jpg"
    while True:
        # Lire une image de la caméra
        ret, frame = camera.read()

        if not ret:
            print("Erreur lors de la capture de l'image.")
            break

        # Afficher l'image capturée dans une fenêtre
        cv2.imshow("Capture d'image", frame)

        # Vérifier les touches pressées
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Touche Échap
            print("Capture annulée.")
            filename = None
            break
        elif key == 32:  # Touche Espace
            # Enregistrer l'image
            cv2.imwrite(filename, frame)
            print(f"Image enregistrée sous le nom : {filename}")
            break

    # Libérer les ressources
    camera.release()
    cv2.destroyAllWindows()
    return filename


def analyze_image_with_ollama(image_path):
    """
    Utilise le modèle d'IA d'Ollama pour analyser une image et en fournir une description.
    """
    if not image_path:
        print("Aucune image à analyser.")
        return

    try:
        response = ollama.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': 'What is in this image?',
                'images': [image_path]
            }]
        )
        print("Description de l'image :")
        print(response)
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'image : {e}")


if __name__ == "__main__":
    # Capture l'image
    image_file = capture_image()

    # Analyse l'image si elle a été capturée
    analyze_image_with_ollama(image_file)
