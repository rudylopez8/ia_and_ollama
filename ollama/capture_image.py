import cv2

def capture_image():
    # Ouvrir la caméra (par défaut, la caméra 0)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Impossible d'accéder à la caméra.")
        return

    print("Appuyez sur 'Espace' pour capturer une image ou 'Échap' pour quitter.")

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
            break
        elif key == 32:  # Touche Espace
            # Enregistrer l'image
            filename = "captured_image.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image enregistrée sous le nom : {filename}")
            break

    # Libérer les ressources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
