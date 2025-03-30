import ollama
from PIL import Image
import os

def convert_to_png(image_path):
    """
    Vérifie si l'image est au format PNG, sinon la convertit en PNG.
    Renvoie le chemin de l'image au format PNG.
    """
    img = Image.open(image_path)
    if img.format != 'PNG':
        # Si l'image n'est pas en PNG, on la convertit
        png_path = os.path.splitext(image_path)[0] + ".png"
        img.save(png_path, format="PNG")
        print(f"L'image {image_path} a été convertie en {png_path}")
        return png_path
    return image_path

for _ in range(1, 10):
    image_path = './img2.png'  # Remplace par le chemin de ton image
    
    # Convertir l'image en PNG si nécessaire
    compatible_image_path = convert_to_png(image_path)

    # Appel à la fonction chat avec l'image au bon format
    res = ollama.chat(
        model="minicpm-v",
        messages=[
            {
                'role': 'user',
                'content': 'Décrit cette image en français',
                'images': [compatible_image_path]
            }
        ]
    )

    print(res['message']['content'])
