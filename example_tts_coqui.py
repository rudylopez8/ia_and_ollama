from TTS.api import TTS
import torchaudio as ta
import torch
import numpy as np

# 1. Détection du device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialisation du modèle Coqui-TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available()).to(device)

# 3. Texte à synthétiser
text = "Bonjour."

# 4. Clonage via audio prompt
wav_clone = tts.tts(
    text=text,
    speaker_wav="source.wav",
    language="fr"
)

# 5. Conversion de list -> ndarray -> tensor 2D
arr = np.array(wav_clone, dtype=np.float32)
tensor = torch.from_numpy(arr)
if tensor.ndim == 1:
    tensor = tensor.unsqueeze(0)  # shape devient [1, n_samples]

# 6. Sauvegarde audio
sr = tts.synthesizer.output_sample_rate
ta.save("coqui_clone.wav", tensor, sr)
print("✅ Fichier généré : coqui_clone.wav")
