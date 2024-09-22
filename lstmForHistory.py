import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import string

# Lire le fichier de données
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Nettoyage du texte
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Création des séquences d'entraînement
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Remplissage des séquences pour qu'elles aient la même longueur
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Création des prédictions (cible)
X = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)
# Définition du modèle
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200, return_sequences=True))

model.add(Dropout(0.1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Affichage du modèle
model.summary()
# Entraînement du modèle
history = model.fit(X, y, epochs=10, verbose=1)
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + predicted_word
    return seed_text

# Test de génération de texte
seed_text = "Le petit lapin"
next_words = 10
print(generate_text(seed_text, next_words, max_sequence_len))

