import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Dense,
    Bidirectional, LSTM, Lambda
)
from keras.preprocessing.sequence import pad_sequences
import json

# 🛠️ Configurações
IMG_DIR = "C:\\projeto-Ia-tensorflow\\imagens"
LABELS_CSV = "C:\\projeto-Ia-tensorflow\\labels.csv"
IMG_WIDTH, IMG_HEIGHT = 128, 32
MAX_LABEL_LEN = 128
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# 1. 📄 Leitura do CSV
df = pd.read_csv(LABELS_CSV)
if df.empty:
    raise ValueError("❌ O arquivo de labels está vazio!")

# 2. 🔠 Criação do vocabulário
all_text = ''.join(df['label'].astype(str).values)
vocab = sorted(set(all_text))
char_to_num = {c: i + 1 for i, c in enumerate(vocab)}  # 0 reservado para blank
num_to_char = {i: c for c, i in char_to_num.items()}
VOCAB_SIZE = len(char_to_num) + 1  # +1 para blank

# 3. 🔧 Funções auxiliares
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def text_to_sequence(text):
    return [char_to_num[c] for c in text if c in char_to_num]

# 4. 🖼️ Pré-processamento de imagens e rótulos
images, labels, label_lengths = [], [], []
for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row['filename'])
    if not os.path.exists(img_path):
        continue
    img = preprocess_image(img_path)
    seq = text_to_sequence(row['label'])
    if len(seq) == 0: continue  # pular rótulos vazios
    images.append(img)
    labels.append(seq)
    label_lengths.append(len(seq))

X = np.array(images)
y = pad_sequences(labels, maxlen=MAX_LABEL_LEN, padding='post')
label_lengths = np.array(label_lengths)
input_lengths = np.ones(len(X)) * (IMG_WIDTH // 4)

# 5. 📦 Dataset com tf.data
def generator():
    for i in range(len(X)):
        yield {
            "image": X[i],
            "label": y[i],
            "input_length": int(input_lengths[i]),
            "label_length": int(label_lengths[i])
        }

output_signature = {
    "image": tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
    "label": tf.TensorSpec(shape=(MAX_LABEL_LEN,), dtype=tf.int32),
    "input_length": tf.TensorSpec(shape=(), dtype=tf.int32),
    "label_length": tf.TensorSpec(shape=(), dtype=tf.int32),
}

dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
dataset = dataset.shuffle(100).batch(BATCH_SIZE).repeat().prefetch(AUTOTUNE)

# 6. 🧠 Modelo OCR com CTC
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image')
labels = Input(name='label', shape=(MAX_LABEL_LEN,), dtype='int32')
input_len = Input(name='input_length', shape=(), dtype='int32')
label_len = Input(name='label_length', shape=(), dtype='int32')

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Cálculo do shape para Reshape
new_w = IMG_WIDTH // 4
new_h = (IMG_HEIGHT // 4) * 64
x = Reshape(target_shape=(new_w, new_h))(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
y_pred = Dense(VOCAB_SIZE, activation='softmax', name='y_pred')(x)

# 7. 🔣 CTC Loss Layer
def ctc_lambda_func(args):
    y_pred, labels, input_len, label_len = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')(
    [y_pred, labels, input_len, label_len])

model = Model(inputs=[input_img, labels, input_len, label_len], outputs=ctc_loss)
model.compile(optimizer='adam')

# 8. 💾 Salvar vocabulário
with open("char_to_num.json", "w", encoding="utf-8") as f:
    json.dump(char_to_num, f, ensure_ascii=False, indent=2)

# 9. 🏋️‍♂️ Treinamento
steps_per_epoch = len(X) // BATCH_SIZE

model.fit(
    dataset.map(lambda x: (
        [x["image"], x["label"], x["input_length"], x["label_length"]],
        tf.zeros((BATCH_SIZE,))  # dummy target para ctc_loss
    )),
    epochs=20,
    steps_per_epoch=steps_per_epoch
)

# 10. 💽 Salvar modelo
model.save("ocr_ctc_model.keras")
print("✅ Modelo treinado e salvo como ocr_ctc_model.keras")