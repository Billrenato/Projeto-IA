import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# Configurações
IMG_WIDTH = 128
IMG_HEIGHT = 32
BATCH_SIZE = 1
AUTOTUNE = tf.data.AUTOTUNE
MAX_LABEL_LENGTH = 12  # Limite fixo de caracteres nas labels

# Carrega e prepara o CSV
df = pd.read_csv("dataset/labels.csv")  # ou "labels.csv", conforme o nome real
df["label"] = df["label"].fillna("")  # Evita erro de NaN
df["label"] = df["label"].str.slice(0, MAX_LABEL_LENGTH)

# Cria vocabulário
all_text = "".join(df["label"].astype(str).values)
vocab = sorted(set(all_text))
char_to_num = keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True)

# Preprocessamento das imagens e conversão das labels
def encode_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)

    label = tf.strings.unicode_split(label, input_encoding='UTF-8')
    label = char_to_num(label)
    return {"image": img, "label": label}

# Cria o Dataset
def create_dataset(df):
    paths = [os.path.join("dataset/images", fname) for fname in df['filename']]
    labels = df['label'].tolist()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(encode_sample, num_parallel_calls=AUTOTUNE)

    def format_batch(batch):
        return ({"image": batch["image"], "label": batch["label"]}, None)

    dataset = dataset.batch(BATCH_SIZE).map(format_batch).prefetch(AUTOTUNE)
    return dataset

train_dataset = create_dataset(df)

# Camada personalizada CTC
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        input_length = tf.ones((batch_size, 1), dtype="int64") * tf.cast(input_length, dtype="int64")
        label_length = tf.ones((batch_size, 1), dtype="int64") * tf.cast(label_length, dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# Modelo OCR com CTC
def build_model():
    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="int64")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax")(x)
    output = CTCLayer()(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output)
    return model

# Compila o modelo
model = build_model()
model.compile(optimizer=keras.optimizers.Adam())

# Resumo
model.summary()

# Treinamento
model.fit(train_dataset, epochs=20)

# Salva modelo
model.save("modelo_ocr.keras")