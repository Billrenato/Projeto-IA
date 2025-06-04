import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Defina as configurações
IMG_WIDTH = 128
IMG_HEIGHT = 32

# Defina a classe CTCLayer novamente
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

# Carrega o modelo salvo
model = keras.models.load_model("modelo_ocr.keras", custom_objects={"CTCLayer": CTCLayer})

# Crie um modelo de predição
x = model.input[0]
y_pred = model.layers[-2].output
prediction_model = keras.Model(inputs=x, outputs=y_pred)

# Defina as variáveis char_to_num e num_to_char novamente
df = pd.read_csv("dataset/labels_dividido.csv")
all_text = "".join(df['label'].values)
vocab = sorted(set(all_text))
char_to_num = keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True)

# Função para decodificar a saída do modelo
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results.numpy():
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Função para testar uma imagem
def test_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img = tf.expand_dims(img, axis=0)
    pred = prediction_model.predict(img)
    pred_text = decode_batch_predictions(pred)
    return pred_text[0]

# Testa uma imagem
img_path = "img001_1.png"
print("Texto reconhecido:", test_image(img_path))
