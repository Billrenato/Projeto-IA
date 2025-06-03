import cv2
import os
import numpy as np
from tensorflow import keras

# Diretório das imagens
diretorio = 'C:\\projeto-ia2\\img'

# Lista para armazenar as imagens e rótulos
imagens = []
labels = []

# Carregar imagens e rótulos
for arquivo in os.listdir(diretorio):
    if arquivo.endswith('.png'):
        imagem = cv2.imread(os.path.join(diretorio, arquivo))
        imagens.append(imagem)
        # Supondo que os arquivos sejam nomeados como "classe_imagem.png"
        label = int(arquivo.split('_')[0])  # Ajuste isso de acordo com o nome dos seus arquivos
        labels.append(label)

# Redimensionar imagens para um tamanho padrão
tamanho = (128, 128)
imagens_redimensionadas = [cv2.resize(imagem, tamanho) for imagem in imagens]

# Converter imagens para escala de cinza
imagens_cinza = [cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) for imagem in imagens_redimensionadas]

# Normalizar as imagens
imagens_normalizadas = [imagem / 255.0 for imagem in imagens_cinza]

# Converter as listas para arrays numpy
imagens_array = np.array(imagens_normalizadas)
labels_array = np.array(labels)

# Adicionar uma dimensão para o canal de cor (já que as imagens estão em escala de cinza)
imagens_array = imagens_array[..., np.newaxis]

# Dividir os dados em conjuntos de treinamento e teste
train_size = int(len(imagens_array) * 0.8)
train_images = imagens_array[:train_size]
train_labels = labels_array[:train_size]
test_images = imagens_array[train_size:]
test_labels = labels_array[train_size:]

# Definir o modelo
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(max(labels_array) + 1, activation='softmax')  # Ajuste o número de classes
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images, train_labels, epochs=10)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Acurácia do modelo: {test_acc:.2f}')
