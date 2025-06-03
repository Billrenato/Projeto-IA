import cv2
import pytesseract
from PIL import Image
import numpy as np

# Caminho do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Abrir a imagem
img = cv2.imread(r'C:\Projeto-IA\img\0_imagem.png')

# 🔧 Pré-processamento:
# Converter para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar threshold para binarizar (preto e branco)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Remover ruído (opcional, se necessário)
kernel = np.ones((1, 1), np.uint8)
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
processed = cv2.medianBlur(processed, 3)

# Visualizar o pré-processamento (opcional)
# cv2.imshow("Pre-processada", processed)
# cv2.waitKey(0)

# 🔍 Realizar OCR
texto = pytesseract.image_to_string(processed, lang='por', config='--psm 6')

print("📝 Texto reconhecido:")
print(texto)