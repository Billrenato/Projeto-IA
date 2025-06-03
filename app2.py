import easyocr
from PIL import Image, ImageEnhance, ImageFilter

# Criar o leitor de OCR
reader = easyocr.Reader(['pt'])  # Português

# Caminho da imagem
caminho_imagem = r'C:\Projeto-IA\img\1_imagem.png'

# Pré-processamento de imagem
img = Image.open(caminho_imagem)
img = img.convert('L')  # Converter para escala de cinza
img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=3))  # Aplicar filtro de nitidez
img.save('imagem_pre_processada.png')

# Ler a imagem pré-processada
resultado = reader.readtext('imagem_pre_processada.png', detail=0)

# Mostrar o resultado
print("📝 Texto reconhecido:")
for linha in resultado:
    print(linha)