import os
import pandas as pd
import shutil
import textwrap

# Caminhos
IMAGEM_DIR = "dataset/imagens"
LABELS_CSV = "dataset/labels.csv"
NOVO_CSV = "dataset/labels_dividido.csv"
MAX_CHARS = 16  # Máximo de caracteres por frase

# Função para dividir texto longo
def dividir_texto(texto, max_chars):
    # Primeiro tenta dividir por pontuação
    partes = []
    temp = ""
    for palavra in texto.split():
        if len(temp) + len(palavra) + 1 <= max_chars:
            temp += (" " + palavra if temp else palavra)
        else:
            partes.append(temp)
            temp = palavra
    if temp:
        partes.append(temp)
    return partes

# Ler CSV original
df = pd.read_csv(LABELS_CSV)
novas_linhas = []

# Garantir pasta de saída
os.makedirs(IMAGEM_DIR, exist_ok=True)

for idx, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]

    partes = dividir_texto(label, MAX_CHARS)

    for i, trecho in enumerate(partes):
        novo_nome = f"{os.path.splitext(filename)[0]}_{i+1}.png"
        # Copia imagem original com novo nome
        origem = os.path.join(IMAGEM_DIR, filename)
        destino = os.path.join(IMAGEM_DIR, novo_nome)
        shutil.copy(origem, destino)

        novas_linhas.append({"filename": novo_nome, "label": trecho})

# Salva novo CSV
novo_df = pd.DataFrame(novas_linhas)
novo_df.to_csv(NOVO_CSV, index=False)

print(f"[✔] Labels longos divididos e salvos em '{NOVO_CSV}'")
print(f"[✔] Total de novas linhas: {len(novo_df)}")