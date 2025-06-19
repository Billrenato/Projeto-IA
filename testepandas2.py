import pandas as pd
import os

def remover_linhas_sem_label_e_imagens(caminho_csv: str, pasta_imagens: str, caminho_saida: str = None):
    """
    Remove linhas com labels ausentes (NaN ou string vazia) de um CSV
    e deleta as imagens correspondentes.

    Parâmetros:
        caminho_csv (str): Caminho para o CSV original.
        pasta_imagens (str): Pasta onde estão as imagens.
        caminho_saida (str): Novo caminho para salvar o CSV limpo. Se None, sobrescreve o original.
    """
    df = pd.read_csv(caminho_csv)

    # Filtra linhas inválidas (sem label)
    linhas_invalidas = df[df['label'].isna() | (df['label'].str.strip() == "")]
    
    # Deleta as imagens dessas linhas
    for filename in linhas_invalidas['filename']:
        caminho_imagem = os.path.join(pasta_imagens, filename)
        if os.path.exists(caminho_imagem):
            os.remove(caminho_imagem)
            print(f"[REMOVIDO] {caminho_imagem}")
        else:
            print(f"[AVISO] Imagem não encontrada: {caminho_imagem}")

    # Mantém apenas as linhas válidas
    df_limpo = df.drop(index=linhas_invalidas.index)

    # Salva o novo CSV
    if caminho_saida is None:
        caminho_saida = caminho_csv  # sobrescreve

    df_limpo.to_csv(caminho_saida, index=False)
    print(f"\nCSV limpo salvo com {len(df_limpo)} linhas em: {caminho_saida}")

# Exemplo de uso:
remover_linhas_sem_label_e_imagens("dataset/labels.csv", "dataset/images")