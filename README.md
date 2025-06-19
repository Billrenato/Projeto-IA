# 📄 Reconhecimento Óptico de Caracteres (OCR) com TensorFlow


Este projeto tem como objetivo desenvolver um modelo de Reconhecimento Óptico de Caracteres (OCR) utilizando TensorFlow. O modelo é capaz de identificar e transcrever textos presentes em imagens, sendo ideal para aplicações como leitura de documentos, automação de entrada de dados e digitalização de informações.

# 🧠 Visão Geral

🔍 Arquitetura Híbrida: Combinação de redes neurais convolucionais (CNN) para extração de características e recorrentes bidirecionais (BiLSTM) para modelagem de sequências.

🔗 CTC (Connectionist Temporal Classification): Utilizada para alinhar sequências de entrada (imagens) com saídas de tamanho variável (texto), sem necessidade de alinhamento prévio entre caracteres.

📄 Dataset Personalizado: O modelo é treinado com um conjunto de imagens e seus respectivos rótulos textuais informados em um arquivo CSV.

🔧 Pipeline Completo: Inclui leitura dos dados, pré-processamento, codificação dos textos, treinamento e salvamento do modelo para inferência futura.


# 📁 Estrutura de Diretórios

projeto-Ia-tensorflow/
│
├── dataset/
│   ├── imagens/
│   │   ├── img001.png
│   │   ├── img002.png
│   │   └── ...
│   └── labels.csv
│
├── modelo_ocr.keras      # Modelo treinado salvo
├── script_treino.py      # Script principal de treinamento
└── README.md             # Este arquivo

# 🔧 Requisitos

    Python 3.11 ou superior

    TensorFlow 2.15 ou superior

    pandas

    numpy



# 🧩 Arquitetura do Modelo

🔸 Camadas Convolucionais (CNN):
Extração de características espaciais da imagem.

🔸 Camadas Bidirecionais LSTM:
Modelagem da sequência de características, permitindo capturar dependências anteriores e futuras no texto.

🔸 Camada StringLookup:
Conversão dos caracteres para IDs numéricos e vice-versa.

🔸 Camada CTCLoss (Customizada):
Implementação da função de perda baseada em CTC, que permite o treinamento sem alinhamento explícito entre a imagem e o texto.

⚙️ Configurações e Restrições
🏷️ Limite de Caracteres: Labels são truncadas para no máximo 16 caracteres.

📐 Resolução das Imagens: Todas são redimensionadas para 128x32 pixels (escala de cinza).

📦 Batch Size: Definido como 1, visando facilitar o debug e experimentos em datasets pequenos.

🔗 Referências
TensorFlow OCR com CTC (Exemplo oficial da Keras)

CTC Loss - Baidu Research