------------------------------------------------------------------------
PRÉ-REQUISITOS
------------------------------------------------------------------------
* Python 3.8 ou superior instalado.
* Linux (Debian/Ubuntu): Certifique-se de ter o pacote 'python3-venv'
  instalado no sistema antes de rodar pela primeira vez:

  sudo apt update && sudo apt install python3-venv


------------------------------------------------------------------------
COMO EXECUTAR
------------------------------------------------------------------------
Abra o terminal na pasta raiz do projeto e utilize os comandos abaixo.

[A] PRIMEIRA VEZ RODANDO O PROJETO
O script dependencias.py cuida de todo o processo automaticamente:
cria o ambiente virtual (.venv), ativa o ambiente internamente, instala
todas as dependências, baixa os modelos (spaCy/NLTK) e inicia o backend.

Windows:
  python dependencias.py

Linux / macOS:
  python3 dependencias.py


[B] NAS EXECUÇÕES SEGUINTES
Como a .venv já estará configurada, você pode escolher uma das opções:

Opção 1:
  Execute diretamente o servidor usando o Python do ambiente virtual:

  Windows:
    .venv\Scripts\python Backend.py

  Linux / macOS:
    .venv/bin/python Backend.py

Opção 2:
  Rode novamente 'python dependencias.py' (ou 'python3 dependencias.py').
  O script detectará a .venv existente, ativará o ambiente e iniciará o backend.

Opção 3:
  Ative o ambiente virtual no seu terminal e execute o arquivo normalmente:

  Windows:
    .venv\Scripts\activate
    python Backend.py

  Linux / macOS:
    source .venv/bin/activate
    python Backend.py


[C] RETREINAR O MODELO (EXTRAS)
Caso queira retreinar o algoritmo de predição utilizando o ambiente isolado:

Windows:
  .venv\Scripts\python Treinamento.py

Linux / macOS:
  .venv/bin/python Treinamento.py


------------------------------------------------------------------------
TECNOLOGIAS UTILIZADAS
------------------------------------------------------------------------
Flask & Flask-CORS      : Framework web para a API.
spaCy (pt_core_news_sm) : Processamento de Linguagem Natural em Português.
NLTK (wordnet, omw-2.0) : Processamento e bases de dados lexicais.
Pandas & Scikit-Learn   : Manipulação de dados e modelos de ML.
Joblib                  : Carregamento e salvamento de modelos treinados.
Wordfreq                : Análise de frequência de palavras.
Transformers            : Manipulação de texto.


pontos que foram considerandos:
    Diversidade no vocabulario.
    --Tamanho do texto--
    Palavra ser comum na lingua
    Tamanho da palavra
    Canonicidade da palavra
    ponderação de clases linguisticas
    conplexidades de uma oração com base na complexidade das palavras contidas nela
    quantidade de vogais em porporção
    sobordinação
    voz passiva
    quantidade de frases
    quatidade de dialogos
    quantidade de conectivos    