


Na pasta raiz do projeto:
python -m venv .venv
.venv\Scripts\activate
pip install wordfreq spacy pandas scikit-learn joblib flask flask-cors
python -m spacy download pt_core_news_sm
python Codigo/main.py
python Backend.py

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