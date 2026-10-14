import os
import sys
import subprocess
import urllib.request
import os
from Iniciar import iniciar_sistema


def main():
    # Define os diretórios base e o caminho do ambiente virtual (.venv)
    diretorio_base = os.path.dirname(os.path.abspath(__file__))
    pasta_venv = os.path.join(diretorio_base, ".venv")

    # Determina o executável do Python de acordo com o sistema operacional
    if sys.platform == "win32":
        python_venv = os.path.join(pasta_venv, "Scripts", "python.exe")
    else:
        python_venv = os.path.join(pasta_venv, "bin", "python")

    # Cria o ambiente virtual caso ele ainda não exista
    if not os.path.exists(pasta_venv):
        print("Criando o ambiente virtual (.venv)...")
        subprocess.run([sys.executable, "-m", "venv", pasta_venv], check=True)

    # Verifica se o script está rodando no Python do sistema
    caminho_atual = os.path.normcase(os.path.abspath(sys.executable))
    caminho_esperado = os.path.normcase(os.path.abspath(python_venv))

    if caminho_atual != caminho_esperado:
        print("Ativando ambiente virtual e reiniciando o processo...")
        subprocess.run([python_venv] + sys.argv)
        sys.exit(0)

    # Lista de dependências do projeto
    pacotes = [
        "wordfreq",
        "spacy",
        "pandas",
        "scikit-learn",
        "joblib",
        "flask",
        "flask-cors",
        "nltk",
        "unimorph",
        "numpy",
    ]

    # Atualiza o pip e instala as dependências necessárias
    print("Instalando dependências do projeto...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True
    )
    subprocess.run([sys.executable, "-m", "pip", "install"] + pacotes, check=True)

    # Verifica e baixa os modelos do spaCy (sm e lg) apenas se não estiverem instalados
    import spacy.util

    modelos_spacy = ["pt_core_news_sm", "pt_core_news_lg"]
    for modelo in modelos_spacy:
        if not spacy.util.is_package(modelo):
            print(f"Baixando modelo do spaCy ({modelo})...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", modelo], check=True
            )
        else:
            print(f"Modelo '{modelo}' já está instalado.")

    # Baixa a tabela morfológica do UniMorph para Português
    caminho_unimorph_pt = os.path.join(diretorio_base, "unimorph_pt.tsv")
    if not os.path.exists(caminho_unimorph_pt):
        print("Baixando dados do UniMorph para português...")
        url_unimorph = "https://raw.githubusercontent.com/unimorph/por/master/por"
        urllib.request.urlretrieve(url_unimorph, caminho_unimorph_pt)

    import nltk

    # Configura o diretório isolado para os dados do NLTK dentro do venv e faz o download
    caminho_nltk_data = os.path.join(pasta_venv, "nltk_data")
    os.makedirs(caminho_nltk_data, exist_ok=True)

    nltk.data.path.append(caminho_nltk_data)

    print("Baixando dados do NLTK...")
    nltk.download("wordnet", download_dir=caminho_nltk_data, quiet=True)
    nltk.download("omw-1.4", download_dir=caminho_nltk_data, quiet=True)
    nltk.download("omw-2.0", download_dir=caminho_nltk_data, quiet=True)


if __name__ == "__main__":
    main()
    iniciar_sistema()
