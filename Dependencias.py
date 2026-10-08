import os
import sys
import subprocess

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

    # Verifica se o script está rodando no Python do sistema. 
    # Se sim, reinicia o próprio script utilizando o Python do ambiente virtual.
    caminho_atual = os.path.normcase(os.path.abspath(sys.executable))
    caminho_esperado = os.path.normcase(os.path.abspath(python_venv))
    
    if caminho_atual != caminho_esperado:
        print("Ativando ambiente virtual e reiniciando o processo...")
        subprocess.run([python_venv] + sys.argv)
        sys.exit(0)

    # --- A PARTIR DAQUI O SCRIPT RODA EXCLUSIVAMENTE DENTRO DO VENV ---

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
    ]

    # Atualiza o pip e instala as dependências necessárias
    print("Instalando dependências do projeto...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install"] + pacotes, check=True)

    # Baixa o modelo de linguagem em português do spaCy
    print("Baixando modelo do spaCy (pt_core_news_sm)...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"], check=True)

    # O import do nltk ocorre aqui embaixo propositalmente para evitar ModuleNotFoundError 
    # na primeira execução, quando o pacote ainda não estava instalado.
    import nltk

    # Configura o diretório isolado para os dados do NLTK dentro do venv e faz o download
    caminho_nltk_data = os.path.join(pasta_venv, "nltk_data")
    os.makedirs(caminho_nltk_data, exist_ok=True)
    
    nltk.data.path.append(caminho_nltk_data)
    
    print("Baixando dados do NLTK...")
    nltk.download("wordnet", download_dir=caminho_nltk_data, quiet=True)
    nltk.download("omw-2.0", download_dir=caminho_nltk_data, quiet=True)

    # Inicia o servidor backend
    caminho_backend = os.path.join(diretorio_base, "Backend.py")
    print("Iniciando o Backend.py...")
    subprocess.run([sys.executable, caminho_backend])

if __name__ == "__main__":
    main()