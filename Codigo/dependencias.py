# Descobre o caminho raiz do projeto e aponta para dentro do .venv
# (Pega a pasta atual 'Codigo/', volta um nível e entra em '.venv/nltk_data')
diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_nltk_data = os.path.join(diretorio_base, '.venv', 'nltk_data')

# Cria a pasta caso ela ainda não exista
os.makedirs(caminho_nltk_data, exist_ok=True)

# 1. Ensina o seu script a procurar os dados nessa pasta específica
nltk.data.path.append(caminho_nltk_data)

# 2. Força o download para dentro da pasta do .venv
# (Você pode comentar com '#' estas duas linhas após rodar a primeira vez com sucesso)
nltk.download('wordnet', download_dir=caminho_nltk_data)
nltk.download('omw-1.4', download_dir=caminho_nltk_data)
nltk.download('omw-2.0', download_dir=caminho_nltk_data)