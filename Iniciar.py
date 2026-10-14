import os
import sys
import subprocess
import webbrowser


def garantir_venv():
    """
    Garante que o script seja executado utilizando o interpretador Python
    do ambiente virtual (.venv). Caso o ambiente não exista, informa a
    necessidade de executar o script de instalação das dependências.
    """
    diretorio_base = os.path.dirname(os.path.abspath(__file__))
    pasta_venv = os.path.join(diretorio_base, ".venv")

    if sys.platform == "win32":
        python_venv = os.path.join(pasta_venv, "Scripts", "python.exe")
    else:
        python_venv = os.path.join(pasta_venv, "bin", "python")

    if not os.path.exists(python_venv):
        print("[ERRO] O ambiente virtual (.venv) não foi encontrado.")
        print(
            "Execute o script 'Dependencias.py' primeiro para criar o ambiente e instalar as bibliotecas."
        )
        sys.exit(1)

    caminho_atual = os.path.normcase(os.path.abspath(sys.executable))
    caminho_esperado = os.path.normcase(os.path.abspath(python_venv))

    if caminho_atual != caminho_esperado:
        subprocess.run([python_venv, __file__] + sys.argv[1:])
        sys.exit(0)


def iniciar_sistema():
    """
    Inicializa a aplicação, garantindo a execução no ambiente virtual e
    configurando o diretório de dados do NLTK.
    """
    garantir_venv()

    diretorio_base = os.path.dirname(os.path.abspath(__file__))
    pasta_nltk_data = os.path.join(diretorio_base, ".venv", "nltk_data")
    if os.path.exists(pasta_nltk_data):
        import nltk

        nltk.data.path.append(pasta_nltk_data)

    from Web.Backend import app, CAMINHO_FRONTEND

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        caminho_html = "file://" + os.path.abspath(CAMINHO_FRONTEND)
        print(f"\nAbrindo aplicação no navegador padrão: {caminho_html}\n")
        webbrowser.open(caminho_html)

    print("Iniciando o servidor Flask...")
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    iniciar_sistema()
