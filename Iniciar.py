import os
import webbrowser
from Web.Backend import app, CAMINHO_FRONTEND

def iniciar_sistema():
    # Evita abrir duas abas quando o Flask recarrega no modo debug
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        caminho_html = "file://" + CAMINHO_FRONTEND
        print(f"Abrindo interface frontend: {caminho_html}")
        webbrowser.open(caminho_html)

    print("Iniciando o servidor Flask...")
    app.run(host="127.0.0.1", port=5000, debug=True)

if __name__ == "__main__":
    iniciar_sistema()