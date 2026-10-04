import os
import webbrowser  # <-- Nova importação nativa do Python
from flask import Flask, jsonify, request
from flask_cors import CORS
import Codigo.Complexidade as Complexidade

app = Flask(__name__)
CORS(app)

# Garante que encontra o ficheiro .pkl na mesma pasta que este script Backend.py
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(DIRETORIO_ATUAL, "modelo_rf_dificuldade.pkl")


@app.route("/api/avaliar", methods=["POST"])
def avaliar_texto():
    dados = request.json or {}
    texto_usuario = dados.get("texto", "")

    if not texto_usuario:
        return jsonify({"erro": "Texto inválido ou vazio."}), 400

    try:
        nivel_previsto = Complexidade.prever_dificuldade_texto(
            texto_usuario, CAMINHO_MODELO
        )
        return jsonify({"nivel": str(nivel_previsto)})
    except Exception as e:
        # Imprime o erro exato no terminal do VS Code para conseguirmos diagnosticar
        print(f"\n[ERRO NA PREDIÇÃO]: {e}\n")
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    # O Flask com debug=True executa o código duas vezes ao iniciar (uma para o monitor de arquivos).
    # Esta condição garante que o navegador só abre uma única aba.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # Constrói o caminho e abre o Frontend_2.HTML no navegador padrão
        caminho_html = "file://" + os.path.join(DIRETORIO_ATUAL, "Frontend.HTML")
        webbrowser.open(caminho_html)

    app.run(host="127.0.0.1", port=5000, debug=True)
