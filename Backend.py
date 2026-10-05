import os
import webbrowser
from flask import Flask, jsonify, request
from flask_cors import CORS
import Codigo.Complexidade as Complexidade
import Codigo.Mutacoes.TrocaPalavra as TrocaPalavra

app = Flask(__name__)
CORS(app)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(DIRETORIO_ATUAL, "modelo_rf_dificuldade.pkl")


# --- ROTA 1: Avaliar Complexidade do Texto ---
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
        print(f"\n[ERRO NA PREDIÇÃO]: {e}\n")
        return jsonify({"erro": str(e)}), 500


# --- ROTA 2: Trocar Palavras (Sinônimos/Antônimos) ---
@app.route("/api/troca-palavra", methods=["POST"])
def trocar_palavras():
    dados = request.json or {}
    texto_usuario = dados.get("texto", "")
    modo = dados.get("modo", "sinonimo")
    prob_troca = float(dados.get("prob_troca", 0.5))

    if not texto_usuario:
        return jsonify({"erro": "Texto inválido ou vazio."}), 400

    try:
        texto_modificado = TrocaPalavra.realizar_data_augmentation(
            texto_usuario, prob_troca=prob_troca, modo=modo
        )
        return jsonify({"texto_modificado": texto_modificado})
    except Exception as e:
        print(f"\n[ERRO NA TROCA DE PALAVRAS]: {e}\n")
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        caminho_html = "file://" + os.path.join(DIRETORIO_ATUAL, "Frontend.HTML")
        webbrowser.open(caminho_html)

    app.run(host="127.0.0.1", port=5000, debug=True)
