import os
import webbrowser
from flask import Flask, jsonify, request
from flask_cors import CORS
import Codigo.Complexidade as Complexidade
import Codigo.Mutacoes.TrocaPalavra as TrocaPalavra
import Codigo.Mutacoes.TrocaGenero as TrocaGenero
import Codigo.Mutacoes.Parafraseio as Parafraseio

app = Flask(__name__)
CORS(app)

# Diretório atual da pasta 'Web'
DIRETORIO_WEB = os.path.dirname(os.path.abspath(__file__))

# Sobe 1 nível para a pasta raiz ('Projeto_Caed')
DIRETORIO_RAIZ = os.path.abspath(os.path.join(DIRETORIO_WEB, ".."))

# Caminhos exatos para os arquivos
CAMINHO_MODELO = os.path.join(DIRETORIO_RAIZ, "modelo_rf_dificuldade.pkl")
CAMINHO_FRONTEND = os.path.join(DIRETORIO_WEB, "Frontend.HTML")


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


# --- ROTA 2: Trocar Palavras ---
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


# --- ROTA 3: Trocar Gênero Gramatical ---
@app.route("/api/troca-genero", methods=["POST"])
def trocar_genero():
    dados = request.json or {}
    texto_usuario = dados.get("texto", "")

    if not texto_usuario:
        return jsonify({"erro": "Texto inválido ou vazio."}), 400

    try:
        # Removido o prob_troca
        texto_modificado = TrocaGenero.trocar_genero(texto_usuario)
        return jsonify({"texto_modificado": texto_modificado})
    except Exception as e:
        print(f"\n[ERRO NA TROCA DE GÊNERO]: {e}\n")
        return jsonify({"erro": str(e)}), 500


# --- ROTA 4: Parafrasear Texto ---
@app.route("/api/parafrasear", methods=["POST"])
def parafrasear_texto():
    dados = request.json or {}
    texto_usuario = dados.get("texto", "")

    if not texto_usuario:
        return jsonify({"erro": "Texto inválido ou vazio."}), 400

    try:
        # Removido o prob_troca
        texto_modificado = Parafraseio.realizar_parafraseio(texto_usuario)
        return jsonify({"texto_modificado": texto_modificado})
    except Exception as e:
        print(f"\n[ERRO NO PARAFRASEIO]: {e}\n")
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        caminho_html = "file://" + CAMINHO_FRONTEND
        webbrowser.open(caminho_html)

    app.run(host="127.0.0.1", port=5000, debug=True)
