import spacy
import pandas as pd
import joblib
import numpy as np

# Importação dos módulos de cálculo
import Codigo.Lexico as Lexico
import Codigo.Sintaxe as Sintaxe
import Codigo.Estilo as Estilo

# Carregar o modelo de NLP apenas uma vez na inicialização do sistema
print("A carregar modelo de Linguagem Natural (spaCy)...")
nlp = spacy.load("pt_core_news_sm")
print("Modelo carregado com sucesso!\n")


def extrair_features(texto: str) -> dict:
    """
    Extrai as métricas dos três módulos (Léxico, Sintaxe, Estilo)
    para transformar o texto num vetor de características (features).
    """
    doc = nlp(texto)

    score_lexico, _ = Lexico.analisar_dificuldade_texto(texto, nlp)
    fator_sintatico = Sintaxe.calcular_fator_proporcao(doc)
    profundidade = Sintaxe.calcular_profundidade_media_arvore(doc)
    voz_passiva = Sintaxe.calcular_proporcao_voz_passiva(doc)
    dens_pontuacao = Sintaxe.calcular_densidade_pontuacao(doc)
    dens_dialogos = Sintaxe.calcular_densidade_dialogos(doc)
    dens_conectivos = Sintaxe.calcular_densidade_conectivos(doc)
    ttr = Estilo.calcular_ttr(texto)

    # Retornamos apenas as features úteis para o modelo
    return {
        "score_lexico": score_lexico,
        "fator_sintatico": fator_sintatico,
        "profundidade_arvore": profundidade,
        "proporcao_voz_passiva": voz_passiva,
        "densidade_pontuacao": dens_pontuacao,
        "densidade_dialogos": dens_dialogos,
        "densidade_conectivos": dens_conectivos,
        "ttr_diversidade": ttr,
    }


def construir_dataset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a extração de features a todas as linhas do DataFrame."""
    lista_features = []
    for texto in df["texto"]:
        features = extrair_features(texto)
        lista_features.append(features)

    df_features = pd.DataFrame(lista_features)
    df_features["nivel_dificuldade"] = df["nivel_dificuldade"].values
    return df_features


def prever_dificuldade_texto(
    texto: str, caminho_modelo: str = "modelo_rf_dificuldade.pkl"
) -> int:
    """
    Carrega o modelo treinado e prevê o nível de dificuldade de um novo texto.
    """
    try:
        dados_carregados = joblib.load(caminho_modelo)
        modelo_rf = dados_carregados["modelo"]
    except FileNotFoundError:
        raise Exception(
            f"O modelo {caminho_modelo} não foi encontrado. Treine o modelo primeiro."
        )

    # 1. Extrai todas as métricas úteis (já sem o comprimento)
    features = extrair_features(texto)
    df_nova_amostra = pd.DataFrame([features])

    # 2. Faz a previsão direta
    previsao = modelo_rf.predict(df_nova_amostra)[0]

    return previsao
