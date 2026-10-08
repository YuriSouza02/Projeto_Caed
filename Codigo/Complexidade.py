import spacy
import pandas as pd
import joblib
import numpy as np

# Importação dos módulos de cálculo
import Codigo.Lexico as Lexico
import Codigo.Sintaxe as Sintaxe

nlp = spacy.load("pt_core_news_sm")


def extrair_features(texto: str) -> dict:
    """
    Consolida as 11 características léxicas, sintáticas e de superfície.
    """
    doc = nlp(texto)

    score_lexico, analise_palavras = Lexico.analisar_dificuldade_texto(texto, nlp)

    total_palavras_validas = len(analise_palavras)
    qtd_raras = sum(1 for p in analise_palavras if p["dificuldade_bruta"] > 0.70)
    densidade_raras = round(
        ((qtd_raras / total_palavras_validas) if total_palavras_validas > 0 else 0.0), 4
    )

    return {
        "score_lexico": score_lexico,
        "densidade_palavras_raras": densidade_raras,
        "fator_sintatico": Sintaxe.calcular_fator_proporcao(doc),
        "profundidade_arvore": Sintaxe.calcular_profundidade_media_arvore(doc),
        "proporcao_voz_passiva": Sintaxe.calcular_proporcao_voz_passiva(doc),
        "densidade_pontuacao": Sintaxe.calcular_densidade_pontuacao(doc),
        "densidade_dialogos": Sintaxe.calcular_densidade_dialogos(doc),
        "densidade_conectivos": Sintaxe.calcular_densidade_conectivos(doc),
        "ttr_diversidade": Sintaxe.calcular_ttr(texto),
        # Novas Variáveis Integradas:
        "media_caracteres_palavra": Sintaxe.calcular_media_caracteres_palavra(doc),
        "comprimento_medio_frase": Sintaxe.calcular_comprimento_medio_frase(doc),
    }


def construir_dataset_features(df: pd.DataFrame) -> pd.DataFrame:
    lista_features = [extrair_features(texto) for texto in df["texto"]]
    df_features = pd.DataFrame(lista_features)
    df_features["nivel_dificuldade"] = df["nivel_dificuldade"].values
    return df_features


def prever_dificuldade_texto(
    texto: str, caminho_modelo: str = "modelo_rf_dificuldade.pkl"
) -> int:
    try:
        dados_carregados = joblib.load(caminho_modelo)
        modelo_rf = dados_carregados["modelo"]
    except FileNotFoundError:
        raise Exception(
            f"O modelo {caminho_modelo} não foi encontrado. Treine o modelo primeiro."
        )

    features = extrair_features(texto)
    df_nova_amostra = pd.DataFrame([features])

    # Correção: Garante alinhamento das colunas com o modelo treinado
    if hasattr(modelo_rf, "feature_names_in_"):
        df_nova_amostra = df_nova_amostra[modelo_rf.feature_names_in_]

    previsao = modelo_rf.predict(df_nova_amostra)[0]
    return previsao
