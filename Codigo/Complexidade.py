import spacy
import pandas as pd
import joblib

import Codigo.Lexico as Lexico
import Codigo.Sintaxe as Sintaxe

nlp = spacy.load("pt_core_news_sm")


def extrair_features(texto: str) -> dict:
    """
    Extrai e consolida características léxicas, sintáticas e discursivas de
    um texto, utilizando o spaCy e os módulos de análise do projeto. Calcula
    nove métricas relacionadas à dificuldade lexical, à raridade das palavras,
    à estrutura sintática, à voz passiva, à pontuação, aos diálogos, aos
    conectivos e à diversidade vocabular. Retorna as métricas em um dicionário.
    """
    doc = nlp(texto)

    score_lexico, analise_palavras = Lexico.analisar_dificuldade_texto(texto, nlp)

    total_palavras_validas = len(analise_palavras)
    qtd_raras = sum(1 for p in analise_palavras if p["dificuldade_bruta"] > 0.70)
    densidade_raras = round(
        ((qtd_raras / total_palavras_validas) if total_palavras_validas > 0 else 0.0), 4
    )

    fator_sintatico = Sintaxe.calcular_fator_proporcao(doc)
    profundidade = Sintaxe.calcular_profundidade_media_arvore(doc)
    voz_passiva = Sintaxe.calcular_proporcao_voz_passiva(doc)
    dens_pontuacao = Sintaxe.calcular_densidade_pontuacao(doc)
    dens_dialogos = Sintaxe.calcular_densidade_dialogos(doc)
    dens_conectivos = Sintaxe.calcular_densidade_conectivos(doc)
    ttr = Sintaxe.calcular_ttr(texto)

    return {
        "score_lexico": score_lexico,
        "densidade_palavras_raras": densidade_raras,
        "fator_sintatico": fator_sintatico,
        "profundidade_arvore": profundidade,
        "proporcao_voz_passiva": voz_passiva,
        "densidade_pontuacao": dens_pontuacao,
        "densidade_dialogos": dens_dialogos,
        "densidade_conectivos": dens_conectivos,
        "ttr_diversidade": ttr,
    }


def construir_dataset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói um conjunto de dados a partir de um DataFrame de textos,
    aplicando a função extrair_features a cada entrada da coluna "texto".
    Associa as características extraídas aos respectivos rótulos da coluna
    "nivel_dificuldade" e retorna um novo DataFrame preparado para o treinamento
    ou a avaliação de modelos de classificação.
    """
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
    Carrega um modelo Random Forest previamente treinado e utiliza as
    características extraídas do texto para prever seu nível de dificuldade.
    Organiza as métricas em um DataFrame, alinhando suas colunas às utilizadas
    durante o treinamento. Retorna a classe prevista pelo modelo ou gera um
    erro caso o arquivo do modelo não seja encontrado.
    """
    try:
        dados_carregados = joblib.load(caminho_modelo)
        modelo_rf = dados_carregados["modelo"]
    except FileNotFoundError:
        raise Exception(
            f"O modelo {caminho_modelo} não foi encontrado. Treine o modelo primeiro."
        )

    features = extrair_features(texto)
    df_nova_amostra = pd.DataFrame([features])

    if hasattr(modelo_rf, "feature_names_in_"):
        df_nova_amostra = df_nova_amostra[modelo_rf.feature_names_in_]

    previsao = modelo_rf.predict(df_nova_amostra)[0]

    return previsao
