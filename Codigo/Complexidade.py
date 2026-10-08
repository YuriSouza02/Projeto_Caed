import spacy
import pandas as pd
import joblib
import numpy as np

# Importação dos módulos de cálculo
import Codigo.Lexico as Lexico
import Codigo.Sintaxe as Sintaxe

# Carregar o modelo de NLP apenas uma vez na inicialização do sistema
nlp = spacy.load("pt_core_news_sm")


def extrair_features(texto: str) -> dict:
    """
    O QUE FAZ: Consolida e extrai o conjunto de métricas léxicas e sintáticas de um texto,
    transformando-o num vetor de características (features).

    METODOLOGIA: Processa a frase através do modelo de linguagem (spaCy) para gerar
    a árvore de dependências e coordena a execução dos submódulos do projeto:
    1. Análise Léxica: Obtém a nota de dificuldade global e calcula a densidade de
       palavras raras (proporção de tokens válidos com dificuldade bruta > 0.70).
    2. Análise Sintática e Discursiva: Mede o fator de proporção gramatical, a
       profundidade média da árvore sintática, o uso de voz passiva e as densidades
       de pontuação, diálogos e conectivos.
    3. Diversidade Vocabular: Avalia a riqueza do vocabulário através do cálculo
       de Type-Token Ratio (TTR).
    No final, retorna todas as métricas agregadas em um dicionário estruturado.
    """
    # Processamento do texto pelo modelo spaCy
    doc = nlp(texto)

    # Análise léxica global e extração do detalhamento por palavra
    score_lexico, analise_palavras = Lexico.analisar_dificuldade_texto(texto, nlp)

    # Cálculo da densidade de palavras raras (dificuldade bruta > 0.70)
    total_palavras_validas = len(analise_palavras)
    qtd_raras = sum(1 for p in analise_palavras if p["dificuldade_bruta"] > 0.70)
    densidade_raras = round(
        ((qtd_raras / total_palavras_validas) if total_palavras_validas > 0 else 0.0), 4
    )

    # Extração das métricas sintáticas, discursivas e de diversidade vocabular
    fator_sintatico = Sintaxe.calcular_fator_proporcao(doc)
    profundidade = Sintaxe.calcular_profundidade_media_arvore(doc)
    voz_passiva = Sintaxe.calcular_proporcao_voz_passiva(doc)
    dens_pontuacao = Sintaxe.calcular_densidade_pontuacao(doc)
    dens_dialogos = Sintaxe.calcular_densidade_dialogos(doc)
    dens_conectivos = Sintaxe.calcular_densidade_conectivos(doc)
    ttr = Sintaxe.calcular_ttr(texto)

    # Consolidação do vetor de características
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
    O QUE FAZ: Itera sobre uma coleção de textos em um DataFrame e constrói
    um novo conjunto de dados estruturado contendo todas as métricas (features)
    extraídas e a variável alvo (nível de dificuldade).

    METODOLOGIA: Percorre sequencialmente cada entrada da coluna "texto" do
    DataFrame de entrada, aplicando a função `extrair_features` para gerar o
    vetor de características correspondente. As métricas calculadas são
    acumuladas e convertidas em um novo `pd.DataFrame`. Em seguida, alinha
    os rótulos originais (`nivel_dificuldade`) à nova estrutura de dados para
    possibilitar o treino ou avaliação de modelos.
    """
    lista_features = []

    # Iteração sobre a coluna de textos para extração individual das métricas
    for texto in df["texto"]:
        features = extrair_features(texto)
        lista_features.append(features)

    # Conversão da lista de dicionários num novo DataFrame de métricas
    df_features = pd.DataFrame(lista_features)

    # Associação do rótulo original (variável alvo) ao conjunto final de dados
    df_features["nivel_dificuldade"] = df["nivel_dificuldade"].values

    return df_features


def prever_dificuldade_texto(
    texto: str, caminho_modelo: str = "modelo_rf_dificuldade.pkl"
) -> int:
    """
    O QUE FAZ: Carrega um modelo preditivo previamente treinado e estima
    o nível de dificuldade de leitura de um texto fornecido.

    METODOLOGIA: Executa o fluxo de inferência em três etapas:
    1. Desserialização do Modelo: Carrega o artefato binário (.pkl) via `joblib`.
    2. Vetorização da Amostra: Extrai as métricas léxicas e sintáticas através
    da função `extrair_features` e formata os dados em um `pd.DataFrame` de linha única compatível com o modelo.
    3. Inferência do Nível: Invoca o método `.predict()` do classificador
       (Random Forest) e retorna o rótulo da classe prevista como um número inteiro.
    """
    try:
        # Carregamento do dicionário contendo o modelo serializado via joblib
        dados_carregados = joblib.load(caminho_modelo)
        modelo_rf = dados_carregados["modelo"]
    except FileNotFoundError:
        raise Exception(
            f"O modelo {caminho_modelo} não foi encontrado. Treine o modelo primeiro."
        )

    # Extração das características do texto e conversão para o formato tabular
    features = extrair_features(texto)
    df_nova_amostra = pd.DataFrame([features])

    # Garante o alinhamento exato das colunas com o modelo treinado
    if hasattr(modelo_rf, "feature_names_in_"):
        df_nova_amostra = df_nova_amostra[modelo_rf.feature_names_in_]
        
    # Inferência da classe de dificuldade pelo modelo classificador
    previsao = modelo_rf.predict(df_nova_amostra)[0]

    return previsao
