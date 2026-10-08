import re
import Codigo.Lexico as Lexico


def calcular_profundidade_media_arvore(doc) -> float:
    """
    O QUE FAZ: Mede a complexidade das frases com base no grau de subordinação
    (quantas orações dependem umas das outras).

    METODOLOGIA: Utiliza a árvore de dependência sintática do spaCy. Para cada
    frase, o código encontra a palavra que está mais "profunda" na árvore estrutural
    (ou seja, que tem o maior número de "ancestrais" ou palavras acima dela na
    hierarquia gramatical). Depois, faz a média dessas profundidades máximas para
    todo o texto. Valores altos indicam frases longas e cheias de orações subordinadas.
    """
    sents = list(doc.sents)
    if not sents:
        return 0.0
    profundidades = [
        max((len(list(t.ancestors)) for t in sent), default=0) for sent in sents
    ]
    return sum(profundidades) / len(profundidades)


def calcular_proporcao_voz_passiva(doc) -> float:
    """
    O QUE FAZ: Calcula a percentagem de frases do texto que utilizam a voz passiva.

    METODOLOGIA: Percorre todas as frases procurando marcadores sintáticos
    específicos identificados pelo spaCy, como sujeitos passivos ("nsubj:pass") ou
    verbos auxiliares na passiva ("aux:pass", ou etiquetas morfológicas de voz).
    A voz passiva (ex: "O rato foi comido pelo gato") exige maior esforço cognitivo
    do que a voz ativa ("O gato comeu o rato").
    """
    sents = list(doc.sents)
    if not sents:
        return 0.0
    frases_com_passiva = sum(
        1
        for sent in sents
        if any(
            t.dep_ in ("nsubj:pass", "aux:pass", "nsubjpass", "auxpass")
            or "Pass" in t.morph.get("Voice", [])
            for t in sent
        )
    )
    return frases_com_passiva / len(sents)


def calcular_densidade_pontuacao(doc) -> float:
    """
    O QUE FAZ: Mede a fragmentação das frases calculando a média de sinais de
    pontuação internos por frase.

    METODOLOGIA: Conta o número de vírgulas, ponto e vírgula, dois pontos e travessões
    no texto e divide pelo número total de frases. Muitas vírgulas geralmente indicam
    muitos incisos, apostos ou enumerações, o que quebra o fluxo de leitura e
    aumenta a dificuldade.
    """
    sents = list(doc.sents)
    if not sents:
        return 0.0
    pontuacoes_alvo = {",", ";", ":", "-", "—", "–"}
    total_pontuacoes = sum(1 for t in doc if t.text in pontuacoes_alvo)
    return total_pontuacoes / len(sents)


def calcular_densidade_dialogos(doc) -> float:
    """
    O QUE FAZ: Verifica se o texto tem características narrativas e conversacionais.

    METODOLOGIA: Conta a proporção de frases que contêm marcadores de fala típicos,
    como travessões ou aspas. Textos com alta densidade de diálogo costumam ser mais
    acessíveis e dinâmicos (como livros infantojuvenis ou crónicas), o que ajudará a
    reduzir a pontuação de dificuldade na função principal.
    """
    sents = list(doc.sents)
    if not sents:
        return 0.0
    marcadores_fala = {"—", "-", "–", '"', "“", "”", "«", "»"}
    frases_dialogo = sum(
        1 for sent in sents if any(t.text in marcadores_fala for t in sent)
    )
    return frases_dialogo / len(sents)


def calcular_densidade_conectivos(doc) -> float:
    """
    O QUE FAZ: Mede a frequência de palavras de transição e ligação (conectivos) no texto.

    METODOLOGIA: Em vez de usar uma lista estática de Expressões Regulares, itera pelos
    tokens processados pelo spaCy. O código verifica a etiqueta de classe gramatical (POS tag).
    Se o token for classificado como 'CCONJ' (conjunção coordenativa) ou 'SCONJ'
    (conjunção subordinativa), é contabilizado como conectivo. Divide-se então este
    total pelo número de palavras válidas.
    """
    tokens_validos = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens_validos:
        return 0.0

    total_conectivos = sum(1 for t in tokens_validos if t.pos_ in ("CCONJ", "SCONJ"))

    return total_conectivos / len(tokens_validos)


def calcular_fator_proporcao(doc) -> float:
    """
    O QUE FAZ: Agrupa todas as métricas sintáticas num único multiplicador (fator)
    que penaliza a dificuldade do texto consoante os obstáculos gramaticais que encontra.

    METODOLOGIA: Base de Cálculo e Acréscimos.
    O fator começa sempre em 1.0 (neutro). O código faz uma contagem das classes
    gramaticais (substantivos, verbos, etc.) e analisa as densidades calculadas nas
    funções acima. Em seguida, aplica uma série de "multas" (penalizações):
    - +0.15 se o texto for demasiado descritivo (muitos substantivos e adjetivos).
    - +0.10 se houver poucos verbos para muitos substantivos (textos estáticos).
    - +0.10 se houver muitos números (dados estatísticos/financeiros).
    - +0.15 a +0.10 para alta profundidade sintática, uso de voz passiva, excesso de
      pontuação e presença de muitos conectivos formais.
    No final, aplica um "bónus de acessibilidade" (-0.15) se detetar que o texto é
    rico em diálogos. O resultado final é devolvido com duas casas decimais.
    """
    tokens_validos = [t for t in doc if not t.is_punct and not t.is_space]
    total_palavras = len(tokens_validos)
    if total_palavras == 0:
        return 1.0

    substantivos, adjetivos, verbos, numerais, qtd_complexas = 0, 0, 0, 0, 0

    for t in tokens_validos:
        if t.pos_ == "NOUN":
            substantivos += 1
        elif t.pos_ == "ADJ":
            adjetivos += 1
        elif t.pos_ == "VERB":
            verbos += 1
        elif t.pos_ == "NUM":
            numerais += 1
        if Lexico.calcular_dificuldade_palavra(t.text) > 0.60:
            qtd_complexas += 1

    fator = 1.0

    # Penalidades Léxicas/POS
    if (substantivos + adjetivos + verbos) / total_palavras > 0.50:
        fator += 0.15
    # Correção 2: Proteção contra divisão por zero se verbos == 0
    if verbos > 0:
        if (substantivos / verbos) > 2.5:
            fator += 0.10
    elif substantivos > 0:
        fator += 0.10
    if numerais / total_palavras > 0.15:
        fator += 0.15
    if qtd_complexas / total_palavras > 0.20:
        fator += 0.15

    # Penalidades Sintáticas e Discursivas
    if calcular_profundidade_media_arvore(doc) > 4.5:
        fator += 0.15
    if calcular_proporcao_voz_passiva(doc) > 0.2:
        fator += 0.10
    if calcular_densidade_pontuacao(doc) > 2.5:
        fator += 0.10
    if calcular_densidade_conectivos(doc) > 0.03:
        fator += 0.10
    if calcular_densidade_dialogos(doc) > 0.10:
        fator -= 0.15

    return round(fator, 2)


def calcular_ttr(texto: str) -> float:
    """
    O QUE FAZ: Mede a riqueza e a diversidade do vocabulário do texto.

    METODOLOGIA: Aplica o cálculo de Type-Token Ratio (TTR). A métrica é a razão
    matemática entre o número de palavras únicas (set(palavras)) e o total absoluto
    de palavras no texto. Textos altamente repetitivos geram pontuações próximas de 0,
    enquanto textos com vocabulário diversificado aproximam-se de 1.
    """
    palavras = [p.lower() for p in re.findall(r"\b\w+\b", texto)]
    total_palavras = len(palavras)

    if total_palavras == 0:
        return 0.0
    ttr = len(set(palavras)) / total_palavras
    return round(ttr, 4)


def calcular_media_caracteres_palavra(doc) -> float:
    """
    O QUE FAZ: Mede a extensão média das palavras do texto em quantidade de caracteres.

    METODOLOGIA: Filtra os tokens do spaCy ignorando pontuações e espaços. Soma o
    comprimento individual (len) de cada palavra válida e divide pelo total de palavras.
    Palavras mais longas (como termos técnicos ou polissílabos) exigem maior tempo
    de fixação ocular e aumentam a carga cognitiva na leitura.
    """
    tokens_validos = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens_validos:
        return 0.0

    total_caracteres = sum(len(t.text) for t in tokens_validos)
    return round(total_caracteres / len(tokens_validos), 2)


def calcular_comprimento_medio_frase(doc) -> float:
    """
    O QUE FAZ: Mede a extensão média das frases calculando a quantidade de palavras
    por oração/frase.

    METODOLOGIA: Percorre as frases identificadas pelo spaCy (doc.sents) e conta
    apenas os tokens válidos (excluindo pontuações e espaços). Em seguida, calcula
    a média dividindo o total de palavras pelo número de frases do texto. Frases longas
    exigem maior retenção na memória de trabalho do leitor, elevando a dificuldade.
    """
    sents = list(doc.sents)
    if not sents:
        return 0.0

    palavras_por_frase = [
        sum(1 for t in sent if not t.is_punct and not t.is_space) for sent in sents
    ]

    total_palavras = sum(palavras_por_frase)
    return round(total_palavras / len(sents), 2)

