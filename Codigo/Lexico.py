import re
from wordfreq import zipf_frequency

# Constantes globais
VOGAIS = set("aeiouáéíóúâêôãõàü")
CONSOANTES = set("bcdfghjklmnpqrstvwxyzç")
"""
METODOLOGIA DE PESOS LÉXICOS:
Diferentes classes gramaticais afetam a leitura de forma distinta.
Palavras com significado denso (Substantivos, Adjetivos, Verbos) têm pesos maiores (1.0 a 0.8).
Palavras de conexão ou gramaticais (Artigos, Preposições, Conjunções) afetam menos 
a dificuldade real e recebem pesos menores (0.1 a 0.3).
"""
PESOS_POS = {
    "NOUN": 2.0,  # Substantivo (Ex: livro, ideia, cidade)
    "ADJ": 2.0,  # Adjetivo (Ex: bonito, rápido, complexo)
    "VERB": 2.0,  # Verbo principal (Ex: correr, estudar, pensar)
    "ADV": 1.0,  # Advérbio (Ex: rapidamente, hoje, muito)
    "NUM": 0.4,  # Numeral (Ex: um, dois, primeiro)
    "PRON": 0.3,  # Pronome (Ex: ele, nós, aquilo)
    "ADP": 0.1,  # Preposição (Ex: em, de, para, com)
    "DET": 0.1,  # Determinante / Artigo (Ex: o, a, um, este)
    "CCONJ": 0.1,  # Conjunção coordenativa (Ex: e, mas, ou)
    "SCONJ": 0.1,  # Conjunção subordinativa (Ex: que, se, porque)
    "AUX": 0.3,  # Verbo auxiliar (Ex: ser, ter, ir - em tempos compostos)
}


def eh_canonica(palavra: str) -> bool:
    """
    O QUE FAZ: Verifica se a palavra possui o padrão silábico mais simples possível
    na língua portuguesa (Consoante-Vogal sucessivamente).

    METODOLOGIA: Avalia a estrutura da palavra. Primeiro, verifica se tem um número
    par de letras, caso não itera pela palavra de duas em duas letras, garantindo que o índice par é sempre
    uma consoante e o índice ímpar é sempre uma vogal (Ex: "bo-ta", "ca-sa", "ga-to").
    """
    p = palavra.lower().strip()

    if len(p) == 0 or len(p) % 2 != 0:
        return False

    for i in range(0, len(p), 2):
        if p[i] not in CONSOANTES or p[i + 1] not in VOGAIS:
            return False

    return True


def calcular_dificuldade_palavra(palavra: str) -> float:
    """
    O QUE FAZ: Atribui uma nota matemática de dificuldade (de 0.0 a 1.0) a uma
    palavra isolada.

    METODOLOGIA: Composição de três métricas penalizadoras:
    1. Frequência (60% do peso): Utiliza a escala de Zipf (biblioteca wordfreq) para
       descobrir quão comum a palavra é no idioma. Palavras muito raras recebem
       uma penalização alta.
    2. Tamanho (20% do peso): Calcula a proporção do tamanho da palavra face a um
       limite de 15 caracteres. Quanto maior, mais difícil.
    3. Canonicidade (20% do peso): Aplica uma penalização inteira (1.0)
       se a palavra for canónica.
    O resultado é a soma destas três fatias matemáticas.
    """
    zipf = zipf_frequency(palavra.lower(), "pt")
    zipf_limitado = min(max(zipf, 1.0), 6.5)

    dif_frequencia = 1.0 - ((zipf_limitado - 1.0) / (6.5 - 1.0))
    dif_tamanho = min(1.0, len(palavra) / 15.0)

    dif_canonicidade = 0.0 if eh_canonica(palavra) else 1.0

    score = (dif_frequencia * 0.60) + (dif_tamanho * 0.20) + (dif_canonicidade * 0.20)
    return round(score, 2)


def analisar_dificuldade_texto(texto: str, nlp_model) -> tuple[float, list]:
    """
    O QUE FAZ: Analisa a frase inteira, processa cada palavra individualmente e
    retorna a nota média de dificuldade lexical de todo o texto.

    METODOLOGIA: Média Ponderada baseada em POS Tagging. O texto passa pelo modelo
    de Inteligência Artificial (spaCy), que separa as palavras (ignorando pontuações)
    e identifica a classe gramatical de cada uma. Em seguida, calcula-se a dificuldade
    bruta da palavra e multiplica-se pelo seu "peso" gramatical (estabelecido no topo).
    Isto impede que preposições ou conjunções complexas inflem artificialmente a
    dificuldade de um texto simples.
    """
    doc = nlp_model(texto)

    palavras_analisadas = []
    soma_dificuldade_ponderada = 0.0
    soma_pesos = 0.0

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        palavra = token.text
        pos = token.pos_

        dif_bruta = calcular_dificuldade_palavra(palavra)
        peso = PESOS_POS.get(pos, 0.5)

        soma_dificuldade_ponderada += dif_bruta * peso
        soma_pesos += peso

        palavras_analisadas.append(
            {
                "palavra": palavra,
                "classe": pos,
                "canonica": "Sim" if eh_canonica(palavra) else "Não",
                "dificuldade_bruta": dif_bruta,
                "peso_aplicado": peso,
            }
        )

    dificuldade_global = (
        (soma_dificuldade_ponderada / soma_pesos) if soma_pesos > 0 else 0.0
    )
    return round(dificuldade_global, 2), palavras_analisadas
