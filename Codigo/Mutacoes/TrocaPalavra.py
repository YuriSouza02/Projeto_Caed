import os
import random
import time
from typing import List, Tuple, Optional
import nltk
import spacy
from nltk.corpus import wordnet as wn

diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_nltk_data = os.path.join(diretorio_base, ".venv", "nltk_data")

if os.path.exists(caminho_nltk_data) and caminho_nltk_data not in nltk.data.path:
    nltk.data.path.append(caminho_nltk_data)

try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    try:
        nlp = spacy.load("pt_core_news_sm")
        print(
            "[AVISO] Modelo 'pt_core_news_lg' nao encontrado. Usando 'pt_core_news_sm'."
        )
    except OSError:
        print("Erro: Nenhum modelo do spaCy foi encontrado.")
        raise

MAPA_POS_WORDNET = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}


def obter_sinonimos_antonimos(
    palavra: str, pos_spacy: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Busca sinônimos e antônimos de uma palavra utilizando a WordNet.
    Inclui suporte completo a adjetivos satélites ('s') e navegação via similar_tos().
    """
    sinonimos = set()
    antonimos = set()

    palavra_limpa = palavra.lower().strip()

    if pos_spacy == "ADJ":
        wn_poses = [wn.ADJ, wn.ADJ_SAT]
    elif pos_spacy in MAPA_POS_WORDNET:
        wn_poses = [MAPA_POS_WORDNET[pos_spacy]]
    else:
        wn_poses = [None]

    synsets = []
    for wn_pos in wn_poses:
        if wn_pos:
            synsets.extend(wn.synsets(palavra_limpa, pos=wn_pos, lang="por"))
        else:
            synsets.extend(wn.synsets(palavra_limpa, lang="por"))

    if not synsets:
        for wn_pos in wn_poses:
            if wn_pos:
                synsets.extend(wn.synsets(palavra_limpa, pos=wn_pos))
            else:
                synsets.extend(wn.synsets(palavra_limpa))

    for syn in synsets:
        synsets_relacionados = [syn]
        if syn.pos() in (wn.ADJ, wn.ADJ_SAT):
            synsets_relacionados.extend(syn.similar_tos())

        for s in synsets_relacionados:
            # 1. Sinônimos em Português
            for lemma in s.lemmas(lang="por"):
                nome = lemma.name().replace("_", " ").strip()
                if nome.lower() != palavra_limpa:
                    sinonimos.add(nome)

            # 2. Antônimos via ponte com o Inglês
            for lemma_eng in s.lemmas(lang="eng"):
                for ant in lemma_eng.antonyms():
                    ant_synsets = [ant.synset()]
                    if ant.synset().pos() in (wn.ADJ, wn.ADJ_SAT):
                        ant_synsets.extend(ant.synset().similar_tos())

                    for ant_s in ant_synsets:
                        for lemma_pt in ant_s.lemmas(lang="por"):
                            nome_ant = lemma_pt.name().replace("_", " ").strip()
                            if nome_ant.lower() != palavra_limpa:
                                antonimos.add(nome_ant)

    return list(sinonimos), list(antonimos)


def filtrar_por_similaridade_vetorial(
    token_original: spacy.tokens.Token,
    candidatos: List[str],
    nlp_model: spacy.language.Language,
    limiar_minimo: float = 0.40,
) -> List[str]:
    """Filtra candidatos descartando opções com baixa similaridade vetorial."""
    if not token_original.has_vector or not candidatos:
        return candidatos

    validos = []
    for cand in candidatos:
        doc_cand = nlp_model(cand)
        if len(doc_cand) > 0 and doc_cand.has_vector:
            sim = token_original.similarity(doc_cand)
            if sim >= limiar_minimo:
                validos.append(cand)

    return validos if validos else candidatos


def formatar_caixa_texto(token_original, nova_palavra: str) -> str:
    """Aplica a formatação de caixa com base no token original."""
    if token_original.is_title:
        return nova_palavra.title()
    elif token_original.is_upper:
        return nova_palavra.upper()
    return nova_palavra.lower()


def singularizar_palavra(palavra: str) -> str:
    """Reduz uma palavra no plural para o singular básico."""
    p = palavra.lower()
    if p.endswith("ões"):
        return p[:-3] + "ão"
    elif p.endswith("ães"):
        return p[:-3] + "ão"
    elif p.endswith("ais"):
        return p[:-3] + "al"
    elif p.endswith("éis"):
        return p[:-3] + "el"
    elif p.endswith("óis"):
        return p[:-3] + "ol"
    elif p.endswith("uis"):
        return p[:-3] + "ul"
    elif p.endswith("is"):
        return p[:-2] + "il"
    elif p.endswith("ns"):
        return p[:-2] + "m"
    elif p.endswith("es") and len(p) > 3:
        return p[:-2]
    elif p.endswith("s") and not p.endswith("ss") and len(p) > 3:
        return p[:-1]
    return p


def flexionar_numero(palavra: str, numero_alvo: str) -> str:
    """Ajusta o número da palavra (Singular ou Plural)."""
    palavra_sing = singularizar_palavra(palavra)
    if not numero_alvo or numero_alvo == "Sing":
        return palavra_sing

    if numero_alvo == "Plur":
        if palavra_sing.endswith(("a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú")):
            return palavra_sing + "s"
        elif palavra_sing.endswith("m"):
            return palavra_sing[:-1] + "ns"
        elif palavra_sing.endswith(("r", "z")):
            return palavra_sing + "es"
        elif palavra_sing.endswith("al"):
            return palavra_sing[:-1] + "is"
        elif palavra_sing.endswith("el"):
            return palavra_sing[:-2] + "éis"
        elif palavra_sing.endswith("ol"):
            return palavra_sing[:-2] + "óis"
        elif palavra_sing.endswith("ul"):
            return palavra_sing[:-2] + "uis"
        elif palavra_sing.endswith("il"):
            return palavra_sing[:-2] + "is"
        elif palavra_sing.endswith("ão"):
            return palavra_sing[:-2] + "ões"
        return palavra_sing + "s"

    return palavra_sing


def flexionar_genero_adjetivo(palavra: str, genero_alvo: Optional[str]) -> str:
    """Flexiona adjetivos entre Masculino e Feminino."""
    if not genero_alvo:
        return palavra

    p = palavra.lower()
    if genero_alvo == "Fem":
        if p.endswith("o"):
            return p[:-1] + "a"
        elif p.endswith("or"):
            return p + "a"
        elif p.endswith("ês"):
            return p[:-2] + "esa"
        elif p.endswith("ão"):
            return p[:-2] + "ã"
    elif genero_alvo == "Masc":
        if p.endswith("esa"):
            return p[:-3] + "ês"
        elif p.endswith("ã"):
            return p[:-1] + "ão"
        elif p.endswith("a") and len(p) > 3:
            return p[:-1] + "o"

    return p


def inferir_genero_substantivo(
    palavra: str, nlp_model: spacy.language.Language
) -> Optional[str]:
    """Infere o gênero gramatical de um substantivo isolado."""
    doc = nlp_model(palavra)
    if len(doc) > 0:
        gen = doc[0].morph.get("Gender")
        if gen:
            return gen[0]

    p = palavra.lower()
    if p.endswith(("a", "ção", "são", "dade", "agem", "ez", "ice")):
        return "Fem"
    elif p.endswith(("o", "or", "ume", "al", "el", "ol", "ul")):
        return "Masc"
    return None


def filtrar_candidatos_por_genero(
    candidatos: List[str],
    token_original: spacy.tokens.Token,
    nlp_model: spacy.language.Language,
) -> List[str]:
    """Garante que substantivos de substituição mantenham o mesmo gênero do original."""
    if token_original.pos_ != "NOUN":
        return candidatos

    genero_orig = token_original.morph.get("Gender")
    if not genero_orig:
        return candidatos

    gen_orig_val = genero_orig[0]
    candidatos_validos = []

    for cand in candidatos:
        cand_sing = singularizar_palavra(cand)
        gen_cand = inferir_genero_substantivo(cand_sing, nlp_model)
        if gen_cand == gen_orig_val:
            candidatos_validos.append(cand_sing)

    return candidatos_validos


def conjugar_verbo(
    infinitivo: str,
    token_original: spacy.tokens.Token,
    nlp_model: spacy.language.Language,
) -> str:
    """Conjuga o verbo no infinitivo para a mesma forma do verbo original."""
    doc_inf = nlp_model(infinitivo)
    if len(doc_inf) > 0 and doc_inf[0].lemma_:
        infinitivo = doc_inf[0].lemma_.lower()

    verb_form = token_original.morph.get("VerbForm")
    if not verb_form:
        return infinitivo

    v_form = verb_form[0]

    irregulares = {
        ("ir", "Ind", "Past", "3", "Sing"): "foi",
        ("ir", "Ind", "Past", "3", "Plur"): "foram",
        ("ser", "Ind", "Past", "3", "Sing"): "foi",
        ("ser", "Ind", "Past", "3", "Plur"): "foram",
        ("ter", "Ind", "Past", "3", "Sing"): "teve",
        ("ter", "Ind", "Past", "3", "Plur"): "tiveram",
        ("fazer", "Ind", "Past", "3", "Sing"): "fez",
        ("fazer", "Ind", "Past", "3", "Plur"): "fizeram",
        ("vir", "Ind", "Past", "3", "Sing"): "veio",
        ("vir", "Ind", "Past", "3", "Plur"): "vieram",
    }

    if v_form == "Ger":
        if infinitivo.endswith("ar"):
            return infinitivo[:-2] + "ando"
        if infinitivo.endswith("er"):
            return infinitivo[:-2] + "endo"
        if infinitivo.endswith("ir"):
            return infinitivo[:-2] + "indo"
        if infinitivo.endswith("or") or infinitivo.endswith("ôr"):
            return infinitivo[:-2] + "ondo"
        return infinitivo

    if v_form == "Part":
        genero = token_original.morph.get("Gender")
        numero = token_original.morph.get("Number")
        gen = genero[0] if genero else "Masc"
        num = numero[0] if numero else "Sing"

        if infinitivo.endswith("ar"):
            radical = infinitivo[:-2] + "ad"
        elif infinitivo.endswith(("er", "ir", "or", "ôr")):
            radical = infinitivo[:-2] + "id"
        else:
            return infinitivo

        sufixo_gen = "a" if gen == "Fem" else "o"
        sufixo_num = "s" if num == "Plur" else ""
        return radical + sufixo_gen + sufixo_num

    if v_form == "Fin":
        mood = token_original.morph.get("Mood")
        tense = token_original.morph.get("Tense")
        person = token_original.morph.get("Person")
        number = token_original.morph.get("Number")

        m = mood[0] if mood else "Ind"
        t = tense[0] if tense else "Pres"
        p = person[0] if person else "3"
        n = number[0] if number else "Sing"

        if (infinitivo, m, t, p, n) in irregulares:
            return irregulares[(infinitivo, m, t, p, n)]

        if not (
            infinitivo.endswith("ar")
            or infinitivo.endswith("er")
            or infinitivo.endswith("ir")
        ):
            return infinitivo

        tipo = infinitivo[-2:]
        radical = infinitivo[:-2]

        desinencias = {
            ("Ind", "Pres", "1", "Sing"): {"ar": "o", "er": "o", "ir": "o"},
            ("Ind", "Pres", "2", "Sing"): {"ar": "as", "er": "es", "ir": "es"},
            ("Ind", "Pres", "3", "Sing"): {"ar": "a", "er": "e", "ir": "e"},
            ("Ind", "Pres", "1", "Plur"): {"ar": "amos", "er": "emos", "ir": "imos"},
            ("Ind", "Pres", "2", "Plur"): {"ar": "ais", "er": "eis", "ir": "is"},
            ("Ind", "Pres", "3", "Plur"): {"ar": "am", "er": "em", "ir": "em"},
            ("Ind", "Past", "1", "Sing"): {"ar": "ei", "er": "i", "ir": "i"},
            ("Ind", "Past", "2", "Sing"): {"ar": "aste", "er": "este", "ir": "iste"},
            ("Ind", "Past", "3", "Sing"): {"ar": "ou", "er": "eu", "ir": "iu"},
            ("Ind", "Past", "1", "Plur"): {"ar": "amos", "er": "emos", "ir": "imos"},
            ("Ind", "Past", "2", "Plur"): {"ar": "astes", "er": "estes", "ir": "istes"},
            ("Ind", "Past", "3", "Plur"): {"ar": "aram", "er": "eram", "ir": "iram"},
            ("Ind", "Imp", "1", "Sing"): {"ar": "ava", "er": "ia", "ir": "ia"},
            ("Ind", "Imp", "2", "Sing"): {"ar": "avas", "er": "ias", "ir": "ias"},
            ("Ind", "Imp", "3", "Sing"): {"ar": "ava", "er": "ia", "ir": "ia"},
            ("Ind", "Imp", "1", "Plur"): {"ar": "ávamos", "er": "íamos", "ir": "íamos"},
            ("Ind", "Imp", "2", "Plur"): {"ar": "áveis", "er": "íeis", "ir": "íeis"},
            ("Ind", "Imp", "3", "Plur"): {"ar": "avam", "er": "iam", "ir": "iam"},
            ("Ind", "Fut", "1", "Sing"): {"ar": "arei", "er": "erei", "ir": "irei"},
            ("Ind", "Fut", "2", "Sing"): {"ar": "arás", "er": "erás", "ir": "irás"},
            ("Ind", "Fut", "3", "Sing"): {"ar": "ará", "er": "erá", "ir": "irá"},
            ("Ind", "Fut", "1", "Plur"): {
                "ar": "aremos",
                "er": "eremos",
                "ir": "iremos",
            },
            ("Ind", "Fut", "2", "Plur"): {"ar": "areis", "er": "ereis", "ir": "ireis"},
            ("Ind", "Fut", "3", "Plur"): {"ar": "arão", "er": "erão", "ir": "irão"},
        }

        sufixo = desinencias.get((m, t, p, n), {}).get(tipo)
        if sufixo:
            return radical + sufixo

    return infinitivo


def realizar_data_augmentation(
    texto: str,
    prob_troca: float = 0.3,
    modo: str = "sinonimo",
    classes_alvo: Optional[List[str]] = None,
    permitir_expressoes_compostas: bool = False,
    fallback_sinonimo: bool = True,
    limiar_similaridade: float = 0.40,
    seed: Optional[int] = None,
    nlp_model: Optional[spacy.language.Language] = None,
) -> str:
    """Realiza data augmentation combinando WordNet, Embeddings e Regras Morfológicas."""
    if seed is not None:
        random.seed(seed)

    if classes_alvo is None:
        classes_alvo = ["NOUN", "ADJ", "VERB", "ADV"]

    doc_pipeline = nlp_model if nlp_model is not None else nlp
    doc = doc_pipeline(texto)
    texto_reconstruido = []

    for token in doc:
        if token.pos_ not in classes_alvo or random.random() > prob_troca:
            texto_reconstruido.append(token.text_with_ws)
            continue

        sinonimos, antonimos = obter_sinonimos_antonimos(
            token.lemma_, pos_spacy=token.pos_
        )

        if modo == "antonimo":
            opcoes = (
                antonimos if antonimos else (sinonimos if fallback_sinonimo else [])
            )
        else:
            opcoes = sinonimos

        if not permitir_expressoes_compostas:
            opcoes = [opt for opt in opcoes if " " not in opt]

        # 1. Filtro de Embeddings (Vetores de Similaridade)
        opcoes = filtrar_por_similaridade_vetorial(
            token, opcoes, doc_pipeline, limiar_minimo=limiar_similaridade
        )

        # 2. Filtro Morfológico (Gênero para Substantivos)
        if token.pos_ == "NOUN":
            opcoes = filtrar_candidatos_por_genero(opcoes, token, doc_pipeline)

        if opcoes:
            nova_palavra = random.choice(opcoes)

            genero_alvo = token.morph.get("Gender")
            genero_val = genero_alvo[0] if genero_alvo else None

            numero_alvo = token.morph.get("Number")
            numero_val = numero_alvo[0] if numero_alvo else None

            if token.pos_ == "VERB":
                nova_palavra = conjugar_verbo(nova_palavra, token, doc_pipeline)
            elif token.pos_ == "ADJ":
                nova_palavra = flexionar_genero_adjetivo(nova_palavra, genero_val)
                nova_palavra = flexionar_numero(nova_palavra, numero_val)
            elif token.pos_ == "NOUN":
                nova_palavra = flexionar_numero(nova_palavra, numero_val)

            nova_palavra_formatada = formatar_caixa_texto(token, nova_palavra)
            texto_reconstruido.append(nova_palavra_formatada + token.whitespace_)
        else:
            texto_reconstruido.append(token.text_with_ws)

    return "".join(texto_reconstruido)


if __name__ == "__main__":

    def destacar_diferencas(original: str, alterado: str) -> str:
        orig_words = original.split()
        alt_words = alterado.split()
        diff = []
        for o, a in zip(orig_words, alt_words):
            if o != a:
                diff.append(f"[{o} -> {a}]")
            else:
                diff.append(o)
        return " ".join(diff)

    def calcular_taxa_mudanca(original: str, alterado: str) -> float:
        orig_words = original.split()
        alt_words = alterado.split()
        if not orig_words:
            return 0.0
        mudancas = sum(1 for o, a in zip(orig_words, alt_words) if o != a)
        return (mudancas / len(orig_words)) * 100

    print("=" * 60)
    print(" INICIANDO BATERIA AVANÇADA DE TESTES DE DATA AUGMENTATION")
    print("=" * 60)

    frases_teste = [
        "O menino feliz correu rapidamente para a escola grande.",
        "A MULHER INTELIGENTE comprou um carro novo.",
        "O sistema está lento, apresentando um problema terrível e grave.",
    ]

    print("\n[TESTE 1] Eficácia de Sinônimos com Embeddings")
    for frase in frases_teste:
        resultado = realizar_data_augmentation(
            frase, prob_troca=1.0, modo="sinonimo", seed=42, limiar_similaridade=0.40
        )
        taxa = calcular_taxa_mudanca(frase, resultado)
        print(f"Original : {frase}")
        print(f"Aumentada: {resultado}")
        print(f"Detalhes : {destacar_diferencas(frase, resultado)}")
        print(f"Taxa de mudança: {taxa:.1f}%\n")

    print("\n[TESTE 2] Eficácia de Antônimos com Embeddings")
    for frase in frases_teste:
        resultado = realizar_data_augmentation(
            frase, prob_troca=1.0, modo="antonimo", seed=42, limiar_similaridade=0.40
        )
        taxa = calcular_taxa_mudanca(frase, resultado)
        print(f"Original : {frase}")
        print(f"Aumentada: {resultado}")
        print(f"Taxa de mudança: {taxa:.1f}%\n")
