import os
import re
from typing import List, Optional
import spacy

# Define o diretório-base do projeto.
diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nlp = spacy.load("pt_core_news_sm")


def _parafrasear_titulo_aposto(doc) -> Optional[str]:
    """
    Estrutura frases que apresentam um substantivo seguido de um nome próprio,
    colocando o nome próprio no início e transformando o substantivo em um aposto
    entre vírgulas.
    "O cantor Roberto Carlos fez aniversário"
    "Roberto Carlos, o cantor, fez aniversário".
    """
    if "," in doc.text:
        return None

    tokens = list(doc)
    for i in range(len(tokens) - 1):
        det_texto = ""
        idx_inicio = i

        if (
            tokens[i].pos_ == "DET"
            and i + 1 < len(tokens)
            and tokens[i + 1].pos_ == "NOUN"
        ):
            det_texto = tokens[i].text.lower() + " "
            idx_substantivo = i + 1
        elif tokens[i].pos_ == "NOUN":
            idx_substantivo = i
        else:
            continue

        substantivo = tokens[idx_substantivo].text.lower()
        nomes_proprios = []
        idx_atual = idx_substantivo + 1

        while idx_atual < len(tokens):
            if tokens[idx_atual].pos_ == "PROPN":
                nomes_proprios.append(tokens[idx_atual].text)
                idx_atual += 1
            elif (
                tokens[idx_atual].pos_ == "ADP"
                and idx_atual + 1 < len(tokens)
                and tokens[idx_atual + 1].pos_ == "PROPN"
            ):
                nomes_proprios.append(tokens[idx_atual].text)
                nomes_proprios.append(tokens[idx_atual + 1].text)
                idx_atual += 2
            else:
                break

        if nomes_proprios:
            nome_completo = " ".join(nomes_proprios)
            antes = "".join([t.text_with_ws for t in tokens[:idx_inicio]])
            depois = "".join([t.text_with_ws for t in tokens[idx_atual:]])
            aposto = f"{det_texto}{substantivo}".strip()
            resultado = f"{antes}{nome_completo}, {aposto}, {depois}".strip()
            resultado = re.sub(r"\s+", " ", resultado)
            resultado = re.sub(r"\s+([.,!?;:])", r"\1", resultado)
            return resultado
    return None


def _parafrasear_bloco_adverbial(doc) -> Optional[str]:
    """
    Reestrutura frases deslocando **blocos** adverbiais. A transformação preserva a oração
    principal e ajusta a capitalização e a pontuação conforme a nova ordem.
    Retorna a frase original e caso não encontre uma estrutura compatível.
    """
    texto = doc.text.strip()

    # Conectivos subordinativos tradicionais
    conectivos = [
        r"à medida que",
        r"à proporção que",
        r"ao passo que",
        r"a fim de que",
        r"de modo que",
        r"de forma que",
        r"de sorte que",
        r"uma vez que",
        r"visto que",
        r"já que",
        r"dado que",
        r"por mais que",
        r"a despeito de que",
        r"ainda que",
        r"mesmo que",
        r"posto que",
        r"a menos que",
        r"contanto que",
        r"salvo se",
        r"logo que",
        r"assim que",
        r"antes que",
        r"depois que",
        r"desde que",
        r"do mesmo modo que",
        r"assim como",
        r"tal qual",
        r"quando",
        r"enquanto",
        r"embora",
        r"conquanto",
        r"conforme",
        r"consoante",
        r"segundo",
        r"porque",
        r"porquanto",
        r"caso",
        r"como",
        r"para que",
        r"para",
        r"se",
    ]
    padrao_conectivos = "|".join(conectivos)

    # Adicionamos a lógica do "Há [tempo]" como um bloco que pode ser capturado inteiro
    padrao_bloco_inicio = (
        rf"^((?:{padrao_conectivos})\s+.+?|há\s+[^,.!?]+(?:\s+atrás)?),\s*(.+)$"
    )
    padrao_bloco_fim = rf"^(.+?)\s*,?\s*((?:{padrao_conectivos})\s+.+?|há\s+[^,.!?]+(?:\s+atrás)?)([.:;!?]?)$"

    # 1. Bloco no INÍCIO -> desloca para o FIM
    match_inicio = re.match(padrao_bloco_inicio, texto, re.IGNORECASE)
    if match_inicio:
        bloco_adverbial, oracao_principal = match_inicio.groups()

        pontuacao = ""
        if oracao_principal[-1] in ".:;!?":
            pontuacao = oracao_principal[-1]
            oracao_principal = oracao_principal[:-1]

        oracao_principal_fmt = oracao_principal[0].upper() + oracao_principal[1:]
        bloco_adverbial_fmt = bloco_adverbial.lower()

        return f"{oracao_principal_fmt} {bloco_adverbial_fmt}{pontuacao}".strip()

    # 2. Bloco no FIM -> desloca para o INÍCIO
    match_fim = re.match(padrao_bloco_fim, texto, re.IGNORECASE)
    if match_fim:
        oracao_principal, bloco_adverbial, pontuacao = match_fim.groups()

        bloco_adverbial_fmt = bloco_adverbial[0].upper() + bloco_adverbial[1:].lower()
        oracao_principal_fmt = oracao_principal[0].lower() + oracao_principal[1:]

        return f"{bloco_adverbial_fmt}, {oracao_principal_fmt}{pontuacao}".strip()

    return None


def _parafrasear_deslocamento_adverbio(doc) -> Optional[str]:
    """
    Reestrutura frases deslocando **palavras** adverbiais. A transformação preserva a oração
    principal e ajusta a capitalização e a pontuação conforme a nova ordem.
    Retorna a frase original e caso não encontre uma estrutura compatível.
    "Nós estudamos muito ontem."
    "Ontem, nós estudamos muito."
    """
    if len(doc) > 2 and doc[0].pos_ == "ADV" and doc[1].text == ",":
        return None

    if len(doc) > 3:
        ultimo_token = doc[-2] if doc[-1].is_punct else doc[-1]

        if ultimo_token.dep_ == "advmod" and ultimo_token.pos_ == "ADV":
            adv = ultimo_token.text.capitalize()
            resto = re.sub(rf"\b{ultimo_token.text}\b", "", doc.text).strip()
            resto = re.sub(r"\s+([.,!?;:])", r"\1", resto)
            resto = resto[0].lower() + resto[1:] if resto else ""

            return f"{adv}, {resto}"
    return None


def _parafrasear_voz_ativa_passiva(doc) -> Optional[str]:
    """
    Converte uma frase da voz ativa para a voz passiva. A função constrói a nova
    estrutura com o objeto como sujeito paciente, o verbo auxiliar "ser" e o
    particípio do verbo principal, seguido do agente da passiva.
    """
    sujeito, verbo, objeto = None, None, None
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            sujeito = token
        elif token.pos_ == "VERB" and token.dep_ == "ROOT":
            verbo = token
        elif token.dep_ == "obj":
            objeto = token

    tem_auxiliar = any(t.dep_ in ["aux", "aux:pass"] for t in doc)

    if sujeito and verbo and objeto and not tem_auxiliar:
        det_obj = " ".join([t.text for t in objeto.lefts if t.pos_ == "DET"])

        is_plural = "Plur" in objeto.morph.get("Number", []) or (
            objeto.text.endswith("s")
        )
        is_fem = "Fem" in objeto.morph.get("Gender", []) or (
            det_obj.strip().lower() in ["a", "as", "uma", "umas"]
        )

        verbo_aux = "foram" if is_plural else "foi"

        radical = verbo.lemma_[:-2]
        terminacao = verbo.lemma_[-2:]
        part = radical + "ad" if terminacao == "ar" else radical + "id"

        if is_fem and is_plural:
            part += "as"
        elif is_fem:
            part += "a"
        elif is_plural:
            part += "os"
        else:
            part += "o"

        det_sujeito = " ".join(
            [t.text for t in sujeito.lefts if t.pos_ == "DET"]
        ).lower()
        if "o" in det_sujeito:
            agente = f"pelo {sujeito.text}"
        elif "a" in det_sujeito:
            agente = f"pela {sujeito.text}"
        elif "os" in det_sujeito:
            agente = f"pelos {sujeito.text}"
        elif "as" in det_sujeito:
            agente = f"pelas {sujeito.text}"
        else:
            agente = f"por {det_sujeito} {sujeito.text}".strip()

        novo_sujeito = f"{det_obj} {objeto.text}".strip().capitalize()

        pontuacao = doc[-1].text if doc[-1].is_punct else "."
        return f"{novo_sujeito} {verbo_aux} {part} {agente}{pontuacao}"
    return None


def _parafrasear_voz_passiva_ativa(doc) -> Optional[str]:
    """
    Faz justo o contrario da função anterior, convertendo uma frase da voz passiva para a voz ativa.
    """
    sujeito_paciente, verbo_principal, verbo_auxiliar, agente_passiva = (
        None,
        None,
        None,
        None,
    )

    for token in doc:
        if (
            token.dep_ in ["nsubj:pass", "nsubjpass", "nsubj"]
            and token.head.pos_ == "VERB"
        ):
            sujeito_paciente = token
        elif token.dep_ in ["aux:pass", "aux", "cop"] and token.lemma_ in [
            "ser",
            "ir",
            "estar",
        ]:
            verbo_auxiliar = token
        elif token.pos_ == "VERB" and token.dep_ == "ROOT":
            verbo_principal = token
        elif token.dep_ in ["obl:agent", "obl"]:
            has_por = any(
                child.text.lower() in ["por", "pelo", "pela", "pelos", "pelas"]
                for child in token.lefts
            )
            if has_por or token.dep_ == "obl:agent":
                agente_passiva = token

    if sujeito_paciente and verbo_principal and verbo_auxiliar and agente_passiva:
        det_agente = " ".join(
            [t.text for t in agente_passiva.lefts if t.pos_ == "DET"]
        ).lower()
        if not det_agente:
            prep = [
                t.text.lower()
                for t in agente_passiva.lefts
                if t.text.lower() in ["pelo", "pela", "pelos", "pelas"]
            ]
            if prep:
                if prep[0] == "pelo":
                    det_agente = "o"
                elif prep[0] == "pela":
                    det_agente = "a"
                elif prep[0] == "pelos":
                    det_agente = "os"
                elif prep[0] == "pelas":
                    det_agente = "as"

        novo_sujeito = f"{det_agente} {agente_passiva.text}".strip().capitalize()

        radical = verbo_principal.lemma_[:-2]
        terminacao = verbo_principal.lemma_[-2:]
        aux_text = verbo_auxiliar.text.lower()

        is_plural = agente_passiva.morph.get("Number") == ["Plur"] or (
            det_agente and det_agente[-1] == "s"
        )
        verbo_conjugado = verbo_principal.lemma_

        if aux_text in ["foi", "foram"]:
            if not is_plural:
                if terminacao == "ar":
                    verbo_conjugado = radical + "ou"
                elif terminacao in ["er", "ir"]:
                    verbo_conjugado = radical + "eu"
            else:
                if terminacao == "ar":
                    verbo_conjugado = radical + "aram"
                elif terminacao in ["er", "ir"]:
                    verbo_conjugado = radical + "eram"
        elif aux_text in ["é", "são"]:
            if not is_plural:
                if terminacao == "ar":
                    verbo_conjugado = radical + "a"
                elif terminacao in ["er", "ir"]:
                    verbo_conjugado = radical + "e"
            else:
                if terminacao == "ar":
                    verbo_conjugado = radical + "am"
                elif terminacao in ["er", "ir"]:
                    verbo_conjugado = radical + "em"

        det_paciente = " ".join(
            [t.text for t in sujeito_paciente.lefts if t.pos_ == "DET"]
        )
        novo_objeto = f"{det_paciente} {sujeito_paciente.text}".strip().lower()

        pontuacao = doc[-1].text if doc[-1].is_punct else "."
        return f"{novo_sujeito} {verbo_conjugado} {novo_objeto}{pontuacao}"

    return None


def realizar_parafraseio(
    texto: str,
    nlp_model: Optional[spacy.language.Language] = None,
) -> str:
    doc_pipeline = nlp_model if nlp_model is not None else nlp
    doc = doc_pipeline(texto)

    # _parafrasear_bloco_adverbial assume a responsabilidade total pelas orações e adjuntos temporais.
    resultado = (
        _parafrasear_bloco_adverbial(doc)
        or _parafrasear_titulo_aposto(doc)
        or _parafrasear_voz_passiva_ativa(doc)
        or _parafrasear_voz_ativa_passiva(doc)
        or _parafrasear_deslocamento_adverbio(doc)
    )

    return resultado if resultado else texto


# ==========================================
# MAIN COM TESTE COMPLETO DE TODAS AS POSSIBILIDADES
# ==========================================
# ==========================================
# BATERIA DE TESTES DEDICADA: VOZES VERBAIS (ATIVA <-> PASSIVA)
# ==========================================
if __name__ == "__main__":
    casos_de_teste_vozes = {
        "1. Voz Ativa -> Voz Passiva (Pretérito Perfeito)": [
            "O cachorro mordeu o carteiro.",
            "O programador finalizou o código.",
            "A empresa comprou o terreno.",
            "Os diretores assinaram os contratos.",
        ],
        "2. Voz Passiva -> Voz Ativa (Pretérito Perfeito)": [
            "O carteiro foi mordido pelo cachorro.",
            "O código foi finalizado pelo programador.",
            "O terreno foi comprado pela empresa.",
            "Os contratos foram assinados pelos diretores.",
        ],
        "3. Voz Ativa -> Voz Passiva (Presente do Indicativo)": [
            "A professora corrige a prova.",
            "Os cientistas realizam a pesquisa.",
            "O gato caça o rato.",
        ],
        "4. Voz Passiva -> Voz Ativa (Presente do Indicativo)": [
            "A prova é corrigida pela professora.",
            "A pesquisa é realizada pelos cientistas.",
            "O rato é caçado pelo gato.",
        ],
        "5. Flexão e Concordância de Agente/Particípio (Gênero e Plural)": [
            "A diretora assinou o documento.",
            "Os bombeiros salvaram as vítimas.",
            "As enfermeiras atenderam os pacientes.",
        ],
        "6. Casos de Controle / Bloqueios (Sem conversão esperada)": [
            "O garoto dormiu cedo ontem.",                # Verbo Intransitivo (Sem Objeto)
            "A aluna vai ter que estudar para a prova.",  # Com locução/auxiliar complexo
            "O sol brilha no horizonte.",                 # Verbo Intransitivo
        ],
    }

    print("=" * 75)
    print("      BATERIA DE TESTES: CONVERSÃO DE VOZES VERBAIS (IDA E VOLTA)")
    print("=" * 75 + "\n")

    for categoria, frases in casos_de_teste_vozes.items():
        print(f"--- {categoria.upper()} ---")
        for frase in frases:
            # 1. Teste de Ida
            parafrase = realizar_parafraseio(frase)
            modificada = parafrase != frase
            status_ida = "(OK)" if modificada else "(-)"

            # 2. Teste de Volta (Reversão)
            parafrase_volta = realizar_parafraseio(parafrase)
            voltou_original = parafrase_volta == frase

            if voltou_original:
                status_volta = "(OK)"
            elif parafrase_volta == parafrase:
                status_volta = "(X)"  # Não sofreu nova alteração ao tentar voltar
            else:
                status_volta = "(!)"  # Sofreu uma mutação inesperada

            print(f"Original:   {frase}")
            print(f"Ida:        {parafrase}  {status_ida}")
            print(f"Volta:      {parafrase_volta}  {status_volta}")
            print("-" * 60)
        print("\n")