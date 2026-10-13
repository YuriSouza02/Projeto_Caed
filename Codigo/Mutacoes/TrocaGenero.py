# CÓDIGO CORRIGIDO E OTIMIZADO
import os
import re
from typing import Dict, Optional, Set
import spacy

DIRETORIO_RAIZ = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CAMINHO_UNIMORPH = os.path.join(DIRETORIO_RAIZ, "unimorph_pt.tsv")

_NLP = None

PALAVRAS_DIRETAS = {
    "o": "a",
    "a": "o",
    "os": "as",
    "as": "os",
    "um": "uma",
    "uma": "um",
    "uns": "umas",
    "umas": "uns",
    "ele": "ela",
    "ela": "ele",
    "eles": "elas",
    "elas": "eles",
    "dele": "dela",
    "dela": "dele",
    "deles": "delas",
    "delas": "deles",
    "nele": "nela",
    "nela": "nele",
    "neles": "nelas",
    "nelas": "neles",
    "este": "esta",
    "esta": "este",
    "estes": "estas",
    "estas": "estes",
    "esse": "essa",
    "essa": "esse",
    "esses": "essas",
    "essas": "esses",
    "aquele": "aquela",
    "aquela": "aquele",
    "aqueles": "aquelas",
    "aquelas": "aqueles",
    "meu": "minha",
    "minha": "meu",
    "meus": "minhas",
    "minhas": "meus",
    "teu": "tua",
    "tua": "teu",
    "teus": "tuas",
    "tuas": "teus",
    "seu": "sua",
    "sua": "seu",
    "seus": "suas",
    "suas": "seus",
    "nosso": "nossa",
    "nossa": "nosso",
    "nossos": "nossas",
    "nossas": "nossos",
    "vosso": "vossa",
    "vossa": "vosso",
    "vossos": "vossas",
    "vossas": "vossos",
    "ao": "à",
    "à": "ao",
    "aos": "às",
    "às": "aos",
    "do": "da",
    "da": "do",
    "dos": "das",
    "das": "dos",
    "no": "na",
    "na": "no",
    "nos": "nas",
    "nas": "nos",
    "pelo": "pela",
    "pela": "pelo",
    "pelos": "pelas",
    "pelas": "pelos",
    "daquele": "daquela",
    "daquela": "daquele",
    "daqueles": "daquelas",
    "daquelas": "daqueles",
}

EXCECOES_GENERO = {
    # Masculino para Feminino
    "senhor": "senhora",
    "senhores": "senhoras",
    "menino": "menina",
    "meninos": "meninas",
    "garoto": "garota",
    "garotos": "garotas",
    "moço": "moça",
    "moços": "moças",
    "tio": "tia",
    "tios": "tias",
    "amigo": "amiga",
    "amigos": "amigas",
    "filho": "filha",
    "filhos": "filhas",
    "aluno": "aluna",
    "alunos": "alunas",
    "médico": "médica",
    "médicos": "médicas",
    "professor": "professora",
    "professores": "professoras",
    "brincalhão": "brincalhona",
    "brincalhões": "brincalhonas",
    "campeão": "campeã",
    "campeões": "campeãs",
    "francês": "francesa",
    "franceses": "francesas",
    "inglês": "inglesa",
    "ingleses": "inglesas",
    "português": "portuguesa",
    "portugueses": "portuguesas",
    "japonês": "japonesa",
    "japonêses": "japonesas",
    "chinês": "chinesa",
    "chinêses": "chinesas",
    "irmão": "irmã",
    "irmãos": "irmãs",
    "órfão": "órfã",
    "órfãos": "órfãs",
    "patrão": "patroa",
    "patrões": "patroas",
    "leão": "leoa",
    "leões": "leoas",
    "comilão": "comilona",
    "comilões": "comilonas",
    "chorão": "chorona",
    "chorões": "choronas",
    "rei": "rainha",
    "reis": "rainhas",
    "príncipe": "princesa",
    "príncipes": "princesas",
    "homem": "mulher",
    "homens": "mulheres",
    "ator": "atriz",
    "atores": "atrizes",
    "pai": "mãe",
    "pais": "mães",
    "avô": "avó",
    "avôs": "avós",
    "padrasto": "madrasta",
    "padrastos": "madrastas",
    "cunhado": "cunhada",
    "cunhados": "cunhadas",
    "genro": "nora",
    "genros": "noras",
    "padrinho": "madrinha",
    "padrinhos": "madrinhas",
    # Feminino para Masculino
    "senhora": "senhor",
    "senhoras": "senhores",
    "menina": "menino",
    "meninas": "meninos",
    "garota": "garoto",
    "garotas": "garotos",
    "moça": "moço",
    "moças": "moços",
    "tia": "tio",
    "tias": "tios",
    "amiga": "amigo",
    "amigas": "amigos",
    "filha": "filho",
    "filhas": "filhos",
    "aluna": "aluno",
    "alunas": "alunos",
    "médica": "médico",
    "médicas": "médicos",
    "professora": "professor",
    "professoras": "professores",
    "brincalhona": "brincalhão",
    "brincalhonas": "brincalhões",
    "campeã": "campeão",
    "campeãs": "campeões",
    "francesa": "francês",
    "francesas": "franceses",
    "inglesa": "inglês",
    "inglesas": "ingleses",
    "portuguesa": "português",
    "portuguesas": "portugueses",
    "japonesa": "japonês",
    "japonesas": "japonêses",
    "chinesa": "chinês",
    "chinesas": "chinêses",
    "irmã": "irmão",
    "irmãs": "irmãos",
    "órfã": "órfão",
    "órfãs": "órfãos",
    "patroa": "patrão",
    "patroas": "patrões",
    "leoa": "leão",
    "leoas": "leões",
    "comilona": "comilão",
    "comilonas": "comilões",
    "chorona": "chorão",
    "choronas": "chorões",
    "rainha": "rei",
    "rainhas": "reis",
    "princesa": "príncipe",
    "princesas": "príncipes",
    "mulher": "homem",
    "mulheres": "homens",
    "atriz": "ator",
    "atrizes": "atores",
    "mãe": "pai",
    "mães": "pais",
    "avó": "avô",
    "avós": "avôs",
    "madrasta": "padrasto",
    "madrastas": "padrastos",
    "cunhada": "cunhado",
    "cunhadas": "cunhados",
    "nora": "genro",
    "noras": "genros",
    "madrinha": "padrinho",
    "madrinhas": "padrinhos",
}

# Palavras invariáveis para evitar mutações acidentais (ex: cedo -> ceda)
INVARIAVEIS = {
    "cedo",
    "tarde",
    "hoje",
    "ontem",
    "amanhã",
    "aqui",
    "ali",
    "lá",
    "acolá",
    "sempre",
    "nunca",
    "já",
    "não",
    "sim",
    "talvez",
    "apenas",
    "quase",
    "logo",
    "longe",
    "perto",
}


def obter_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("pt_core_news_sm")
        except Exception:
            _NLP = spacy.blank("pt")
    return _NLP


def inverter_genero_palavra(token, genero_alvo_forcado: Optional[str] = None) -> str:
    texto_lower = token.text.lower()

    if texto_lower in PALAVRAS_DIRETAS:
        return PALAVRAS_DIRETAS[texto_lower]
    if texto_lower in EXCECOES_GENERO:
        return EXCECOES_GENERO[texto_lower]

    generos_spacy = token.morph.get("Gender")
    if generos_spacy:
        genero_atual = generos_spacy[0]
    else:
        if texto_lower.endswith(
            (
                "a",
                "as",
                "ora",
                "oras",
                "ã",
                "ãs",
                "esa",
                "esas",
                "à",
                "às",
                "da",
                "das",
                "na",
                "nas",
                "pela",
                "pelas",
            )
        ):
            genero_atual = "Fem"
        elif texto_lower.endswith(
            (
                "o",
                "os",
                "or",
                "ores",
                "ão",
                "ões",
                "ês",
                "eses",
                "ao",
                "aos",
                "do",
                "dos",
                "no",
                "nos",
                "pelo",
                "pelos",
            )
        ):
            genero_atual = "Masc"
        else:
            return token.text

    genero_alvo = (
        genero_alvo_forcado.upper()
        if genero_alvo_forcado
        else ("FEM" if genero_atual == "Masc" else "MASC")
    )

    if (genero_atual == "Masc" and genero_alvo == "MASC") or (
        genero_atual == "Fem" and genero_alvo == "FEM"
    ):
        return token.text

    if genero_alvo == "FEM":
        if texto_lower.endswith("ão"):
            return texto_lower[:-2] + "ã"
        elif texto_lower.endswith("ões"):
            return texto_lower[:-3] + "ãs"
        elif texto_lower.endswith("ês"):
            return texto_lower[:-2] + "esa"
        elif texto_lower.endswith("eses"):
            return texto_lower[:-4] + "esas"
        elif texto_lower.endswith("or"):
            return texto_lower + "a"
        elif texto_lower.endswith("ores"):
            return texto_lower[:-2] + "as"
        elif texto_lower.endswith("o"):
            return texto_lower[:-1] + "a"
        elif texto_lower.endswith("os"):
            return texto_lower[:-2] + "as"
    elif genero_alvo == "MASC":
        if texto_lower.endswith("esa"):
            return texto_lower[:-3] + "ês"
        elif texto_lower.endswith("esas"):
            return texto_lower[:-4] + "eses"
        elif texto_lower.endswith("ora"):
            return texto_lower[:-1]
        elif texto_lower.endswith("oras"):
            return texto_lower[:-2]
        elif texto_lower.endswith("ã"):
            return texto_lower[:-1] + "ão"
        elif texto_lower.endswith("ãs"):
            return texto_lower[:-2] + "ões"
        elif texto_lower.endswith("a"):
            return texto_lower[:-1] + "o"
        elif texto_lower.endswith("as"):
            return texto_lower[:-2] + "os"

    return token.text


def formatar_caixa_texto(token_original, nova_palavra: str) -> str:
    if token_original.text.isupper():
        return nova_palavra.upper()
    elif token_original.text.istitle():
        return nova_palavra.title()
    return nova_palavra.lower()


def eh_elemento_animado(token) -> bool:
    """
    Identifica possíveis elementos animados com base em exceções cadastradas,
    classes gramaticais e sufixos associados a substantivos que designam
    pessoas ou agentes. Pronomes e nomes próprios são considerados animados,
    enquanto uma lista de substantivos inanimados comuns impede classificações
    indevidas.
    """

    texto_lower = token.text.lower()
    if texto_lower in EXCECOES_GENERO or token.lemma_.lower() in EXCECOES_GENERO:
        return True
    if token.pos_ in ["PRON", "PROPN"]:
        return True
    if token.pos_ == "NOUN":
        sufixos_animados = (
            "dor",
            "dora",
            "tor",
            "triz",
            "sor",
            "sora",
            "eiro",
            "eira",
            "ista",
            "ente",
            "ano",
            "ana",
            "ino",
            "ina",
            "ato",
            "ata",
        )
        inanimados_comuns = {
            "computador",
            "madeira",
            "campeonato",
            "estado",
            "carro",
            "casa",
        }
        if texto_lower in inanimados_comuns:
            return False
        if texto_lower.endswith(sufixos_animados):
            return True
    return False


def encontrar_nucleo_regente(
    token, nucleos_modificados: Dict[int, str]
) -> Optional[spacy.tokens.Token]:
    """
    Identifica o núcleo nominal ou pronominal associado ao token, priorizando
    uma cabeça já identificada como elemento animado.
    """
    if token.head.i in nucleos_modificados:
        return token.head

    if token.head.pos_ in ["NOUN", "PRON", "PROPN"]:
        return token.head

    if token.head.pos_ in ["VERB", "AUX"]:
        for child in token.head.children:
            if child.dep_ in ["nsubj", "nsubj:pass"] and (
                child.pos_ in ["NOUN", "PRON", "PROPN"]
                or child.i in nucleos_modificados
            ):
                return child
    return None


def trocar_genero(
    texto: str, nlp_model: Optional[spacy.language.Language] = None
) -> str:
    """
    Altera o gênero gramatical de elementos animados e tenta propagar a
    transformação para determinantes, adjetivos e outros termos relacionados.
    """
    if not texto or not texto.strip():
        return texto
    doc_pipeline = nlp_model if nlp_model is not None else obter_nlp()
    doc = doc_pipeline(texto)

    nucleos_modificados: Dict[int, str] = {}

    for token in doc:
        if eh_elemento_animado(token):
            generos = token.morph.get("Gender")
            genero_atual = (
                generos[0]
                if generos
                else ("Fem" if token.text.lower().endswith(("a", "as")) else "Masc")
            )
            nucleos_modificados[token.i] = "MASC" if genero_atual == "Fem" else "FEM"

    resultado = []

    for token in doc:
        texto_lower = token.text.lower()
        palavra_transformada = None

        # Evita alterar verbos, advérbios, conjunções e auxiliares, além de palavras cadastradas como invariáveis.
        if (
            token.pos_ in ["VERB", "ADV", "SCONJ", "CCONJ", "AUX"]
            or texto_lower in INVARIAVEIS
        ):
            resultado.append(token.text_with_ws)
            continue

        if token.i in nucleos_modificados:
            genero_alvo = nucleos_modificados[token.i]
            palavra_transformada = inverter_genero_palavra(
                token, genero_alvo_forcado=genero_alvo
            )
        else:
            # Permite a alteração de preposições apenas quando elas estão explicitamente mapeadas.
            is_adp_transformable = (
                token.pos_ == "ADP" and texto_lower in PALAVRAS_DIRETAS
            )

            if (
                token.pos_ in ["DET", "ADJ"]
                or is_adp_transformable
                or texto_lower in PALAVRAS_DIRETAS
            ):
                nucleo_regente = encontrar_nucleo_regente(token, nucleos_modificados)
                if nucleo_regente and nucleo_regente.i in nucleos_modificados:
                    genero_alvo = nucleos_modificados[nucleo_regente.i]
                    palavra_transformada = inverter_genero_palavra(
                        token, genero_alvo_forcado=genero_alvo
                    )
                elif token.pos_ == "PRON":
                    palavra_transformada = inverter_genero_palavra(token)
                elif token.dep_ == "conj":
                    ancestor = encontrar_nucleo_regente(token.head, nucleos_modificados)
                    if ancestor and ancestor.i in nucleos_modificados:
                        palavra_transformada = inverter_genero_palavra(
                            token, genero_alvo_forcado=nucleos_modificados[ancestor.i]
                        )

        if palavra_transformada:
            palavra_final = formatar_caixa_texto(token, palavra_transformada)
            resultado.append(palavra_final + token.whitespace_)
        else:
            resultado.append(token.text_with_ws)

    return "".join(resultado)


# ==========================================
# BATERIA COMPLETA DE TESTES DE TROCA DE GÊNERO
# ==========================================
if __name__ == "__main__":
    casos_de_teste = {
        "1. Troca Direta (Masculino <-> Feminino)": [
            "O menino inteligente estuda na escola.",
            "A garota simpática encontrou a tia na praça.",
            "Os professores novos conversaram com os alunos.",
        ],
        "2. Animação + Inanimado (Preservação de Objetos/Locais)": [
            "O menino alto comprou uma casa amarela.",
            "A médica talentosa vendeu o carro vermelho.",
            "O gato preto dormiu na mesa de madeira.",
            "A aluna nova comprou um livro antigo na biblioteca.",
        ],
        "3. Pronomes, Artigos e Contrações (PALAVRAS_DIRETAS)": [
            "Ele falou comigo sobre o filho dele.",
            "Aquele moço é o nosso vizinho.",
            "Nós fomos ao parque com o meu irmão.",
            "Pelo visto, este aluno é do grupo do professor.",
        ],
        "4. Exceções e Pares Relacionais Irregulares (EXCECOES_GENERO)": [
            "O rei e o príncipe conversaram com o pai do ator.",
            "O homem é o genro daquela senhora.",
            "O avô e o padrinho ajudaram o garoto.",
            "A rainha é a mãe da princesa e a madrinha da menina.",
        ],
        "5. Adjetivos Predicativos (Conexão via Verbo de Ligação)": [
            "Ele é um médico muito dedicado e mora em um apartamento grande.",
            "A professora permaneceu calada e atenta durante a palestra.",
            "Os alunos ficaram muito felizes e motivados com o resultado.",
        ],
        "6. Preservação da Caixa de Texto (Capitalização e Maiúsculas)": [
            "O PROFESSOR É MUITO EXIGENTE COM OS ALUNOS.",
            "Ele É Um Médico Muito Dedicado.",
            "A MÃE DO GAROTO CHEGOU CEDO.",
        ],
        "7. Pluralidade e Regras de Sufixos (-ão, -ês, -or)": [
            "O garoto brincalhão e o leão faminto.",
            "Os franceses e os ingleses venceram o campeonato.",
            "O professor lecionou para o aluno comilão.",
        ],
        "8. Casos de Controle (Apenas Inanimados / Sem Alteração Esperada)": [
            "A mesa de madeira fica na sala azul.",
            "O sol brilha sobre a cidade grande.",
            "O computador antigo travou na mesa nova.",
        ],
    }

    print("=" * 80)
    print("       BATERIA DE TESTES COMPLETA: TROCA DE GÊNERO SINTÁTICA (IDA E VOLTA)")
    print("=" * 80 + "\n")

    total_frases = 0
    total_modificadas = 0
    total_revertidas = 0

    for categoria, frases in casos_de_teste.items():
        print(f"--- {categoria.upper()} ---")
        for frase in frases:
            # Teste de Ida
            resultado = trocar_genero(frase)
            # Teste de Volta (Reversão)
            revertido = trocar_genero(resultado)

            total_frases += 1
            foi_modificada = resultado != frase
            voltou_original = revertido == frase

            # Status de Ida
            if foi_modificada:
                total_modificadas += 1
                status_ida = "(OK)"
            else:
                status_ida = "(-)"

            # Status de Volta
            if voltou_original:
                total_revertidas += 1
                status_volta = "(OK)"
            else:
                status_volta = "(X)"

            print(f"Original:   {frase}")
            print(f"Ida:        {resultado}  {status_ida}")
            print(f"Volta:      {revertido}  {status_volta}")
            print("-" * 60)
        print("\n")
