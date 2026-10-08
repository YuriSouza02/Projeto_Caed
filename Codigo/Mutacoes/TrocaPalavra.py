import os
import random
import nltk
import spacy
from nltk.corpus import wordnet as wn

# Configura o NLTK para buscar os dados na pasta do ambiente virtual local
diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_nltk_data = os.path.join(diretorio_base, '.venv', 'nltk_data')

if os.path.exists(caminho_nltk_data) and caminho_nltk_data not in nltk.data.path:
    nltk.data.path.append(caminho_nltk_data)

# Carrega o modelo de NLP do spaCy em português.
# Mantido em escopo global para evitar o recarregamento oneroso a cada chamada de função.
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    print("Erro: O modelo 'pt_core_news_sm' não foi encontrado.")
    print("Execute: python -m spacy download pt_core_news_sm")
    raise


def obter_sinonimos_antonimos(palavra):
    """
    Busca sinônimos e antônimos de uma palavra utilizando a WordNet.
    Retorna duas listas: uma de sinônimos e outra de antônimos.
    """
    sinonimos = set()
    antonimos = set()

    for syn in wn.synsets(palavra, lang='por'):
        # 1. Coleta os sinônimos diretamente disponíveis em português
        for lemma in syn.lemmas(lang='por'):
            if lemma.name().lower() != palavra.lower():
                sinonimos.add(lemma.name().replace('_', ' '))

        # 2. Coleta antônimos fazendo ponte com o inglês
        # A WordNet pt-BR não mapeia antônimos diretamente de forma robusta. 
        # O código busca os lemas em inglês, extrai os antônimos deles e,
        # em seguida, busca a tradução desses antônimos de volta para o português.
        for lemma_eng in syn.lemmas(lang='eng'):
            for ant in lemma_eng.antonyms():
                for lemma_pt in ant.synset().lemmas(lang='por'):
                    antonimos.add(lemma_pt.name().replace('_', ' '))

    return list(sinonimos), list(antonimos)


def realizar_data_augmentation(texto, prob_troca=0.3, modo='sinonimo'):
    """
    Realiza o aumento de dados (Data Augmentation) substituindo palavras aleatoriamente.
    
    Parâmetros:
    - texto: String original a ser processada.
    - prob_troca: Chance (0.0 a 1.0) de uma palavra válida ser substituída.
    - modo: 'sinonimo' ou 'antonimo'. Define a estratégia de substituição.
    """
    doc = nlp(texto)
    texto_reconstruido = []

    # Restringe as substituições a substantivos, adjetivos, verbos e advérbios
    classes_alvo = ['NOUN', 'ADJ', 'VERB', 'ADV']

    for token in doc:
        # Se a palavra não é das classes alvo ou não caiu na probabilidade, mantém a original
        if token.pos_ not in classes_alvo or random.random() > prob_troca:
            texto_reconstruido.append(token.text_with_ws)
            continue

        # Busca substitutos com base na raiz da palavra (lemma)
        sinonimos, antonimos = obter_sinonimos_antonimos(token.lemma_)
        
        # Define as opções de troca. Se pedir antônimo e não tiver, tenta sinônimo como fallback.
        opcoes = antonimos if (modo == 'antonimo' and antonimos) else sinonimos

        if opcoes:
            nova_palavra = random.choice(opcoes)

            # Preserva a formatação original (maiúsculas/minúsculas)
            if token.is_title:
                nova_palavra = nova_palavra.title()
            elif token.is_upper:
                nova_palavra = nova_palavra.upper()

            # Adiciona a nova palavra e restaura o espaçamento original
            texto_reconstruido.append(nova_palavra + token.whitespace_)
        else:
            # Caso não encontre substitutos disponíveis
            texto_reconstruido.append(token.text_with_ws)

    return "".join(texto_reconstruido)