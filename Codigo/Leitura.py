import os
import random
import pandas as pd


def carregar_dados_do_diretorio(
    caminho_base="\.DataBase", limite_por_pasta=None
) -> pd.DataFrame:
    """
    Lê os ficheiros .txt dentro das subpastas da base de dados e constrói um DataFrame.
    Pode limitar a quantidade de ficheiros lidos aleatoriamente por pasta.
    """
    dados = []

    mapeamento_niveis = {
        "1_Ensino_Fundamental_I": 1,
        "2_Ensino_Fundamental_II": 2,
        "3_Ensino_Medio": 3,
        "4_Ensino_Superior": 4,
    }

    if not os.path.exists(caminho_base):
        raise FileNotFoundError(
            f"A pasta '{caminho_base}' não foi encontrada na raiz do projeto."
        )

    for nome_pasta, nivel in mapeamento_niveis.items():
        caminho_pasta = os.path.join(caminho_base, nome_pasta)

        if os.path.exists(caminho_pasta):
            # Mapeia apenas ficheiros com extensão .txt
            ficheiros_disponiveis = [
                f for f in os.listdir(caminho_pasta) if f.endswith(".txt")
            ]

            # Aplica o limite aleatório se especificado pelo utilizador
            if (
                limite_por_pasta is not None
                and len(ficheiros_disponiveis) > limite_por_pasta
            ):
                ficheiros_selecionados = random.sample(
                    ficheiros_disponiveis, limite_por_pasta
                )
            else:
                ficheiros_selecionados = ficheiros_disponiveis

            for nome_ficheiro in ficheiros_selecionados:
                caminho_ficheiro = os.path.join(caminho_pasta, nome_ficheiro)

                try:
                    with open(caminho_ficheiro, "r", encoding="utf-8") as f:
                        texto = f.read().strip()
                        if texto:  # Ignora ficheiros vazios
                            dados.append({"texto": texto, "nivel_dificuldade": nivel})
                except Exception as e:
                    print(f"Erro ao ler o ficheiro {caminho_ficheiro}: {e}")
        else:
            print(
                f"Aviso: A pasta '{nome_pasta}' não foi encontrada dentro de '{caminho_base}'."
            )

    return pd.DataFrame(dados)
