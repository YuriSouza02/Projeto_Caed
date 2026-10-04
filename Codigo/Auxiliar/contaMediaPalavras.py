from pathlib import Path

def calcular_media_palavras(diretorio_base=None):
    """
    Lê todos os arquivos .txt das 4 pastas especificadas e calcula 
    a média de palavras por arquivo.
    """
    pastas = [
        "1_Ensino_Fundamental_I",
        "2_Ensino_Fundamental_II",
        "3_Ensino_Medio",
        "4_Ensino_Superior"
    ]
    
    # Se não for passado um diretório, localiza a pasta 'DataBase' relativa ao script atual
    if diretorio_base is None:
        # Script está em: Projeto_Caed/Codigo/Auxiliar/contaMediaPalavras.py
        # .parents[2] sobe 3 níveis até a raiz 'Projeto_Caed'
        caminho_base = Path(__file__).resolve().parents[2] / "DataBase"
    else:
        caminho_base = Path(diretorio_base)

    total_palavras = 0
    total_arquivos = 0

    for pasta in pastas:
        caminho_pasta = caminho_base / pasta
        
        if not caminho_pasta.is_dir():
            print(f"Aviso: pasta '{pasta}' não encontrada em '{caminho_pasta}'.")
            continue

        # Busca todos os arquivos .txt dentro da pasta
        for arquivo in caminho_pasta.glob("*.txt"):
            try:
                conteudo = arquivo.read_text(encoding="utf-8")
                palavras = conteudo.split()
                
                total_palavras += len(palavras)
                total_arquivos += 1
            except Exception as e:
                print(f"Erro ao ler o arquivo '{arquivo.name}': {e}")

    if total_arquivos == 0:
        print("Nenhum arquivo TXT foi lido.")
        return 0.0

    media = total_palavras / total_arquivos
    
    print(f"Total de arquivos processados: {total_arquivos}")
    print(f"Total de palavras contadas: {total_palavras}")
    print(f"Média de palavras por arquivo: {media:.2f}")
    
    return media

# Chamada sem argumentos (calcula o caminho automaticamente)
media = calcular_media_palavras()