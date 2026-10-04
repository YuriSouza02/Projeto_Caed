import sys

# Importação corrigida para disponibilizar o nome 'Complexidade' diretamente
from Codigo import Complexidade


def main():
    print("=" * 60)
    print(" " * 12 + "AVALIADOR DE COMPLEXIDADE DE TEXTO")
    print("=" * 60)
    print("Escreva ou cole o seu texto abaixo.")
    print("(Para encerrar o programa, digite 'sair')\n")

    caminho_modelo = "modelo_rf_dificuldade.pkl"

    while True:
        try:
            # Recebe o texto do utilizador pelo terminal
            texto_usuario = input("Cole o texto aqui: ").strip()

            # Condição de paragem
            if texto_usuario.lower() == "sair":
                print("\nA encerrar o avaliador. Até logo!")
                break

            # Validação de texto vazio
            if not texto_usuario:
                print("Aviso: Por favor, insira um texto válido.\n")
                continue

            # Chama a função do módulo Complexidade para prever o nível
            print("A processar a sintaxe, léxico e estilo...")
            nivel_previsto = Complexidade.prever_dificuldade_texto(
                texto_usuario, caminho_modelo
            )

            # Exibe o resultado
            print("-" * 60)
            print(f">>> NÍVEL DE DIFICULDADE PREVISTO PELA IA: {nivel_previsto} <<<")
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"\n[ERRO] Não foi possível classificar o texto: {e}")
            print(
                "Certifique-se de que o ficheiro 'modelo_rf_dificuldade.pkl' existe na pasta."
            )
            print(
                "Se não existir, corra o 'main.py' primeiro para treinar e gerar o modelo."
            )
            break


if __name__ == "__main__":
    main()
