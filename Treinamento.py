import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Importações diretas dos módulos que estão na mesma pasta
import Codigo.Complexidade as Complexidade
from Codigo.Leitura import carregar_dados_do_diretorio


def treinar_modelo(df: pd.DataFrame, caminho_modelo: str = "modelo_rf_dificuldade.pkl"):
    """
    O QUE FAZ: Treina, otimiza e avalia um classificador Random Forest para estimar
    o nível de dificuldade de leitura de textos, exportando o modelo final ajustado.

    METODOLOGIA: Aplica um pipeline completo de aprendizado de máquina supervisionado:
    1. Vetorização do Corpus: Transforma cada texto do DataFrame em um vetor de
       métrica através de `construir_dataset_features`.
    2. Divisão dos Dados: Separa os atributos preditores (X) do rótulo alvo (y).
    3. Otimização de Hiperparâmetros: Instancia um algoritmo Random Forest e executa busca.
    4. Avaliação de Desempenho: Preve os rótulos do conjunto de teste retido,
       exibe o relatório de classificação e ordena a importância relativa de cada
       métrica sintático-léxica na decisão do modelo.
    """
    print("A extrair características (features) do corpus de treino...")

    # Extração e construção do conjunto de características
    df_features = Complexidade.construir_dataset_features(df)

    # Separação da matriz de atributos preditores (X) e da variável alvo (y)
    X = df_features.drop(["nivel_dificuldade"], axis=1)
    y = df_features["nivel_dificuldade"]

    # Particionamento dos dados em subconjuntos de treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("A treinar Classificador com GridSearchCV...")
    # Configuração do estimador base com suporte a classes desbalanceadas
    rf_base = RandomForestClassifier(random_state=7, class_weight="balanced")

    # Definição do espaço de busca de hiperparâmetros
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2],
    }

    # Configuração e execução da busca em grade com validação cruzada 5-fold
    grid_search = GridSearchCV(
        estimator=rf_base, param_grid=param_grid, cv=5, n_jobs=-1, scoring="f1_macro"
    )
    grid_search.fit(X_train, y_train)

    # Seleção do melhor modelo encontrado durante o GridSearch
    modelo_rf = grid_search.best_estimator_
    print(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")

    # Avaliação de desempenho do modelo no subconjunto de teste
    y_pred = modelo_rf.predict(X_test)
    print("\nRelatório de Classificação Final:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cálculo e exibição da importância relativa das métricas linguísticas
    importances = modelo_rf.feature_importances_
    features_nomes = X_train.columns
    print("\nImportância de cada métrica no modelo:")
    for nome, imp in sorted(
        zip(features_nomes, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{nome}: {imp:.4f}")

    # Serialização do modelo treinado e persistência do artefato em disco
    dados_exportacao = {"modelo": modelo_rf}
    joblib.dump(dados_exportacao, caminho_modelo)
    print(f"\nModelo treinado e guardado em '{caminho_modelo}'.\n")

    return modelo_rf


if __name__ == "__main__":
    # Interação com o utilizador para definir o limite de amostragem por nível
    entrada = input(
        "Quantos ficheiros deseja ler por nível de dificuldade? (Deixe em branco e prima Enter para ler TODOS): "
    )

    # Tratamento e validação do parâmetro de limite de leitura
    if entrada.strip().isdigit():
        limite = int(entrada.strip())
        print(f"\nConfigurado para ler {limite} ficheiros aleatórios por nível.")
    else:
        limite = None
        print("\nConfigurado para ler TODOS os ficheiros disponíveis.")

    # Carregamento do corpus textual a partir do diretório base
    print("A ler os textos da pasta DataBase...")
    df_base = carregar_dados_do_diretorio("DataBase", limite_por_pasta=limite)

    # Validação do conjunto de dados e execução do pipeline de treino e inferência
    if df_base.empty:
        print(
            "Erro: Nenhum texto foi carregado. Verifique se existem ficheiros .txt nas subpastas."
        )
    else:
        print(
            f"Sucesso! Foram carregados {len(df_base)} textos no total para o treino.\n"
        )

        # Treinamento, otimização e persistência em disco do modelo Random Forest
        treinar_modelo(df_base, "modelo_rf_dificuldade.pkl")

        print("-" * 50)
        # Amostras de verificação para validação do modelo preditivo
        textos_teste = [
            "A gata bebeu o leite todo da tigela.",
            "A epistemologia subjacente à metodologia empírica requer um escrutínio rigoroso.",
        ]

        # Execução das predições de teste no modelo treinado
        for txt in textos_teste:
            nivel_previsto = Complexidade.prever_dificuldade_texto(
                txt, "modelo_rf_dificuldade.pkl"
            )
            print(f"TEXTO: '{txt}'")
            print(f"NÍVEL PREVISTO PELA RANDOM FOREST: {nivel_previsto}\n")
