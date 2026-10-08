import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

import Codigo.Complexidade as Complexidade
from Codigo.Leitura import carregar_dados_do_diretorio



def treinar_modelo(df: pd.DataFrame, caminho_modelo: str = "modelo_rf_dificuldade.pkl"):
    print("A extrair características (features) do corpus de treino...")
    df_features = Complexidade.construir_dataset_features(df)

    X = df_features.drop(["nivel_dificuldade"], axis=1)
    y = df_features["nivel_dificuldade"]

    # Correção: Adicionado stratify=y para manter proporções de classes no teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("A treinar Classificador com GridSearchCV...")
    rf_base = RandomForestClassifier(random_state=7, class_weight="balanced")

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=rf_base, param_grid=param_grid, cv=5, n_jobs=-1, scoring="f1_macro"
    )
    grid_search.fit(X_train, y_train)

    modelo_rf = grid_search.best_estimator_
    print(f"\nMelhores parâmetros encontrados: {grid_search.best_params_}")

    y_pred = modelo_rf.predict(X_test)
    print("\nRelatório de Classificação Final:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    importances = modelo_rf.feature_importances_
    features_nomes = X_train.columns
    print("\nImportância de cada métrica no modelo:")
    for nome, imp in sorted(
        zip(features_nomes, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{nome}: {imp:.4f}")

    dados_exportacao = {"modelo": modelo_rf}
    joblib.dump(dados_exportacao, caminho_modelo)
    print(f"\nModelo treinado e guardado em '{caminho_modelo}'.\n")

    return modelo_rf


if __name__ == "__main__":
    entrada = input(
        "Quantos ficheiros deseja ler por nível de dificuldade? (Deixe em branco e prima Enter para ler TODOS): "
    )

    if entrada.strip().isdigit():
        limite = int(entrada.strip())
        print(f"\nConfigurado para ler {limite} ficheiros aleatórios por nível.")
    else:
        limite = None
        print("\nConfigurado para ler TODOS os ficheiros disponíveis.")

    print("A ler os textos da pasta DataBase...")
    df_base = carregar_dados_do_diretorio("DataBase", limite_por_pasta=limite)

    if df_base.empty:
        print("Erro: Nenhum texto foi carregado.")
    else:
        print(f"Sucesso! Foram carregados {len(df_base)} textos no total.\n")
        treinar_modelo(df_base, "modelo_rf_dificuldade.pkl")

        print("-" * 50)
        textos_teste = [
            "A gata bebeu o leite todo da tigela.",
            "A epistemologia subjacente à metodologia empírica requer um escrutínio rigoroso.",
        ]

        for txt in textos_teste:
            nivel_previsto = Complexidade.prever_dificuldade_texto(
                txt, "modelo_rf_dificuldade.pkl"
            )
            print(f"TEXTO: '{txt}'")
            print(f"NÍVEL PREVISTO PELA RANDOM FOREST: {nivel_previsto}\n")