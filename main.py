from analysis.analysis_stocks.cluster_analysis.cluster_analysis import (
    create_model_cluster_analysis_with_portfolio,
)
from analysis.analysis_stocks.tree_solver_analysis.tree_solver_analysis import (
    create_model_tree_solver,
)
from shared.config import ACTIONS_MAIN
from api.parser_smartlab_api import get_fundamental_shares


def main():
    while True:
        avaliable_actions()
        action = int(input("Действие: "))

        if action == ACTIONS_MAIN.get("DOWNLOAD_FUNDAMENTALS_DATA"):
            print("Загрузка фундаментальных показателей")
            get_fundamental_shares()
        elif action == ACTIONS_MAIN.get("DOWNLOAD_TICKERS"):
            print("Загрузка тикеров...")
        elif action == ACTIONS_MAIN.get("DOWNLOAD_DATA_HISTORICAL_PRICES_SHARES"):
            print("Загрузка котировок...")
        elif action == ACTIONS_MAIN.get("CLUSTER_ANALYSIS"):
            print("Кластерный анализ")
            portfolios, clusters = create_model_cluster_analysis_with_portfolio()
        elif action == ACTIONS_MAIN.get("TREE_SOLVE_ANALYSIS"):
            print("Анализ с помощью деревья решений")
            create_model_tree_solver()
        elif action == ACTIONS_MAIN.get("CAPM"):
            print("Анализ с помощью модели CAPM")
        elif action == ACTIONS_MAIN.get("CAPM_FF"):
            print("Анализ с помощью модели CAPM_FF")
        elif action == ACTIONS_MAIN.get("MULTIPLICATORS"):
            print("Анализ с помощью мультипликаторов")
        elif action == ACTIONS_MAIN.get("EXIT"):
            print("Выход...")
            return True


def avaliable_actions():
    print("1. Загрузить фундаментальные данные по акциям")
    print("2. Загрузить тикеры акций")
    print("3. Загрузить котировки акций за последний год")
    print("4. Загрузить котировки товаров")
    print("5. Кластерный анализ акций")
    print("6. Анализ акций с помощью деревья решений")
    print("7. Анализ акции с помощью CAPM")
    print("8. Анализ акции с помощью CAPM Fam-French")
    print("9. Анализ акции с помощью мультипликаторов")
    print("0. Выход")


if __name__ == "__main__":
    main()
