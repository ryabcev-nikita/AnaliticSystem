from analysis import (
    create_full_model_tree_solver,
    create_model_ai_risk_analysis,
    create_model_cluster_analysis,
    create_model_regression_analysis,
)
from api.t_invest_api.t_invest_api import get_t_api_data
from shared.config import ACTIONS_MAIN


def do_full_analysis():
    print("---------------------Анализ акций----------------------")
    create_model_regression_analysis()
    create_full_model_tree_solver()
    create_model_cluster_analysis()
    create_model_ai_risk_analysis()
    print("-------------------------------------------------------")


def get_data():
    get_t_api_data()


def main():
    while True:
        avaliable_actions()
        action = int(input("Действие: "))

        if action == ACTIONS_MAIN.get("DOWNLOAD_DATA"):
            get_data()
        elif action == ACTIONS_MAIN.get("FULL_ANALYSIS"):
            do_full_analysis()
        elif action == ACTIONS_MAIN.get("REGRESSION_ANALYSIS"):
            create_model_regression_analysis()
        elif action == ACTIONS_MAIN.get("CLUSTER_ANALYSIS"):
            create_model_cluster_analysis()
        elif action == ACTIONS_MAIN.get("TREE_SOLVER_ANALYSIS"):
            create_full_model_tree_solver()
        elif action == ACTIONS_MAIN.get("RISK_AI_ANALYSIS"):
            create_model_ai_risk_analysis()
        elif action == ACTIONS_MAIN.get("EXIT"):
            print("Выход...")
            return True
        else:
            print("Выход...")
            return True


def avaliable_actions():
    print("1. Загрузить данные")
    print("2. Сделать полный анализ российских акций")
    print("3. Сделать регрессионый анализ акций")
    print("4. Сделать кластерный анализ акций")
    print("5. Сделать анализ акций с помощью дерева решений")
    print("6. Сделать анализ рисков акций с помощью нейронной сети")
    print("0. Выход")


if __name__ == "__main__":
    main()
