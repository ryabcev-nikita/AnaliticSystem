from analysis.analysis_stocks.tree_solver_analysis.tree_solver_analysis import (
    create_full_model_tree_solver,
)
from analysis.analysis_stocks.ai_analysis.ai_anomaly_detector_ae.ai_anomaly_detector_ae import (
    create_model_ai_anomaly_detector_ae,
)
from analysis.analysis_stocks.ai_analysis.ai_risk_analysis.ai_risk_analysis import (
    create_model_ai_risk_analysis,
)
from analysis.analysis_stocks.cluster_analysis.cluster_analysis import (
    create_model_cluster_analysis,
)
from analysis.analysis_stocks.regression_analysis.regression_analysis import (
    create_model_regression_analysis,
)
from api.t_invest_api import get_full_data_t_api
from shared.config import ACTIONS_MAIN


def do_full_analysis():
    get_full_data_t_api()
    create_model_regression_analysis()
    create_full_model_tree_solver()
    create_model_cluster_analysis()
    create_model_ai_risk_analysis()
    create_model_ai_anomaly_detector_ae()


def main():
    while True:
        avaliable_actions()
        action = int(input("Действие: "))

        if action == ACTIONS_MAIN.get("FULL_ANALYSIS"):
            do_full_analysis()
        elif action == ACTIONS_MAIN.get("EXIT"):
            print("Выход...")
            return True


def avaliable_actions():
    print("1. Сделать полный анализ")
    print("0. Выход")


if __name__ == "__main__":
    main()
