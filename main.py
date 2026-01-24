from shared.config import ACTIONS_MAIN
from analysis.cluster_analysis import create_model_cluster_analysis
from analysis.tree_solver_analysis.tree_solver_analysis import create_model_tree_solver
from analysis.capm_analysis.model_capm import create_model_capm
from analysis.capm_analysis.model_capm_ff import create_model_capm_ff
from analysis.multiplicator_analysis.model_multiplicator import create_model_pe, create_model_ps, create_model_peg, create_model_ev_ebitda, create_model_pbv
from api.parser_smartlab_api import get_fundamental_shares

def main():
    while True:
        avaliable_actions()
        action = int(input("Действие: "))

        if action == ACTIONS_MAIN.get('DOWNLOAD_FUNDAMENTALS_DATA'):
            print("Загрузка фундаментальных показателей")
            get_fundamental_shares()
        elif action == ACTIONS_MAIN.get('DOWNLOAD_TICKERS'):
            print("Загрузка тикеров...")
        elif action == ACTIONS_MAIN.get('DOWNLOAD_DATA_HISTORICAL_PRICES_SHARES'):
            print("Загрузка котировок...")
        elif action == ACTIONS_MAIN.get('CLUSTER_ANALYSIS'):
            print("Кластерный анализ")
            create_model_cluster_analysis()
        elif action == ACTIONS_MAIN.get('TREE_SOLVE_ANALYSIS'):
            print("Анализ с помощью деревья решений")
            create_model_tree_solver()
        elif action == ACTIONS_MAIN.get('CAPM'):
            print("Анализ с помощью модели CAPM")
            create_model_capm()
        elif action == ACTIONS_MAIN.get('CAPM_FF'):
            print("Анализ с помощью модели CAPM_FF")
            create_model_capm_ff()
        elif action == ACTIONS_MAIN.get('MULTIPLICATOR_P_E'):
            print("Анализ с помощью мультипликатора P/E")
            create_model_pe()
        elif action == ACTIONS_MAIN.get('MULTIPLICATOR_P_BV'):
            print("Анализ с помощью мультипликатора P/BV")
            create_model_pbv()
        elif action == ACTIONS_MAIN.get('MULTIPLICATOR_P_S'):
            print("Анализ с помощью мультипликатора P/S")
            create_model_ps()
        elif action == ACTIONS_MAIN.get('MULTIPLICATOR_EV_EBITDA'):
            print("Анализ с помощью мультипликатора EV/EBITDA")
            create_model_ev_ebitda()
        elif action == ACTIONS_MAIN.get('MULTIPLICATOR_PEG'):
            print("Анализ с помощью мультипликатора PEG")
            create_model_peg()                
        elif action == ACTIONS_MAIN.get('EXIT'):
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
    print("9. Анализ акции с помощью мультипликатора P/E")
    print("10. Анализ акции с помощью мультипликатора P/BV")
    print("11. Анализ акции с помощью мультипликатора P/S")
    print("12. Анализ акции с помощью мультипликатора EV/EBITDA")
    print("13. Анализ акции с помощью мультипликатора PEG")
    print("14. Анализ акций с помощью нейросети")
    print("0. Выход")
    
if __name__ == "__main__":
    main()