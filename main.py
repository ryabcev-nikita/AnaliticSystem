import os
import sys
import logging

# Настраиваем логгирование до всех остальных действий
logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # Теперь импорты внутри try – ошибки при импорте будут пойманы
    from analysis import (
        create_full_model_tree_solver,
        create_model_cluster_analysis,
        create_model_regression_analysis,
    )
    from api.t_invest_api.t_invest_api import get_t_api_data
    from shared.config import ACTIONS_MAIN
except Exception as e:
    logging.exception("Ошибка при импорте модулей")
    # Дублируем в консоль (если окно ещё открыто)
    print("Критическая ошибка при загрузке модулей. Подробности в error.log")
    if sys.platform == "win32":
        os.system("pause")
    else:
        input("Нажмите Enter для выхода...")
    sys.exit(1)


# Далее ваши функции (без изменений)
def do_full_analysis():
    print("---------------------Анализ акций----------------------")
    create_model_regression_analysis()
    create_full_model_tree_solver()
    create_model_cluster_analysis()
    print("-------------------------------------------------------")


def get_data():
    get_t_api_data()


def avaliable_actions():
    print("1. Загрузить данные")
    print("2. Сделать полный анализ российских акций")
    print("3. Сделать регрессионный анализ акций")
    print("4. Сделать кластерный анализ акций")
    print("5. Сделать анализ акций с помощью дерева решений")
    #print("6. Сделать анализ рисков акций с помощью нейронной сети")
    print("0. Выход")


def main():
    while True:
        avaliable_actions()
        action = int(input("Действие: "))
        # ... ваш код ...


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Ошибка во время выполнения")
        print("Ошибка записана в файл error.log")
    finally:
        if sys.platform == "win32":
            os.system("pause")
        else:
            input("Нажмите Enter для выхода...")
