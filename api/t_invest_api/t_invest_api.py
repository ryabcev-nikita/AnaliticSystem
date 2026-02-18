"""
Скрипт для загрузки данных из API Т-Банк Инвестиции
"""

import time
from .constants.config import BONDS_EXCEL_PATH, SHARES_EXCEL_PATH
from .clients.bond_client import BondsClient
from .clients.share_client import SharesClient


def get_t_api_bonds_data():
    # === Облигации ===
    print("\n📊 РАЗДЕЛ: ОБЛИГАЦИИ")
    print("-" * 40)

    bonds_client = BondsClient()

    # Получаем список облигаций
    bonds = bonds_client.get_bonds_list()

    if bonds:
        # Обогащаем ставками купонов
        bonds = bonds_client.enrich_with_coupon_rates(bonds)

        # Сохраняем в Excel
        bonds_client.save_to_excel(bonds, BONDS_EXCEL_PATH)
        print(f"✅ Облигации сохранены в: {BONDS_EXCEL_PATH}")


def get_t_api_shares_data():
    # === Акции ===
    print("\n\n📈 РАЗДЕЛ: АКЦИИ")
    print("-" * 40)

    shares_client = SharesClient()

    # Получаем список акций
    shares = shares_client.get_shares_list()

    if shares:
        # Получаем фундаментальные показатели
        fundamentals = shares_client.get_fundamentals(shares)

        if fundamentals:
            # Сохраняем в Excel
            shares_client.save_fundamentals_to_excel(fundamentals, SHARES_EXCEL_PATH)
            print(f"✅ Акции сохранены в: {SHARES_EXCEL_PATH}")

    print("\n" + "=" * 60)
    print("✅ Скрипт успешно завершен!")


def get_t_api_data():
    """Основная функция"""
    print("🚀 Запуск скрипта для работы с API Т-Банк Инвестиции")
    print("=" * 60)

    # === Облигации ===
    print("\n📊 РАЗДЕЛ: ОБЛИГАЦИИ")
    print("-" * 40)

    bonds_client = BondsClient()

    # Получаем список облигаций
    bonds = bonds_client.get_bonds_list()

    if bonds:
        # Обогащаем ставками купонов
        bonds = bonds_client.enrich_with_coupon_rates(bonds)

        # Сохраняем в Excel
        bonds_client.save_to_excel(bonds, BONDS_EXCEL_PATH)
        print(f"✅ Облигации сохранены в: {BONDS_EXCEL_PATH}")

    print("Ограничение API, ждём 30 секунд...")
    time.sleep(30)

    # === Акции ===
    print("\n\n📈 РАЗДЕЛ: АКЦИИ")
    print("-" * 40)

    shares_client = SharesClient()

    # Получаем список акций
    shares = shares_client.get_shares_list()

    if shares:
        # Получаем фундаментальные показатели
        fundamentals = shares_client.get_fundamentals(shares)

        if fundamentals:
            # Сохраняем в Excel
            shares_client.save_fundamentals_to_excel(fundamentals, SHARES_EXCEL_PATH)
            print(f"✅ Акции сохранены в: {SHARES_EXCEL_PATH}")

    print("\n" + "=" * 60)
    print("✅ Скрипт успешно завершен!")


if __name__ == "__main__":
    get_t_api_data()
