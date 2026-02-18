# Токен API
TOKEN = "t.V4QVXUA5khrTJcMQNsCCDC3IfD94uJA5Yj_FpR8UfaMs3KxSY0tlIlSDe3ix6G7CcKYMbfQTNlLSWR2l1aHQjQ"

import os


def find_root_dir(marker="main.py"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, marker)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Достигли корня файловой системы
            return None
        current_dir = parent_dir


# ROOT_DIR = find_root_dir()

# Пути к директориям
PARENT_DIR = find_root_dir()
DATA_DIR = PARENT_DIR + "/" + "data"

# Пути к файлам
SHARES_EXCEL_PATH = DATA_DIR + "/" + "fundamentals_shares.xlsx"
BONDS_EXCEL_PATH = DATA_DIR + "/" + "bonds_data.xlsx"
SHARES_JSON_PATH = DATA_DIR + "/" + "shares.json"
BONDS_JSON_PATH = DATA_DIR + "/" + "bonds_data.json"

# Настройки пакетной обработки
CHUNK_SIZE = 30
DELAY_BETWEEN_CHUNKS = 3  # секунды
DELAY_BETWEEN_REQUESTS = 0.1  # секунды
MAX_RETRIES = 3
RETRY_DELAY = 10  # секунды

# Валюты для фильтрации
TARGET_CURRENCY = "rub"
