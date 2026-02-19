import os

import pandas as pd

from indexes_models.indexes_constants import (
    END_DATE,
    MARKET_DATA_DIR,
    MARKET_DATA_FILES,
    START_DATE,
)


class DataLoader:
    """Класс для загрузки и предобработки данных"""

    @staticmethod
    def load_market_data() -> pd.DataFrame:
        """Загрузка данных из CSV файлов"""
        print(f"   Загрузка данных из: {MARKET_DATA_DIR}")
        data = {}

        for name, file in MARKET_DATA_FILES.items():
            file_path = os.path.join(MARKET_DATA_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            data[name] = df["Close"]

        prices = pd.DataFrame(data)

        # Фильтруем по датам
        mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
        prices = prices[mask]

        return prices
