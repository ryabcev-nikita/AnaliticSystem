import os

import pandas as pd

from commodities_models.commodities_constants import (
    END_DATE,
    MARKET_DATA_DIR,
    METALS_DATA_FILES,
    START_DATE,
)


class MetalsDataLoader:
    """Класс для загрузки и предобработки данных по металлам"""

    @staticmethod
    def load_metals_data() -> pd.DataFrame:
        """Загрузка данных из CSV файлов металлов"""
        print(f"   Загрузка данных из: {MARKET_DATA_DIR}")
        data = {}

        for name, file in METALS_DATA_FILES.items():
            file_path = os.path.join(MARKET_DATA_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            # Для некоторых металлов Volume может быть 0 или отсутствовать
            if "Close" in df.columns:
                data[name] = df["Close"]
            else:
                # Если нет Close, используем последнюю доступную колонку
                data[name] = df.iloc[:, 0]

        prices = pd.DataFrame(data)

        # Фильтруем по датам
        mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
        prices = prices[mask]

        # Удаляем строки с пропущенными значениями
        prices = prices.dropna()

        return prices
