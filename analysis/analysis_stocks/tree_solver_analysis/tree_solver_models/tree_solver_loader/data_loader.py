# ==================== КЛАСС ЗАГРУЗЧИКА ДАННЫХ ====================
import re
import numpy as np
import pandas as pd
from ...tree_solver_models.tree_solver_constants.tree_solver_constants import CONVERSION


class DataLoader:
    """Загрузка и первичная обработка данных"""

    @staticmethod
    def convert_to_float(value):
        """Конвертация строк с числами в float"""
        if pd.isna(value) or value == "" or value == 0:
            return np.nan
        if isinstance(value, (int, float)):
            return value

        value = str(value).strip()
        value = value.replace(CONVERSION.THOUSAND_SEPARATOR, "").replace(
            CONVERSION.DECIMAL_SEPARATOR, "."
        )

        if CONVERSION.BILLION_PATTERN in value:
            return float(re.sub(r"[^\d.]", "", value)) * CONVERSION.BILLION
        elif CONVERSION.MILLION_PATTERN in value:
            return float(re.sub(r"[^\d.]", "", value)) * CONVERSION.MILLION
        else:
            try:
                return float(re.sub(r"[^\d.-]", "", value))
            except:
                return np.nan

    @staticmethod
    def load_and_clean_data(filepath: str) -> pd.DataFrame:
        """Загрузка и очистка данных"""
        df = pd.read_excel(filepath, sheet_name="Sheet1")

        numeric_columns = [
            "Рыночная капитализация",
            "EV",
            "Выручка",
            "Чистая прибыль",
            "EBITDA",
            "P/E",
            "P/B",
            "P/S",
            "P/FCF",
            "ROE",
            "ROA",
            "ROIC",
            "EV/EBITDA",
            "EV/S",
            "Payot Ratio",
            "NPM",
            "Debt",
            "Debt/Capital",
            "Net_Debt/EBITDA",
            "Debt/EBITDA",
            "EPS",
            "Averange_dividend_yield",
            "Свободный денежный поток",
            "Бета",
            "Дивиденд на акцию",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader.convert_to_float)

        return df
