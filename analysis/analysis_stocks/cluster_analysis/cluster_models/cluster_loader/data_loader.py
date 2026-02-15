# ==================== КЛАСС ЗАГРУЗЧИКА ДАННЫХ ====================


import re
import numpy as np
import pandas as pd
from ...cluster_models.cluster_constants.cluster_constants import (
    SECTOR_KEYWORDS_CLUSTER,
    SECTOR_NAMES_CLUSTER,
)


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
        value = value.replace(" ", "").replace(",", ".")

        if "млрд" in value:
            return float(re.sub(r"[^\d.]", "", value)) * 1e9
        elif "млн" in value:
            return float(re.sub(r"[^\d.]", "", value)) * 1e6
        else:
            try:
                return float(re.sub(r"[^\d.-]", "", value))
            except:
                return np.nan

    @staticmethod
    def load_and_clean_data(filepath: str) -> pd.DataFrame:
        """Загрузка и очистка данных"""
        df = pd.read_excel(filepath, sheet_name="Sheet1")

        column_mapping = {
            "Тикер": "Ticker",
            "Название": "Company",
            "Рыночная капитализация": "Market_Cap",
            "P/E": "PE",
            "P/B": "PB",
            "P/S": "PS",
            "P/FCF": "PFCF",
            "ROE": "ROE",
            "ROA": "ROA",
            "ROIC": "ROIC",
            "EV/EBITDA": "EV_EBITDA",
            "Averange_dividend_yield": "Div_Yield",
            "Бета": "Beta",
            "Debt/Capital": "Debt_Capital",
            "Свободный денежный поток": "FCF",
            "Чистая прибыль": "Net_Income",
            "Выручка": "Revenue",
        }

        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)

        numeric_columns = [
            "Market_Cap",
            "PE",
            "PB",
            "PS",
            "PFCF",
            "ROE",
            "ROA",
            "ROIC",
            "EV_EBITDA",
            "Div_Yield",
            "Beta",
            "Debt_Capital",
            "FCF",
            "Net_Income",
            "Revenue",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader.convert_to_float)

        df["Sector"] = df["Company"].apply(DataLoader.assign_sector)

        return df

    @staticmethod
    def assign_sector(name: str) -> str:
        """Определение сектора компании"""
        if pd.isna(name):
            return SECTOR_NAMES_CLUSTER.OTHER

        name = str(name).lower()

        sector_mappings = [
            (SECTOR_KEYWORDS_CLUSTER.BANKS, SECTOR_NAMES_CLUSTER.BANKS),
            (SECTOR_KEYWORDS_CLUSTER.OIL_GAS, SECTOR_NAMES_CLUSTER.OIL_GAS),
            (SECTOR_KEYWORDS_CLUSTER.METALS, SECTOR_NAMES_CLUSTER.METALS),
            (SECTOR_KEYWORDS_CLUSTER.ENERGY, SECTOR_NAMES_CLUSTER.ENERGY),
            (SECTOR_KEYWORDS_CLUSTER.TELECOM, SECTOR_NAMES_CLUSTER.TELECOM),
            (SECTOR_KEYWORDS_CLUSTER.RETAIL, SECTOR_NAMES_CLUSTER.RETAIL),
            (SECTOR_KEYWORDS_CLUSTER.CHEMICAL, SECTOR_NAMES_CLUSTER.CHEMICAL),
            (SECTOR_KEYWORDS_CLUSTER.IT, SECTOR_NAMES_CLUSTER.IT),
        ]

        for keywords, sector_name in sector_mappings:
            if any(word in name for word in keywords):
                return sector_name

        return SECTOR_NAMES_CLUSTER.OTHER
