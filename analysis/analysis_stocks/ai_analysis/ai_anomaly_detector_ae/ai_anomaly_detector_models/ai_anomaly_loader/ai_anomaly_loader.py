# ==================== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ====================


import pandas as pd

from ...ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_COLUMN,
)


class AEDataLoader:
    """Загрузка и подготовка данных для автоэнкодера"""

    @staticmethod
    def load_and_prepare_excel_data(file_path):
        """Загрузка и подготовка данных из Excel файла"""
        df = pd.read_excel(file_path, sheet_name="Sheet1")

        for col in AE_COLUMN.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(
                        AE_COLUMN.BILLION_SUFFIX, AE_COLUMN.BILLION_REPLACE, regex=False
                    )
                )
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(
                        AE_COLUMN.MILLION_SUFFIX, AE_COLUMN.MILLION_REPLACE, regex=False
                    )
                )
                df[col] = df[col].astype(str).str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["dividend_yield"] = df["Дивидендная доходность"] / 100

        for old_col, new_col in AE_COLUMN.COLUMN_MAPPING.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        return df
