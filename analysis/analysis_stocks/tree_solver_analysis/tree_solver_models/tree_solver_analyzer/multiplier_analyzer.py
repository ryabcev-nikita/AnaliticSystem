# ==================== ДОПОЛНИТЕЛЬНЫЙ КЛАСС ДЛЯ АНАЛИЗА МУЛЬТИПЛИКАТОРОВ ====================


from typing import Dict
import pandas as pd

from tree_solver_models.tree_solver_constants.tree_solver_constants import CONVERSION


class MultiplierAnalyzer:
    """Детальный анализ мультипликаторов"""

    @staticmethod
    def analyze_sector_multipliers(df: pd.DataFrame) -> pd.DataFrame:
        """Анализ мультипликаторов по секторам"""
        sector_stats = []

        for sector in df["Сектор"].unique():
            sector_df = df[df["Сектор"] == sector]

            stats = {
                "Сектор": sector,
                "Количество": len(sector_df),
                "P/E_медиана": sector_df["P/E"].median(),
                "P/B_медиана": sector_df["P/B"].median(),
                "ROE_медиана": sector_df["ROE"].median(),
                "Див.доходность_медиана": sector_df["Дивидендная доходность"].median(),
                "Бета_медиана": sector_df["Бета"].median(),
                "Капитализация_медиана": sector_df["Рыночная капитализация"].median()
                / CONVERSION.BILLION,
            }
            sector_stats.append(stats)

        return pd.DataFrame(sector_stats)

    @staticmethod
    def find_best_values(df: pd.DataFrame) -> Dict:
        """Поиск лучших значений мультипликаторов"""
        best_values = {
            "Минимальный P/E": df[df["P/E"] > 0]["P/E"].min(),
            "Минимальный P/B": df[df["P/B"] > 0]["P/B"].min(),
            "Максимальный ROE": df["ROE"].max(),
            "Максимальная див.доходность": df["Дивидендная доходность"].max(),
            "Минимальная бета": df[df["Бета"] > 0]["Бета"].min(),
        }
        return best_values
