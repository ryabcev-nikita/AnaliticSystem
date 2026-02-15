from typing import Dict
from ...ai_risk_models.ai_risk_constants.ai_risk_constants import (
    NN_FEATURE,
    NN_THRESHOLD,
    RISK_SCORE,
)


class NNRiskUtils:
    """Утилиты для нейросетевого анализа рисков"""

    @staticmethod
    def clip_extreme_values(feature: str, value: float) -> float:
        """Нормализация экстремальных значений"""
        if feature in ["P/E", "P/B", "P/S", "EV/EBITDA"]:
            if value > NN_FEATURE.PE_MAX:
                return NN_FEATURE.PE_MAX
            elif value < NN_FEATURE.PE_MIN:
                return NN_FEATURE.PE_MIN
        elif feature in ["ROE", "ROA", "ROIC", "NPM"]:
            if value > NN_FEATURE.ROE_MAX:
                return NN_FEATURE.ROE_MAX
            elif value < NN_FEATURE.ROE_MIN:
                return NN_FEATURE.ROE_MIN
        elif feature in ["Debt/Capital", "Debt/EBITDA"]:
            if value > NN_FEATURE.DEBT_CAPITAL_MAX:
                return NN_FEATURE.DEBT_CAPITAL_MAX
            elif value < NN_FEATURE.DEBT_CAPITAL_MIN:
                return NN_FEATURE.DEBT_CAPITAL_MIN
        return value

    @staticmethod
    def get_risk_level_for_feature(feature: str, value: float, stats: Dict) -> int:
        """Определение уровня риска для конкретного признака"""
        median = stats.get("median", 0)
        q1 = stats.get("q1", 0)
        q3 = stats.get("q3", 0)
        iqr = stats.get("iqr", 1)

        if feature in [
            "P/E",
            "P/B",
            "P/S",
            "P/FCF",
            "EV/EBITDA",
            "EV/S",
            "Debt/Capital",
            "Debt/EBITDA",
            "Net Debt/EBITDA",
        ]:
            if value <= NN_THRESHOLD.PE_RISK_FREE:
                return RISK_SCORE.RISK_HIGH
            elif value <= median * NN_THRESHOLD.PE_LOW_RISK:
                return RISK_SCORE.RISK_VERY_LOW
            elif value <= median * NN_THRESHOLD.PE_MEDIUM_RISK:
                return RISK_SCORE.RISK_LOW
            elif value <= median * NN_THRESHOLD.PE_HIGH_RISK:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        elif feature in [
            "ROE",
            "ROA",
            "ROIC",
            "NPM",
            "EBITDA margin",
            "dividend_yield",
            "EPS",
        ]:
            if value <= NN_THRESHOLD.PE_RISK_FREE:
                return RISK_SCORE.RISK_HIGH
            elif value >= median * NN_THRESHOLD.ROE_LOW_RISK:
                return RISK_SCORE.RISK_VERY_LOW
            elif value >= median * NN_THRESHOLD.ROE_MEDIUM_RISK:
                return RISK_SCORE.RISK_LOW
            elif value >= median * NN_THRESHOLD.ROE_HIGH_RISK:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        elif feature in ["Beta", "Бета"]:
            if value < NN_THRESHOLD.BETA_VERY_LOW:
                return RISK_SCORE.RISK_LOW
            elif value < NN_THRESHOLD.BETA_LOW:
                return RISK_SCORE.RISK_VERY_LOW
            elif value < NN_THRESHOLD.BETA_MEDIUM:
                return RISK_SCORE.RISK_LOW
            elif value < NN_THRESHOLD.BETA_HIGH:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        else:
            deviation = abs(value - median) / (iqr if iqr > 0 else 1)
            if deviation < NN_THRESHOLD.DEVIATION_LOW:
                return RISK_SCORE.RISK_VERY_LOW
            elif deviation < NN_THRESHOLD.DEVIATION_MEDIUM:
                return RISK_SCORE.RISK_LOW
            elif deviation < NN_THRESHOLD.DEVIATION_HIGH:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

    @staticmethod
    def get_feature_weight(feature: str) -> float:
        """Получение веса для признака"""
        if feature in RISK_SCORE.HIGH_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_HIGH
        elif feature in RISK_SCORE.MEDIUM_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_MEDIUM
        elif feature in RISK_SCORE.ABOVE_AVG_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_ABOVE_AVG
        else:
            return RISK_SCORE.WEIGHT_BASE
