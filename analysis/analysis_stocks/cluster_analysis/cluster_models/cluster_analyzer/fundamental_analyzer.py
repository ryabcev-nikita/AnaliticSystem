# ==================== КЛАСС ФУНДАМЕНТАЛЬНОГО АНАЛИЗА ====================


import pandas as pd
from cluster_models.cluster_constants.cluster_constants import (
    CLUSTER_THRESHOLDS,
    PORTFOLIO_CLUSTER,
    RETURN_PREMIUMS_CLUSTER,
    RISK_PREMIUMS_CLUSTER,
    SCORING,
)


class FundamentalAnalyzer:
    """Расчет фундаментальных метрик и скоринга"""

    @staticmethod
    def calculate_value_score(row: pd.Series) -> float:
        """Расчет скора стоимости (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("PE")) and row["PE"] > 0:
            if row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_STRONG:
                score += SCORING.PE_DEEP_VALUE_BONUS
            elif row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_MEDIUM:
                score += SCORING.PE_VALUE_BONUS
            elif row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_WEAK:
                score += SCORING.PE_FAIR_BONUS
            elif row["PE"] > CLUSTER_THRESHOLDS.SCORE_PE_OVER:
                score += SCORING.PE_OVER_PENALTY

        if pd.notna(row.get("PB")) and row["PB"] > 0:
            if row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_STRONG:
                score += SCORING.PB_DEEP_VALUE_BONUS
            elif row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_MEDIUM:
                score += SCORING.PB_VALUE_BONUS
            elif row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_WEAK:
                score += SCORING.PB_FAIR_BONUS
            elif row["PB"] > CLUSTER_THRESHOLDS.SCORE_PB_OVER:
                score += SCORING.PB_OVER_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_quality_score(row: pd.Series) -> float:
        """Расчет скора качества (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_HIGH:
                score += SCORING.ROE_HIGH_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_GOOD:
                score += SCORING.ROE_GOOD_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_MEDIUM:
                score += SCORING.ROE_MEDIUM_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_LOW:
                score += SCORING.ROE_LOW_BONUS
            elif row["ROE"] < 0:
                score += SCORING.ROE_NEGATIVE_PENALTY

        if pd.notna(row.get("Debt_Capital")):
            if row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_LOW:
                score += SCORING.DEBT_LOW_BONUS
            elif row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_MEDIUM:
                score += SCORING.DEBT_MEDIUM_BONUS
            elif row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_HIGH:
                score += SCORING.DEBT_HIGH_BONUS
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_CRITICAL:
                score += SCORING.DEBT_CRITICAL_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_growth_score(row: pd.Series) -> float:
        """Расчет скора роста (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_GOOD:
                score += SCORING.ROE_GOOD_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_MEDIUM:
                score += SCORING.ROE_MEDIUM_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_LOW:
                score += SCORING.ROE_LOW_BONUS

        if pd.notna(row.get("PS")) and row["PS"] > 0:
            if row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_OVER:
                score += SCORING.PS_HIGH_BONUS
            elif row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_WEAK:
                score += SCORING.PS_GOOD_BONUS
            elif row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_MEDIUM:
                score += SCORING.PS_MEDIUM_BONUS

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_income_score(row: pd.Series) -> float:
        """Расчет скора дивидендного дохода (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("Div_Yield")):
            dy = row["Div_Yield"]
            if dy > CLUSTER_THRESHOLDS.SCORE_DIV_HIGH:
                score += SCORING.DIV_HIGH_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_GOOD:
                score += SCORING.DIV_GOOD_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_MEDIUM:
                score += SCORING.DIV_MEDIUM_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_LOW:
                score += SCORING.DIV_LOW_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_POOR:
                score += SCORING.DIV_POOR_BONUS
            elif dy < CLUSTER_THRESHOLDS.SCORE_DIV_MIN:
                score += SCORING.DIV_MIN_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_expected_return(row: pd.Series) -> float:
        """Расчет ожидаемой доходности"""
        base_return = PORTFOLIO_CLUSTER.BASE_EXPECTED_RETURN

        if pd.notna(row.get("PE")) and row["PE"] > 0:
            if row["PE"] < CLUSTER_THRESHOLDS.PE_DEEP_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_DEEP_PREMIUM
            elif row["PE"] < CLUSTER_THRESHOLDS.PE_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_VALUE_PREMIUM
            elif row["PE"] < CLUSTER_THRESHOLDS.PE_FAIR:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_FAIR_PREMIUM

        if pd.notna(row.get("PB")) and row["PB"] > 0:
            if row["PB"] < CLUSTER_THRESHOLDS.PB_DEEP_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_DEEP_PREMIUM
            elif row["PB"] < CLUSTER_THRESHOLDS.PB_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_VALUE_PREMIUM
            elif row["PB"] < CLUSTER_THRESHOLDS.PB_FAIR:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_FAIR_PREMIUM

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.ROE_HIGH:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_HIGH_PREMIUM
            elif row["ROE"] > CLUSTER_THRESHOLDS.ROE_GOOD:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_GOOD_PREMIUM
            elif row["ROE"] > CLUSTER_THRESHOLDS.ROE_GOOD * 0.75:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_MEDIUM_PREMIUM

        if pd.notna(row.get("Div_Yield")):
            base_return += (
                row["Div_Yield"] / 100
            ) * RETURN_PREMIUMS_CLUSTER.DIVIDEND_PREMIUM_FACTOR

        return min(RETURN_PREMIUMS_CLUSTER.MAX_RETURN, base_return)

    @staticmethod
    def calculate_risk(row: pd.Series) -> float:
        """Расчет риска"""
        base_risk = PORTFOLIO_CLUSTER.BASE_RISK

        if pd.notna(row.get("Beta")):
            base_risk += (row["Beta"] - 1) * RISK_PREMIUMS_CLUSTER.BETA_RISK_FACTOR

        if pd.notna(row.get("Debt_Capital")):
            if row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_CRITICAL:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_CRITICAL_PENALTY
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_HIGH:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_HIGH_PENALTY
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_MEDIUM:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_MEDIUM_PENALTY

        if pd.notna(row.get("PE")):
            if row["PE"] < 0 or row["PE"] > CLUSTER_THRESHOLDS.PE_GROWTH * 3:
                base_risk += RISK_PREMIUMS_CLUSTER.PE_EXTREME_PENALTY
            elif pd.isna(row["PE"]):
                base_risk += RISK_PREMIUMS_CLUSTER.PE_MISSING_PENALTY

        return max(
            PORTFOLIO_CLUSTER.MIN_RISK, min(PORTFOLIO_CLUSTER.MAX_RISK, base_risk)
        )
