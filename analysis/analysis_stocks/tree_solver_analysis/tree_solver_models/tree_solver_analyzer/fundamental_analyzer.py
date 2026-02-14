# ==================== КЛАСС ФУНДАМЕНТАЛЬНОГО АНАЛИЗА ====================
import pandas as pd
from tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FINANCIAL,
    RETURN_PREMIUMS,
    RISK_PREMIUMS,
    TARGET_MAPPING,
)
from tree_solver_models.tree_solver_market.market_benchmarks import MarketBenchmarks


class FundamentalAnalyzer:
    """Фундаментальный анализ и расчет доходности/риска"""

    def __init__(self, benchmarks: MarketBenchmarks):
        self.benchmarks = benchmarks

    def calculate_expected_return(self, row: pd.Series) -> float:
        """Расчет ожидаемой доходности на основе фундаментальных показателей"""
        base_return = FINANCIAL.BASE_RETURN
        score = 0.0

        # P/E премия
        if pd.notna(row.get("P/E")):
            pe = row["P/E"]
            if (
                pe
                < self.benchmarks.pe_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and pe > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PE_PREMIUM
            elif (
                pe < self.benchmarks.pe_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and pe > 0
            ):
                score += RETURN_PREMIUMS.PE_PREMIUM

        # P/S премия
        if pd.notna(row.get("P/S")):
            ps = row["P/S"]
            if (
                ps
                < self.benchmarks.ps_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and ps > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PS_PREMIUM
            elif (
                ps < self.benchmarks.ps_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and ps > 0
            ):
                score += RETURN_PREMIUMS.PS_PREMIUM

        # P/B премия
        if pd.notna(row.get("P/B")) and pd.notna(row.get("ROE")):
            pb = row["P/B"]
            if (
                pb < self.benchmarks.pb_median * FINANCIAL.PB_STRONG_THRESHOLD
                and row["ROE"] > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PB_PREMIUM
            elif (
                pb < self.benchmarks.pb_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and row["ROE"] > 0
            ):
                score += RETURN_PREMIUMS.PB_PREMIUM

        # ROE премия
        if pd.notna(row.get("ROE")):
            roe = row["ROE"]
            if (
                roe > self.benchmarks.roe_median * FINANCIAL.ROE_STRONG_THRESHOLD
                and roe > 0
            ):
                score += RETURN_PREMIUMS.STRONG_ROE_PREMIUM
            elif (
                roe > self.benchmarks.roe_median * FINANCIAL.ROE_GOOD_THRESHOLD
                and roe > 0
            ):
                score += RETURN_PREMIUMS.ROE_PREMIUM

        # Дивидендная премия
        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row["Дивидендная доходность"]
            if (
                div_yield
                > self.benchmarks.div_yield_median * FINANCIAL.DIVIDEND_STRONG_THRESHOLD
            ):
                score += RETURN_PREMIUMS.STRONG_DIVIDEND_PREMIUM
            elif (
                div_yield
                > self.benchmarks.div_yield_median * FINANCIAL.DIVIDEND_GOOD_THRESHOLD
            ):
                score += RETURN_PREMIUMS.DIVIDEND_PREMIUM

        # Бонус за оценку модели
        if pd.notna(row.get("Predicted_Оценка")):
            if row["Predicted_Оценка"] == TARGET_MAPPING.STRONG_UNDERVALUED:
                score += RETURN_PREMIUMS.MODEL_STRONG_PREMIUM
            elif row["Predicted_Оценка"] == TARGET_MAPPING.UNDERVALUED:
                score += RETURN_PREMIUMS.MODEL_PREMIUM

        return base_return + score

    def calculate_risk(self, row: pd.Series) -> float:
        """Расчет риска на основе беты, долга и волатильности"""
        base_risk = RISK_PREMIUMS.BASE_RISK

        # Бета риск
        if pd.notna(row.get("Бета")):
            beta = row["Бета"]
            if beta > self.benchmarks.beta_median * FINANCIAL.BETA_HIGH_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_HIGH_PENALTY
            elif beta > self.benchmarks.beta_median * FINANCIAL.UNDERVALUED_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_MEDIUM_PENALTY
            elif beta < self.benchmarks.beta_median * FINANCIAL.BETA_LOW_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_LOW_BONUS

        # Долговой риск
        if pd.notna(row.get("Debt/Capital")):
            debt = row["Debt/Capital"]
            if (
                debt
                > self.benchmarks.debt_capital_median * FINANCIAL.DEBT_HIGH_THRESHOLD
            ):
                base_risk += RISK_PREMIUMS.DEBT_HIGH_PENALTY
            elif (
                debt
                > self.benchmarks.debt_capital_median * FINANCIAL.UNDERVALUED_THRESHOLD
            ):
                base_risk += RISK_PREMIUMS.DEBT_MEDIUM_PENALTY

        # Штраф за переоцененность
        if pd.notna(row.get("Predicted_Оценка")):
            if row["Predicted_Оценка"] == TARGET_MAPPING.OVERVALUED:
                base_risk += RISK_PREMIUMS.OVERVALUED_PENALTY

        return max(RISK_PREMIUMS.MIN_RISK, min(RISK_PREMIUMS.MAX_RISK, base_risk))
