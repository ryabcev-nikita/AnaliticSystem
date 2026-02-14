# ==================== КЛАСС АНАЛИЗАТОРА РЫНКА ===================
from typing import Dict
import pandas as pd
from tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FINANCIAL,
    SECTOR_KEYWORDS,
    SECTOR_NAMES,
    TARGET_MAPPING,
    VALUATION_SCORES,
)
from tree_solver_models.tree_solver_market.market_benchmarks import MarketBenchmarks


class MarketAnalyzer:
    """Анализ рыночных мультипликаторов и бенчмарков"""

    @staticmethod
    def calculate_benchmarks(df: pd.DataFrame) -> MarketBenchmarks:
        """Расчет медианных значений мультипликаторов"""
        return MarketBenchmarks(
            pe_median=df["P/E"].median(),
            pb_median=df["P/B"].median(),
            ps_median=df["P/S"].median(),
            roe_median=df["ROE"].median(),
            div_yield_median=df["Averange_dividend_yield"].median(),
            debt_capital_median=df["Debt/Capital"].median(),
            beta_median=df["Бета"].median(),
        )

    @staticmethod
    def assign_sector(name: str) -> str:
        """Определение сектора компании"""
        name = str(name).lower()

        sector_mappings = [
            (SECTOR_KEYWORDS.BANKS, SECTOR_NAMES.BANKS),
            (SECTOR_KEYWORDS.OIL_GAS, SECTOR_NAMES.OIL_GAS),
            (SECTOR_KEYWORDS.METALS, SECTOR_NAMES.METALS),
            (SECTOR_KEYWORDS.ENERGY, SECTOR_NAMES.ENERGY),
            (SECTOR_KEYWORDS.TELECOM, SECTOR_NAMES.TELECOM),
            (SECTOR_KEYWORDS.RETAIL, SECTOR_NAMES.RETAIL),
            (SECTOR_KEYWORDS.CHEMICAL, SECTOR_NAMES.CHEMICAL),
            (SECTOR_KEYWORDS.IT, SECTOR_NAMES.IT),
        ]

        for keywords, sector_name in sector_mappings:
            if any(word in name for word in keywords):
                return sector_name

        return SECTOR_NAMES.OTHER

    @staticmethod
    def calculate_relative_valuation(
        row: pd.Series, benchmarks: MarketBenchmarks
    ) -> Dict:
        """Оценка относительной стоимости на основе медиан"""
        scores = {}

        # P/E оценка
        if pd.notna(row.get("P/E")):
            pe_ratio = row["P/E"]
            if (
                pe_ratio
                < benchmarks.pe_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and pe_ratio > 0
            ):
                scores["pe_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                pe_ratio < benchmarks.pe_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and pe_ratio > 0
            ):
                scores["pe_score"] = VALUATION_SCORES.BUY
            elif (
                pe_ratio > benchmarks.pe_median * FINANCIAL.OVERVALUED_THRESHOLD
                or pe_ratio < 0
            ):
                scores["pe_score"] = VALUATION_SCORES.SELL
            else:
                scores["pe_score"] = VALUATION_SCORES.HOLD
        else:
            scores["pe_score"] = VALUATION_SCORES.HOLD

        # P/S оценка
        if pd.notna(row.get("P/S")):
            ps_ratio = row["P/S"]
            if (
                ps_ratio
                < benchmarks.ps_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
            ):
                scores["ps_score"] = VALUATION_SCORES.STRONG_BUY
            elif ps_ratio < benchmarks.ps_median * FINANCIAL.UNDERVALUED_THRESHOLD:
                scores["ps_score"] = VALUATION_SCORES.BUY
            elif ps_ratio > benchmarks.ps_median * FINANCIAL.OVERVALUED_THRESHOLD:
                scores["ps_score"] = VALUATION_SCORES.SELL
            else:
                scores["ps_score"] = VALUATION_SCORES.HOLD
        else:
            scores["ps_score"] = VALUATION_SCORES.HOLD

        # P/B оценка
        if pd.notna(row.get("P/B")) and pd.notna(row.get("ROE")):
            pb_ratio = row["P/B"]
            if (
                pb_ratio < benchmarks.pb_median * FINANCIAL.PB_STRONG_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                pb_ratio < benchmarks.pb_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.BUY
            elif (
                pb_ratio > benchmarks.pb_median * FINANCIAL.PB_OVERVAULED_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.SELL
            else:
                scores["pb_score"] = VALUATION_SCORES.HOLD
        else:
            scores["pb_score"] = VALUATION_SCORES.HOLD

        # ROE оценка
        if pd.notna(row.get("ROE")):
            roe = row["ROE"]
            if roe > benchmarks.roe_median * FINANCIAL.ROE_STRONG_THRESHOLD and roe > 0:
                scores["roe_score"] = VALUATION_SCORES.STRONG_BUY
            elif roe > benchmarks.roe_median * FINANCIAL.ROE_GOOD_THRESHOLD and roe > 0:
                scores["roe_score"] = VALUATION_SCORES.BUY
            else:
                scores["roe_score"] = VALUATION_SCORES.HOLD
        else:
            scores["roe_score"] = VALUATION_SCORES.HOLD

        # Дивидендная оценка
        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row["Дивидендная доходность"]
            if (
                div_yield
                > benchmarks.div_yield_median * FINANCIAL.DIVIDEND_STRONG_THRESHOLD
            ):
                scores["div_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                div_yield
                > benchmarks.div_yield_median * FINANCIAL.DIVIDEND_GOOD_THRESHOLD
            ):
                scores["div_score"] = VALUATION_SCORES.BUY
            else:
                scores["div_score"] = VALUATION_SCORES.HOLD
        else:
            scores["div_score"] = VALUATION_SCORES.HOLD

        # Итоговая оценка
        total_score = sum(scores.values())
        scores["total_score"] = total_score

        if total_score >= VALUATION_SCORES.STRONG_BUY_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[
                TARGET_MAPPING.STRONG_UNDERVALUED
            ]
            scores["valuation_code"] = TARGET_MAPPING.STRONG_UNDERVALUED
        elif total_score >= VALUATION_SCORES.BUY_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.UNDERVALUED]
            scores["valuation_code"] = TARGET_MAPPING.UNDERVALUED
        elif total_score <= VALUATION_SCORES.SELL_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.OVERVALUED]
            scores["valuation_code"] = TARGET_MAPPING.OVERVALUED
        else:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.FAIR_VALUE]
            scores["valuation_code"] = TARGET_MAPPING.FAIR_VALUE

        return scores
