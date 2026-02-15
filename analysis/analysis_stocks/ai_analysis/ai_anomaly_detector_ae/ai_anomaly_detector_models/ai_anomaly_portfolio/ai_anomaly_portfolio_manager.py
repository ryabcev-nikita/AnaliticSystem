# ==================== КЛАСС УПРАВЛЕНИЯ ПОРТФЕЛЕМ ====================


from typing import Dict

import numpy as np
import pandas as pd
from ...ai_anomaly_detector_models.ai_anomaly_analyzer.ai_anomaly_analyzer import (
    AEPortfolioOptimizer,
)
from ...ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_FORMAT,
    AE_REC,
)


class AEPortfolioManager:
    """Управление портфелем на основе результатов автоэнкодера"""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        weights: np.ndarray,
        optimizer: AEPortfolioOptimizer,
    ):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.optimizer = optimizer

        expected_returns = self.df["AE_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        self.metrics = optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or AE_FORMAT.TOP_POSITIONS_SUMMARY
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()

    def get_risk_contribution(self) -> pd.Series:
        """Вклад в риск портфеля"""
        if len(self.df) == 0:
            return pd.Series()

        weights = self.weights
        risks = self.df["AE_Volatility"].values
        risk_contribution = weights * risks / np.sum(weights * risks)

        tickers = self.df.get("Тикер", self.df.index)
        return pd.Series(risk_contribution, index=tickers)

    def get_anomaly_allocation(self) -> Dict:
        """Распределение по статусу аномалий"""
        anomaly_mask = self.df["AE_Аномалия"] == True
        top_mask = self.df["AE_Топ_недооцененные"] == True

        return {
            AE_REC.CATEGORY_ANOMALIES: (
                self.weights[anomaly_mask].sum() if anomaly_mask.any() else 0
            ),
            AE_REC.CATEGORY_TOP_UNDERVALUED: (
                self.weights[top_mask].sum() if top_mask.any() else 0
            ),
            AE_REC.CATEGORY_NORMAL: self.weights[~(anomaly_mask | top_mask)].sum(),
        }
