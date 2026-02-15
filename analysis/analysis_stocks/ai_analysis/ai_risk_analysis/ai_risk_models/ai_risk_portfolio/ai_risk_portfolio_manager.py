# ==================== КЛАСС УПРАВЛЕНИЯ ПОРТФЕЛЕМ ====================


import numpy as np
import pandas as pd

from ...ai_risk_models.ai_risk_constants.ai_risk_constants import NN_FORMAT
from ...ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_optimizer import (
    NNRiskPortfolioOptimizer,
)


class NNRiskPortfolioManager:
    """Управление портфелем на основе нейросетевой оценки риска"""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        weights: np.ndarray,
        optimizer: NNRiskPortfolioOptimizer,
    ):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.optimizer = optimizer

        expected_returns = self.df["NN_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        self.metrics = optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        if "Сектор" in self.df.columns:
            return self.df.groupby("Сектор")["Weight"].sum()
        return pd.Series()

    def get_risk_category_allocation(self) -> pd.Series:
        """Распределение по категориям риска"""
        if "NN_Категория_текст" in self.df.columns:
            return self.df.groupby("NN_Категория_текст")["Weight"].sum()
        return pd.Series()

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or NN_FORMAT.TOP_POSITIONS_SUMMARY
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()

    def get_risk_contribution(self) -> pd.Series:
        """Вклад в риск портфеля"""
        if len(self.df) == 0:
            return pd.Series()

        weights = self.weights
        risks = self.df["NN_Volatility"].values
        risk_contribution = weights * risks / np.sum(weights * risks)

        tickers = self.df.get("Тикер", self.df.index)
        return pd.Series(risk_contribution, index=tickers)
