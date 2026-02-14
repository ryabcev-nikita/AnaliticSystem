# ==================== КЛАСС ПОРТФЕЛЬНОГО МЕНЕДЖЕРА ====================
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tree_solver_models.tree_solver_portfolio.portfolio_optimizer_tree import (
    PortfolioOptimizerTree,
)
from tree_solver_models.tree_solver_constants.tree_solver_constants import (
    PORTFOLIO_CONSTANTS,
)


@dataclass
class PortfolioMetrics:
    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float


class PortfolioManager:
    """Управление портфелем и расчет метрик"""

    def __init__(self, df: pd.DataFrame, weights: np.ndarray):
        self.df = df.copy()
        self.weights = weights
        self.df["weights"] = weights
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        exp_return = np.sum(self.df["Ожидаемая_доходность"] * self.weights)

        optimizer = PortfolioOptimizerTree()
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        sharpe = exp_return / risk if risk > 0 else 0

        # Индекс диверсификации Херфиндаля-Хиршмана
        hhi = np.sum(self.weights**2)
        n = len(self.weights)
        if n > 1:
            diversification = 1 - (hhi - 1 / n) / (1 - 1 / n)
        else:
            diversification = 0

        return PortfolioMetrics(
            expected_return=exp_return,
            risk=risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        return self.df.groupby("Сектор")["weights"].sum()

    def get_top_positions(
        self, n: int = PORTFOLIO_CONSTANTS.TOP_POSITIONS_N
    ) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()
