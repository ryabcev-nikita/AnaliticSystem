# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================
from typing import Dict
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from tree_solver_models.tree_solver_constants.tree_solver_constants import (
    PORTFOLIO_CONSTANTS,
)


class PortfolioOptimizerTree:
    """Оптимизация портфеля по Марковицу"""

    def __init__(self, min_weight: float = None, max_weight: float = None):
        self.min_weight = min_weight or PORTFOLIO_CONSTANTS.MIN_WEIGHT
        self.max_weight = max_weight or PORTFOLIO_CONSTANTS.MAX_WEIGHT

    def create_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Создание матрицы ковариации"""
        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["Риск"].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    correlation = (
                        PORTFOLIO_CONSTANTS.INTRASECTOR_CORRELATION
                        if df.iloc[i]["Сектор"] == df.iloc[j]["Сектор"]
                        else PORTFOLIO_CONSTANTS.INTERSECTOR_CORRELATION
                    )
                    cov_matrix[i, j] = correlation * risks[i] * risks[j]

        return cov_matrix

    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict:
        """Оптимизация портфеля"""
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_risk if port_risk > 0 else -np.inf

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        init_guess = [1 / n] * n

        # Максимизация Шарпа
        result_sharpe = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Минимизация риска
        result_min_risk = minimize(
            portfolio_risk,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Комбинированный портфель
        combined_weights = (
            PORTFOLIO_CONSTANTS.SHARPE_PORTFOLIO_WEIGHT * result_sharpe.x
            + PORTFOLIO_CONSTANTS.MIN_RISK_PORTFOLIO_WEIGHT * result_min_risk.x
        )
        combined_weights = combined_weights / combined_weights.sum()

        return {
            "sharpe_weights": result_sharpe.x,
            "min_risk_weights": result_min_risk.x,
            "combined_weights": combined_weights,
            "cov_matrix": cov_matrix,
        }
