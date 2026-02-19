from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from commodities_models.commodities_constants import (
    EF_POINTS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    NUM_PORTFOLIOS,
    RISK_FREE_RATE,
    TRADING_DAYS,
)


class MetalsPortfolioOptimizer:
    """Класс для оптимизации портфеля металлов по Марковицу"""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean().values * TRADING_DAYS

    def portfolio_statistics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Расчет доходности, волатильности и Шарпа для портфеля"""
        portfolio_return = np.sum(self.returns.mean() * weights) * TRADING_DAYS
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * TRADING_DAYS, weights))
        )
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def optimize_max_sharpe(self, target_return: Optional[float] = None) -> np.ndarray:
        """Оптимизация для максимального коэффициента Шарпа"""
        init_weights = np.array([1 / self.n_assets] * self.n_assets)
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(self.n_assets))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if target_return is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(self.returns.mean() * x) * TRADING_DAYS
                    - target_return,
                }
            )

        def neg_sharpe(weights):
            _, _, sharpe = self.portfolio_statistics(weights)
            return -sharpe

        result = minimize(
            neg_sharpe,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        return result.x

    def optimize_min_volatility(
        self, target_return: Optional[float] = None
    ) -> np.ndarray:
        """Оптимизация для минимальной волатильности"""
        init_weights = np.array([1 / self.n_assets] * self.n_assets)
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(self.n_assets))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if target_return is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(self.returns.mean() * x) * TRADING_DAYS
                    - target_return,
                }
            )

        def min_volatility(weights):
            return np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * TRADING_DAYS, weights))
            )

        result = minimize(
            min_volatility,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        return result.x

    def simulate_portfolios(
        self, num_portfolios: int = NUM_PORTFOLIOS
    ) -> Tuple[np.ndarray, List]:
        """Симуляция случайных портфелей"""
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
            weights_record.append(weights)

            ret, vol, sharpe = self.portfolio_statistics(weights)

            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe

        return results, weights_record

    def efficient_frontier(self, points: int = EF_POINTS) -> pd.DataFrame:
        """Построение эффективной границы"""
        min_ret = self.returns.mean().min() * TRADING_DAYS
        max_ret = self.returns.mean().max() * TRADING_DAYS

        # Добавляем небольшой запас
        min_ret = max(min_ret, -0.1)  # Минимум -10%
        max_ret = min(max_ret, 0.5)  # Максимум 50%

        target_returns = np.linspace(min_ret, max_ret, points)
        efficient_portfolios = []

        for target in target_returns:
            try:
                weights = self.optimize_min_volatility(target_return=target)
                _, vol, _ = self.portfolio_statistics(weights)

                efficient_portfolios.append(
                    {"return": target, "volatility": vol, "weights": weights}
                )
            except:
                continue

        return pd.DataFrame(efficient_portfolios)
