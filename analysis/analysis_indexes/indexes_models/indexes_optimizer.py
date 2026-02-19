from typing import Dict, List, Optional, Tuple

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from indexes_models.indexes_constants import (
    EF_POINTS,
    INDEX_NAMES,
    MAX_WEIGHT,
    MIN_WEIGHT,
    NUM_PORTFOLIOS,
    PLOT_STYLE,
    RISK_FREE_RATE,
    TRADING_DAYS,
)


class CorrelationAnalyzer:
    """Класс для корреляционного анализа"""

    def __init__(self):
        self.correlation_matrix = None

    def analyze(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Проведение корреляционного анализа"""
        self.correlation_matrix = returns.corr()
        return self.correlation_matrix

    def get_top_correlations(self, n: int = 5) -> List[Dict]:
        """Получение наиболее коррелированных пар"""
        corr_pairs = []
        matrix = self.correlation_matrix

        for i in range(len(matrix.columns)):
            for j in range(i + 1, len(matrix.columns)):
                corr_pairs.append(
                    {
                        "index1": matrix.columns[i],
                        "index2": matrix.columns[j],
                        "correlation": matrix.iloc[i, j],
                    }
                )

        return sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:n]

    def plot(
        self, returns: pd.DataFrame, save_path: str, show_plot: bool = True
    ) -> None:
        """Визуализация корреляционной матрицы"""
        if self.correlation_matrix is None:
            self.analyze(returns)

        plt.figure(figsize=PLOT_STYLE["figure_size"]["large"])
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            xticklabels=[INDEX_NAMES[x] for x in self.correlation_matrix.columns],
            yticklabels=[INDEX_NAMES[x] for x in self.correlation_matrix.columns],
        )

        plt.title(
            "Correlation Matrix of Global Market Indices",
            fontsize=PLOT_STYLE["title_fontsize"],
            pad=20,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()


class PortfolioOptimizer:
    """Класс для оптимизации портфеля по Марковицу"""

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

        target_returns = np.linspace(min_ret, max_ret, points)
        efficient_portfolios = []

        for target in target_returns:
            weights = self.optimize_min_volatility(target_return=target)
            _, vol, _ = self.portfolio_statistics(weights)

            efficient_portfolios.append(
                {"return": target, "volatility": vol, "weights": weights}
            )

        return pd.DataFrame(efficient_portfolios)
