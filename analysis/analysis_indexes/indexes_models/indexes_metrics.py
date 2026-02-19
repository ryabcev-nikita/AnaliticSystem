from typing import Tuple

import numpy as np
import pandas as pd

from indexes_models.indexes_constants import BENCHMARK_INDEX
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


class RiskMetricsCalculator:
    """Класс для расчета метрик риска"""

    @staticmethod
    def calculate_var_cvar(
        returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Расчет VaR и CVaR для портфеля"""
        portfolio_returns = returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return var, cvar

    @staticmethod
    def calculate_beta(
        returns: pd.DataFrame, weights: np.ndarray, market_index: str = BENCHMARK_INDEX
    ) -> float:
        """Расчет бета коэффициента портфеля относительно рынка"""
        portfolio_returns = returns.dot(weights)
        market_returns = returns[market_index]

        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        variance = np.var(market_returns)
        return covariance / variance

    @staticmethod
    def calculate_hhi(weights: np.ndarray) -> float:
        """Расчет индекса Херфиндаля-Хиршмана (концентрация)"""
        return np.sum(weights**2)


@dataclass
class PortfolioMetrics:
    """Класс для хранения метрик портфеля"""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_99: Optional[float] = None
    hhi: Optional[float] = None
    n_assets: Optional[int] = None

    def to_dict(self) -> Dict:
        """Преобразование в словарь для Excel"""
        return {
            "Expected Return": self.expected_return,
            "Volatility": self.volatility,
            "Sharpe Ratio": self.sharpe_ratio,
            "Beta": self.beta,
            "VaR 95%": self.var_95,
            "CVaR 95%": self.cvar_95,
            "VaR 99%": self.var_99,
            "CVaR 99%": self.cvar_99,
            "HHI": self.hhi,
            "Number of Assets": self.n_assets,
        }
