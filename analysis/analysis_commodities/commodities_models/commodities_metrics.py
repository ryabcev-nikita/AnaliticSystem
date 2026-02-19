from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from commodities_models.commodities_constants import (
    BENCHMARK_INDEX,
    ENERGY,
    INDUSTRIAL_METALS,
    PRECIOUS_METALS,
)


@dataclass
class MetalsPortfolioMetrics:
    """Класс для хранения метрик портфеля металлов"""

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
    precious_metals_weight: Optional[float] = None
    industrial_metals_weight: Optional[float] = None
    energy_weight: Optional[float] = None

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
            "Precious Metals Weight": self.precious_metals_weight,
            "Industrial Metals Weight": self.industrial_metals_weight,
            "Energy Weight": self.energy_weight,
        }


class MetalsRiskMetricsCalculator:
    """Класс для расчета метрик риска для металлов"""

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
        returns: pd.DataFrame, weights: np.ndarray, benchmark: str = BENCHMARK_INDEX
    ) -> float:
        """Расчет бета коэффициента портфеля относительно бенчмарка"""
        portfolio_returns = returns.dot(weights)
        benchmark_returns = returns[benchmark]

        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance != 0 else 0

        return beta

    @staticmethod
    def calculate_hhi(weights: np.ndarray) -> float:
        """Расчет индекса Херфиндаля-Хиршмана (концентрация)"""
        return np.sum(weights**2)

    @staticmethod
    def calculate_group_weights(weights_series: pd.Series) -> Dict:
        """Расчет весов по группам металлов"""
        precious_weight = weights_series[
            [idx for idx in weights_series.index if idx in PRECIOUS_METALS]
        ].sum()
        industrial_weight = weights_series[
            [idx for idx in weights_series.index if idx in INDUSTRIAL_METALS]
        ].sum()
        energy_weight = weights_series[
            [idx for idx in weights_series.index if idx in ENERGY]
        ].sum()

        return {
            "precious": precious_weight,
            "industrial": industrial_weight,
            "energy": energy_weight,
        }
