from typing import Dict

import numpy as np
import pandas as pd

from commodities_models.commodities_constants import RISK_FREE_RATE, TRADING_DAYS


class MetalsStatisticsCalculator:
    """Класс для расчета статистик по металлам"""

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Расчет логарифмических доходностей"""
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    @staticmethod
    def calculate_annual_statistics(returns: pd.DataFrame) -> pd.DataFrame:
        """Расчет годовых статистик"""
        annual_returns = returns.mean() * TRADING_DAYS
        annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe_ratios = (annual_returns - RISK_FREE_RATE) / annual_volatility

        return pd.DataFrame(
            {
                "Annual Return (%)": annual_returns * 100,
                "Annual Volatility (%)": annual_volatility * 100,
                "Sharpe Ratio": sharpe_ratios,
            }
        )

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Расчет максимальной просадки для одного металла"""
        wealth_index = prices / prices.iloc[0]
        previous_peaks = wealth_index.expanding().max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    @staticmethod
    def calculate_all_max_drawdowns(prices: pd.DataFrame) -> Dict[str, float]:
        """Расчет максимальных просадок для всех металлов"""
        return {
            col: MetalsStatisticsCalculator.calculate_max_drawdown(prices[col])
            for col in prices.columns
        }

    @staticmethod
    def calculate_skewness_kurtosis(returns: pd.DataFrame) -> pd.DataFrame:
        """Расчет skewness и kurtosis для доходностей"""
        stats = pd.DataFrame(
            {
                "Skewness": returns.skew(),
                "Kurtosis": returns.kurtosis(),
                "Min Daily Return": returns.min(),
                "Max Daily Return": returns.max(),
            }
        )
        return stats
