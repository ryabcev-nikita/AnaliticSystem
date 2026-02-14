# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================
from dataclasses import dataclass
from typing import Dict
import pandas as pd
from scipy.optimize import minimize
import numpy as np

from ai_risk_models.ai_risk_constants.ai_risk_constants import (
    NN_PORTFOLIO,
    RISK_CAT,
)


@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""

    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float
    var_95: float
    cvar_95: float


class NNRiskPortfolioOptimizer:
    """Оптимизация портфеля на основе нейросетевой оценки риска"""

    def __init__(
        self,
        min_weight: float = None,
        max_weight: float = None,
        risk_free_rate: float = None,
    ):
        self.min_weight = min_weight or NN_PORTFOLIO.MIN_WEIGHT
        self.max_weight = max_weight or NN_PORTFOLIO.MAX_WEIGHT
        self.risk_free_rate = risk_free_rate or NN_PORTFOLIO.RISK_FREE_RATE

    def calculate_expected_return(self, row: pd.Series) -> float:
        """Расчет ожидаемой доходности с учетом нейросетевой оценки риска"""
        base_return = NN_PORTFOLIO.BASE_RETURN

        risk_category = row.get("NN_Категория_текст", "")
        risk_premium_map = {
            RISK_CAT.RISK_A_NAME: NN_PORTFOLIO.RISK_A_PREMIUM,
            RISK_CAT.RISK_B_NAME: NN_PORTFOLIO.RISK_B_PREMIUM,
            RISK_CAT.RISK_C_NAME: NN_PORTFOLIO.RISK_C_PREMIUM,
            RISK_CAT.RISK_D_NAME: NN_PORTFOLIO.RISK_D_PREMIUM,
        }
        base_return += risk_premium_map.get(risk_category, 0)

        if pd.notna(row.get("ROE")):
            if row["ROE"] > NN_PORTFOLIO.ROE_HIGH_THRESHOLD:
                base_return += NN_PORTFOLIO.ROE_HIGH_PREMIUM
            elif row["ROE"] > NN_PORTFOLIO.ROE_MEDIUM_THRESHOLD:
                base_return += NN_PORTFOLIO.ROE_MEDIUM_PREMIUM

        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row.get("Дивидендная доходность", 0)
            base_return += div_yield / 100 * NN_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR

        confidence = row.get("NN_Уверенность", 0.5)
        base_return *= (
            NN_PORTFOLIO.CONFIDENCE_SCALE
            + NN_PORTFOLIO.CONFIDENCE_MAX_FACTOR * confidence
        )

        return min(NN_PORTFOLIO.MAX_RETURN, max(NN_PORTFOLIO.MIN_RETURN, base_return))

    def calculate_volatility(self, row: pd.Series) -> float:
        """Расчет волатильности на основе нейросетевой оценки"""
        risk_category = row.get("NN_Категория_текст", "")

        volatility_map = {
            RISK_CAT.RISK_A_NAME: NN_PORTFOLIO.VOLATILITY_A,
            RISK_CAT.RISK_B_NAME: NN_PORTFOLIO.VOLATILITY_B,
            RISK_CAT.RISK_C_NAME: NN_PORTFOLIO.VOLATILITY_C,
            RISK_CAT.RISK_D_NAME: NN_PORTFOLIO.VOLATILITY_D,
        }

        base_vol = volatility_map.get(risk_category, NN_PORTFOLIO.BASE_VOLATILITY)

        if pd.notna(row.get("Бета")):
            beta = row.get("Бета", 1)
            base_vol *= (
                NN_PORTFOLIO.BETA_VOL_FACTOR_MIN
                + NN_PORTFOLIO.BETA_VOL_FACTOR_MAX * beta
            )

        if pd.notna(row.get("Debt/Capital")):
            debt = row.get("Debt/Capital", 0)
            base_vol *= (
                NN_PORTFOLIO.DEBT_VOL_FACTOR_MIN
                + NN_PORTFOLIO.DEBT_VOL_FACTOR_MAX
                * (debt / NN_PORTFOLIO.DEBT_NORMALIZATION)
            )

        return min(
            NN_PORTFOLIO.MAX_VOLATILITY, max(NN_PORTFOLIO.MIN_VOLATILITY, base_vol)
        )

    def create_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Создание матрицы ковариации с учетом корреляций"""
        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["NN_Volatility"].values
        risk_categories = df.get("NN_Категория_текст", ["Unknown"] * n).values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    corr = (
                        NN_PORTFOLIO.INTRA_CATEGORY_CORRELATION
                        if risk_categories[i] == risk_categories[j]
                        else NN_PORTFOLIO.INTER_CATEGORY_CORRELATION
                    )
                    cov_matrix[i, j] = corr * risks[i] * risks[j]

        return cov_matrix

    def calculate_portfolio_metrics(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        port_return = np.sum(expected_returns * weights)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

        var_95 = port_return - NN_PORTFOLIO.VAR_95_COEFF * port_risk
        cvar_95 = port_return - NN_PORTFOLIO.CVAR_95_COEFF * port_risk

        hhi = np.sum(weights**2)
        n = len(weights)
        diversification = 1 - (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 0

        return PortfolioMetrics(
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def optimize_portfolio(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> Dict:
        """Оптимизация портфеля с несколькими целями"""
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return (
                -(port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
            )

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def negative_return(weights):
            return -np.sum(expected_returns * weights)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        init_guess = np.array([1 / n] * n)

        result_sharpe = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_min_risk = minimize(
            portfolio_risk,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_max_return = minimize(
            negative_return,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        combined_weights = (
            NN_PORTFOLIO.SHARPE_PORTFOLIO_WEIGHT * result_sharpe.x
            + NN_PORTFOLIO.MIN_RISK_PORTFOLIO_WEIGHT * result_min_risk.x
            + NN_PORTFOLIO.MAX_RETURN_PORTFOLIO_WEIGHT * result_max_return.x
        )
        combined_weights = combined_weights / combined_weights.sum()

        return {
            "sharpe_weights": result_sharpe.x,
            "min_risk_weights": result_min_risk.x,
            "max_return_weights": result_max_return.x,
            "combined_weights": combined_weights,
            "cov_matrix": cov_matrix,
        }

    def optimize_risk_based_portfolios(self, df: pd.DataFrame) -> Dict:
        """Создание портфелей на основе категорий риска"""
        portfolios = {}

        conservative = df[
            df["NN_Категория_текст"].isin([RISK_CAT.RISK_A_NAME, RISK_CAT.RISK_B_NAME])
        ].copy()
        if len(conservative) > 0:
            weights = self._equal_weight_by_category(conservative)
            portfolios["Консервативный"] = (conservative, weights)

        balanced = df.copy()
        if len(balanced) > 0:
            weights = self._risk_weighted_allocation(balanced)
            portfolios["Сбалансированный"] = (balanced, weights)

        aggressive = df[
            df["NN_Категория_текст"].isin([RISK_CAT.RISK_C_NAME, RISK_CAT.RISK_D_NAME])
        ].copy()
        if len(aggressive) > 0:
            weights = self._return_weighted_allocation(aggressive)
            portfolios["Агрессивный"] = (aggressive, weights)

        dividend = df[
            df["Дивидендная доходность"].fillna(0)
            > NN_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR * 10
        ].copy()
        if len(dividend) > 0:
            weights = self._dividend_weighted_allocation(dividend)
            portfolios["Дивидендный"] = (dividend, weights)

        confidence = df[df["NN_Уверенность"] > NN_PORTFOLIO.CONFIDENCE_SCALE].copy()
        if len(confidence) > 0:
            weights = self._confidence_weighted_allocation(confidence)
            portfolios["Нейросетевой"] = (confidence, weights)

        return portfolios

    def _equal_weight_by_category(self, df: pd.DataFrame) -> np.ndarray:
        """Равномерное распределение по категориям риска"""
        n = len(df)
        weights = np.ones(n) / n
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _risk_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе обратного риска"""
        risks = 1 / df["NN_Volatility"].values
        weights = risks / risks.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _return_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе ожидаемой доходности"""
        returns = df["NN_Expected_Return"].values
        weights = returns / returns.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _dividend_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе дивидендной доходности"""
        div_yield = df["Дивидендная доходность"].fillna(0).values
        weights = div_yield / div_yield.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _confidence_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе уверенности нейросети"""
        confidence = df["NN_Уверенность"].values
        weights = confidence / confidence.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()
