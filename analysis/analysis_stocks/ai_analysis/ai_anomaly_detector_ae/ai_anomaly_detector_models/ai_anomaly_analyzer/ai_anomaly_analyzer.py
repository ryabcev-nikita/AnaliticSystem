# ==================== КЛАСС АВТОЭНКОДЕРА ====================

from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch.nn as nn
from scipy.optimize import minimize
from typing import Dict

from ...ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_ARCH,
    AE_PORTFOLIO,
    AE_THRESHOLD,
)


# ==================== КЛАССЫ ДАННЫХ ====================
@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""

    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float
    var_95: float
    cvar_95: float


class AnomalyDetectorAE(nn.Module):
    """Автоэнкодер для обнаружения аномалий в мультипликаторах"""

    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, AE_ARCH.ENCODER_LAYER_1),
            nn.ReLU(),
            nn.Linear(AE_ARCH.ENCODER_LAYER_1, AE_ARCH.ENCODER_LAYER_2),
            nn.ReLU(),
            nn.Linear(AE_ARCH.ENCODER_LAYER_2, AE_ARCH.ENCODER_LAYER_3),
            nn.ReLU(),
            nn.Linear(AE_ARCH.ENCODER_LAYER_3, AE_ARCH.ENCODER_LAYER_4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(AE_ARCH.ENCODER_LAYER_4, AE_ARCH.DECODER_LAYER_1),
            nn.ReLU(),
            nn.Linear(AE_ARCH.DECODER_LAYER_1, AE_ARCH.DECODER_LAYER_2),
            nn.ReLU(),
            nn.Linear(AE_ARCH.DECODER_LAYER_2, AE_ARCH.DECODER_LAYER_3),
            nn.ReLU(),
            nn.Linear(AE_ARCH.DECODER_LAYER_3, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================


class AEPortfolioOptimizer:
    """Оптимизация портфеля на основе результатов автоэнкодера"""

    def __init__(
        self,
        min_weight: float = None,
        max_weight: float = None,
        risk_free_rate: float = None,
    ):
        self.min_weight = min_weight or AE_PORTFOLIO.MIN_WEIGHT
        self.max_weight = max_weight or AE_PORTFOLIO.MAX_WEIGHT
        self.risk_free_rate = risk_free_rate or AE_PORTFOLIO.RISK_FREE_RATE

    def calculate_expected_return(self, row: pd.Series) -> float:
        """Расчет ожидаемой доходности на основе результатов автоэнкодера"""
        base_return = AE_PORTFOLIO.BASE_RETURN

        undervalued_score = row.get("AE_Недооцененность", 0.5)
        base_return += undervalued_score * AE_PORTFOLIO.UNDERVALUED_SCORE_PREMIUM

        if row.get("AE_Топ_недооцененные", False):
            base_return += AE_PORTFOLIO.TOP_UNDERVALUED_PREMIUM

        if row.get("AE_Аномалия", False):
            base_return += AE_PORTFOLIO.ANOMALY_PENALTY
        if row.get("AE_Сильная_аномалия", False):
            base_return += AE_PORTFOLIO.STRONG_ANOMALY_PENALTY

        if pd.notna(row.get("P_E")) and row["P_E"] > 0:
            if row["P_E"] < AE_THRESHOLD.PE_STRONG_UNDERVALUED:
                base_return += AE_PORTFOLIO.PE_STRONG_PREMIUM
            elif row["P_E"] < AE_THRESHOLD.PE_UNDERVALUED:
                base_return += AE_PORTFOLIO.PE_MEDIUM_PREMIUM
            elif row["P_E"] > AE_THRESHOLD.PE_OVERVALUED:
                base_return += AE_PORTFOLIO.PE_OVER_PENALTY

        if pd.notna(row.get("P_B")) and row["P_B"] > 0:
            if row["P_B"] < AE_THRESHOLD.PB_STRONG_UNDERVALUED:
                base_return += AE_PORTFOLIO.PB_STRONG_PREMIUM
            elif row["P_B"] < AE_THRESHOLD.PB_UNDERVALUED:
                base_return += AE_PORTFOLIO.PB_MEDIUM_PREMIUM

        if pd.notna(row.get("P_S")) and row["P_S"] > 0:
            if row["P_S"] < AE_THRESHOLD.PS_STRONG_UNDERVALUED:
                base_return += AE_PORTFOLIO.PS_STRONG_PREMIUM
            elif row["P_S"] < AE_THRESHOLD.PS_UNDERVALUED:
                base_return += AE_PORTFOLIO.PS_MEDIUM_PREMIUM

        if pd.notna(row.get("ROE")):
            if row["ROE"] > AE_THRESHOLD.ROE_HIGH:
                base_return += AE_PORTFOLIO.ROE_HIGH_PREMIUM
            elif row["ROE"] > AE_THRESHOLD.ROE_MEDIUM:
                base_return += AE_PORTFOLIO.ROE_MEDIUM_PREMIUM

        if pd.notna(row.get("dividend_yield")):
            base_return += row["dividend_yield"] * AE_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR
        elif pd.notna(row.get("Averange_dividend_yield")):
            div_yield = row["Averange_dividend_yield"] / 100
            base_return += div_yield * AE_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR

        return max(AE_PORTFOLIO.MIN_RETURN, min(AE_PORTFOLIO.MAX_RETURN, base_return))

    def calculate_volatility(self, row: pd.Series) -> float:
        """Расчет волатильности на основе результатов автоэнкодера"""
        base_vol = AE_PORTFOLIO.BASE_VOLATILITY

        recon_error = row.get("AE_Ошибка_реконструкции", 0.1)
        base_vol += recon_error * AE_PORTFOLIO.ERROR_VOL_FACTOR

        if row.get("AE_Аномалия", False):
            base_vol += AE_PORTFOLIO.ANOMALY_VOL_PENALTY
        if row.get("AE_Сильная_аномалия", False):
            base_vol += AE_PORTFOLIO.STRONG_ANOMALY_VOL_PENALTY

        if row.get("AE_Топ_недооцененные", False):
            base_vol += AE_PORTFOLIO.TOP_UNDERVALUED_VOL_BONUS

        if pd.notna(row.get("Бета")):
            beta = row["Бета"]
            base_vol *= (
                AE_PORTFOLIO.BETA_VOL_FACTOR_MIN
                + AE_PORTFOLIO.BETA_VOL_FACTOR_MAX * beta
            )

        if pd.notna(row.get("debt_capital")):
            debt = row["debt_capital"]
            base_vol *= (
                AE_PORTFOLIO.DEBT_VOL_FACTOR_MIN
                + AE_PORTFOLIO.DEBT_VOL_FACTOR_MAX
                * (debt / AE_PORTFOLIO.DEBT_NORMALIZATION)
            )

        return max(
            AE_PORTFOLIO.MIN_VOLATILITY, min(AE_PORTFOLIO.MAX_VOLATILITY, base_vol)
        )

    def create_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Создание матрицы ковариации с учетом корреляций"""
        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["AE_Volatility"].values

        recon_errors = df["AE_Ошибка_реконструкции"].values
        error_median = np.median(recon_errors)

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    error_diff = abs(recon_errors[i] - recon_errors[j])
                    if error_diff < error_median * AE_PORTFOLIO.ERROR_DIFF_LOW:
                        corr = AE_PORTFOLIO.ERROR_CORR_HIGH
                    elif error_diff < error_median * AE_PORTFOLIO.ERROR_DIFF_MEDIUM:
                        corr = AE_PORTFOLIO.ERROR_CORR_MEDIUM
                    else:
                        corr = AE_PORTFOLIO.ERROR_CORR_LOW
                    cov_matrix[i, j] = corr * risks[i] * risks[j]

        return cov_matrix

    def calculate_portfolio_metrics(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        port_return = np.sum(expected_returns * weights)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

        var_95 = port_return - AE_PORTFOLIO.VAR_95_COEFF * port_risk
        cvar_95 = port_return - AE_PORTFOLIO.CVAR_95_COEFF * port_risk

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
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        undervalued_boost: bool = True,
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
            options={"maxiter": AE_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_min_risk = minimize(
            portfolio_risk,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": AE_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_max_return = minimize(
            negative_return,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": AE_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        if undervalued_boost:
            combined_weights = (
                AE_PORTFOLIO.SHARPE_WEIGHT_BOOST * result_sharpe.x
                + AE_PORTFOLIO.MIN_RISK_WEIGHT_BOOST * result_min_risk.x
                + AE_PORTFOLIO.MAX_RETURN_WEIGHT_BOOST * result_max_return.x
            )
        else:
            combined_weights = (
                AE_PORTFOLIO.SHARPE_WEIGHT_NORMAL * result_sharpe.x
                + AE_PORTFOLIO.MIN_RISK_WEIGHT_NORMAL * result_min_risk.x
                + AE_PORTFOLIO.MAX_RETURN_WEIGHT_NORMAL * result_max_return.x
            )

        combined_weights = combined_weights / combined_weights.sum()

        return {
            "sharpe_weights": result_sharpe.x,
            "min_risk_weights": result_min_risk.x,
            "max_return_weights": result_max_return.x,
            "combined_weights": combined_weights,
            "cov_matrix": cov_matrix,
        }

    def create_ae_based_portfolios(self, df: pd.DataFrame) -> Dict:
        """Создание портфелей на основе результатов автоэнкодера"""
        portfolios = {}

        undervalued = df[df["AE_Топ_недооцененные"] == True].copy()
        if len(undervalued) > 0:
            weights = self._score_weighted_allocation(undervalued, "AE_Недооцененность")
            portfolios["Недооцененные"] = (undervalued, weights)

        no_anomalies = df[df["AE_Аномалия"] == False].copy()
        if len(no_anomalies) > 0:
            weights = self._risk_weighted_allocation(no_anomalies)
            portfolios["Без_аномалий"] = (no_anomalies, weights)

        combined = df[
            (df["AE_Топ_недооцененные"] == True)
            | (
                (df["AE_Аномалия"] == False)
                & (df["AE_Недооцененность"] > df["AE_Недооцененность"].median())
            )
        ].copy()
        if len(combined) > 0:
            weights = self._combined_score_allocation(combined)
            portfolios["Комбинированный"] = (combined, weights)

        dividend = df[
            (df["AE_Топ_недооцененные"] == True)
            & (df["dividend_yield"].fillna(0) > AE_PORTFOLIO.DIVIDEND_YIELD_THRESHOLD)
        ].copy()
        if len(dividend) > 0:
            weights = self._dividend_weighted_allocation(dividend)
            portfolios["Дивидендный"] = (dividend, weights)

        return portfolios

    def _score_weighted_allocation(
        self, df: pd.DataFrame, score_col: str
    ) -> np.ndarray:
        """Распределение на основе скора"""
        scores = df[score_col].values
        weights = scores / scores.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _risk_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе обратного риска"""
        risks = df["AE_Volatility"].values
        inv_risks = 1 / risks
        weights = inv_risks / inv_risks.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _dividend_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе дивидендной доходности"""
        div_yield = df["dividend_yield"].fillna(0).values
        weights = div_yield / div_yield.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _combined_score_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Комбинированное распределение (недооцененность + 1/риск)"""
        scores = df["AE_Недооцененность"].values
        inv_risks = 1 / df["AE_Volatility"].values

        scores_norm = scores / scores.sum()
        risks_norm = inv_risks / inv_risks.sum()

        weights = (
            AE_PORTFOLIO.UNDERVALUED_WEIGHT * scores_norm
            + AE_PORTFOLIO.RISK_WEIGHT * risks_norm
        )
        weights = weights / weights.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()
