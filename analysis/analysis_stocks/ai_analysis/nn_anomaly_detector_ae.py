import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import warnings

# Импорт констант
from ae_constants import (
    AE_ARCH,
    AE_THRESHOLD,
    AE_PORTFOLIO,
    AE_SCORING,
    AE_FEATURE,
    AE_COLUMN,
    AE_FILES,
    AE_FORMAT,
    AE_REC,
)

warnings.filterwarnings("ignore")


# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


class AEPathConfig:
    """Конфигурация путей к файлам для автоэнкодера"""

    @staticmethod
    def setup_directories():
        """Создание необходимых директорий"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nn_ae_detector_dir = f"{parent_dir}/../data/nn_ae_anomaly_detector"
        os.makedirs(nn_ae_detector_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "nn_ae_detector_dir": nn_ae_detector_dir,
            "input_file": f"{parent_dir}/../data/fundamentals_shares.xlsx",
            "output_file": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_RESULTS}",
            "ae_anomaly_file": f"{nn_ae_detector_dir}/{AE_FILES.AE_ANOMALY_ANALYSIS}",
            "ae_portfolio_comparison": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_COMPARISON}",
            "ae_portfolio_optimal": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_OPTIMAL}",
            "ae_portfolio_summary": f"{nn_ae_detector_dir}/{AE_FILES.AE_PORTFOLIO_SUMMARY}",
        }


AE_PATHS = AEPathConfig.setup_directories()


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


@dataclass
class StockRecommendation:
    """Рекомендация по акции"""

    ticker: str
    name: str
    undervalued_score: float
    anomaly_score: float
    expected_return: float
    volatility: float
    allocation_max: float
    risk_category: str


# ==================== КЛАСС АВТОЭНКОДЕРА ====================


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


# ==================== КЛАСС УПРАВЛЕНИЯ ПОРТФЕЛЕМ ====================


class AEPortfolioManager:
    """Управление портфелем на основе результатов автоэнкодера"""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        weights: np.ndarray,
        optimizer: AEPortfolioOptimizer,
    ):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.optimizer = optimizer

        expected_returns = self.df["AE_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        self.metrics = optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or AE_FORMAT.TOP_POSITIONS_SUMMARY
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()

    def get_risk_contribution(self) -> pd.Series:
        """Вклад в риск портфеля"""
        if len(self.df) == 0:
            return pd.Series()

        weights = self.weights
        risks = self.df["AE_Volatility"].values
        risk_contribution = weights * risks / np.sum(weights * risks)

        tickers = self.df.get("Тикер", self.df.index)
        return pd.Series(risk_contribution, index=tickers)

    def get_anomaly_allocation(self) -> Dict:
        """Распределение по статусу аномалий"""
        anomaly_mask = self.df["AE_Аномалия"] == True
        top_mask = self.df["AE_Топ_недооцененные"] == True

        return {
            AE_REC.CATEGORY_ANOMALIES: (
                self.weights[anomaly_mask].sum() if anomaly_mask.any() else 0
            ),
            AE_REC.CATEGORY_TOP_UNDERVALUED: (
                self.weights[top_mask].sum() if top_mask.any() else 0
            ),
            AE_REC.CATEGORY_NORMAL: self.weights[~(anomaly_mask | top_mask)].sum(),
        }


# ==================== КЛАСС ВИЗУАЛИЗАЦИИ ====================


class AEPortfolioVisualizer:
    """Визуализация портфелей на основе автоэнкодера"""

    @staticmethod
    def plot_portfolio_summary(
        portfolio_manager: AEPortfolioManager,
        filename: str = None,
    ):
        """Сводная визуализация портфеля"""
        if filename is None:
            filename = (
                f"{AE_PATHS['ae_portfolio_optimal']}_{portfolio_manager.name}.png"
            )

        fig, axes = plt.subplots(2, 2, figsize=AE_FILES.FIGURE_SIZE_SUMMARY)

        AEPortfolioVisualizer._plot_anomaly_allocation(portfolio_manager, axes[0, 0])
        AEPortfolioVisualizer._plot_risk_return_scatter(portfolio_manager, axes[0, 1])
        AEPortfolioVisualizer._plot_risk_contribution(portfolio_manager, axes[1, 0])
        AEPortfolioVisualizer._plot_portfolio_metrics(portfolio_manager, axes[1, 1])

        plt.suptitle(
            f"Портфель на основе автоэнкодера: {portfolio_manager.name}",
            fontsize=AE_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=AE_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_anomaly_allocation(pm: AEPortfolioManager, ax):
        """Визуализация распределения по статусу аномалий"""
        anomaly_alloc = pm.get_anomaly_allocation()

        if sum(anomaly_alloc.values()) > 0:
            labels = []
            sizes = []
            colors = []

            for key, value in anomaly_alloc.items():
                if value > AE_THRESHOLD.SIGNIFICANT_WEIGHT_THRESHOLD:
                    labels.append(key)
                    sizes.append(value)
                    colors.append(
                        AE_REC.CATEGORY_COLORS.get(key, AE_FORMAT.COLOR_NORMAL)
                    )

            if sizes:
                ax.pie(
                    sizes,
                    labels=labels,
                    autopct=AE_FORMAT.MATPLOTLIB_PERCENT,
                    startangle=90,
                    colors=colors,
                    explode=[AE_FORMAT.PIE_EXPLODE_FACTOR] * len(sizes),
                )
                ax.set_title(
                    "Распределение по категориям",
                    fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE,
                    fontweight="bold",
                )

    @staticmethod
    def _plot_risk_return_scatter(pm: AEPortfolioManager, ax):
        """Визуализация риск-доходность позиций"""
        scatter = ax.scatter(
            pm.df["AE_Volatility"],
            pm.df["AE_Expected_Return"],
            s=pm.weights * AE_FORMAT.WEIGHT_SCALE_FACTOR,
            c=pm.df["AE_Недооцененность"],
            cmap=AE_FORMAT.COLOR_CONFIDENCE_CMAP,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        top_n = min(AE_FORMAT.TOP_POSITIONS_SUMMARY, len(pm.df))
        top_positions = pm.get_top_positions(top_n)

        for _, row in top_positions.iterrows():
            ax.annotate(
                row.get("Тикер", "N/A"),
                (row["AE_Volatility"], row["AE_Expected_Return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=AE_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.scatter(
            pm.metrics.risk,
            pm.metrics.expected_return,
            s=AE_FORMAT.SCATTER_POINT_SIZE_PORTFOLIO,
            c=AE_FORMAT.COLOR_OPTIMAL_MARKER,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label=f"Портфель (Шарп: {pm.metrics.sharpe_ratio:.2f})",
        )

        ax.set_xlabel("Волатильность", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Ожидаемая доходность", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Risk-Return профиль",
            fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label="Скор недооцененности")

    @staticmethod
    def _plot_risk_contribution(pm: AEPortfolioManager, ax):
        """Визуализация вклада в риск портфеля"""
        risk_contrib = pm.get_risk_contribution()

        if len(risk_contrib) > 0:
            top_n = min(AE_FORMAT.TOP_RISK_CONTRIBUTION, len(risk_contrib))
            top_risk = risk_contrib.nlargest(top_n)

            colors = plt.cm.get_cmap(AE_FORMAT.COLOR_RISK_CONTRIBUTION_CMAP)(
                np.linspace(0.2, 0.8, len(top_risk))
            )
            bars = ax.barh(
                range(len(top_risk)), top_risk.values, color=colors, edgecolor="black"
            )

            ax.set_yticks(range(len(top_risk)))
            ax.set_yticklabels(top_risk.index)
            ax.set_xlabel("Вклад в риск", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
            ax.set_title(
                "Распределение риска",
                fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )
            ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5, axis="x")

            for bar, value in zip(bars, top_risk.values):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    ha="left",
                    va="center",
                    fontsize=AE_FORMAT.BAR_TEXT_FONT_SIZE,
                )

    @staticmethod
    def _plot_portfolio_metrics(pm: AEPortfolioManager, ax):
        """Визуализация метрик портфеля"""
        ax.axis("off")

        metrics_text = f"""
        📊 МЕТРИКИ ПОРТФЕЛЯ: {pm.name}
        
        Ожидаемая доходность: {AE_FORMAT.PERCENT_FORMAT.format(pm.metrics.expected_return)}
        Риск (волатильность): {AE_FORMAT.PERCENT_FORMAT.format(pm.metrics.risk)}
        Коэффициент Шарпа: {AE_FORMAT.FLOAT_FORMAT_2D.format(pm.metrics.sharpe_ratio)}
        
        📈 РИСК-МЕТРИКИ
        VaR (95%): {AE_FORMAT.PERCENT_FORMAT.format(pm.metrics.var_95)}
        CVaR (95%): {AE_FORMAT.PERCENT_FORMAT.format(pm.metrics.cvar_95)}
        
        📊 ДИВЕРСИФИКАЦИЯ
        Индекс диверсификации: {AE_FORMAT.PERCENT_FORMAT.format(pm.metrics.diversification_score)}
        Количество позиций: {len(pm.df)}
        Макс. доля: {AE_FORMAT.PERCENT_FORMAT.format(pm.weights.max())}
        
        🤖 АВТОЭНКОДЕР
        Средняя ошибка: {AE_FORMAT.FLOAT_FORMAT_6D.format(pm.df['AE_Ошибка_реконструкции'].mean())}
        Топ-недооцененных: {pm.df['AE_Топ_недооцененные'].sum()}
        """

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=AE_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=AE_FORMAT.COLOR_ANOMALY_BG,
                edgecolor=AE_FORMAT.COLOR_ANOMALY_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_portfolio_comparison(
        portfolios: Dict[str, AEPortfolioManager],
        filename: str = None,
    ):
        """Сравнение портфелей"""
        if not portfolios:
            return

        if filename is None:
            filename = AE_PATHS["ae_portfolio_comparison"]

        fig, axes = plt.subplots(2, 2, figsize=AE_FILES.FIGURE_SIZE_COMPARISON)

        names = []
        returns = []
        risks = []
        sharpes = []
        var_95s = []

        for name, pm in portfolios.items():
            names.append(name)
            returns.append(pm.metrics.expected_return)
            risks.append(pm.metrics.risk)
            sharpes.append(pm.metrics.sharpe_ratio)
            var_95s.append(pm.metrics.var_95)

        AEPortfolioVisualizer._plot_risk_return_comparison(
            axes[0, 0], names, returns, risks, sharpes
        )
        AEPortfolioVisualizer._plot_sharpe_comparison(axes[0, 1], names, sharpes)
        AEPortfolioVisualizer._plot_var_comparison(axes[1, 0], names, var_95s)
        AEPortfolioVisualizer._plot_best_portfolio_summary(axes[1, 1], portfolios)

        plt.suptitle(
            "Сравнение портфелей на основе автоэнкодера",
            fontsize=AE_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=AE_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_return_comparison(ax, names, returns, risks, sharpes):
        """Визуализация сравнения риск-доходность"""
        scatter = ax.scatter(
            risks,
            returns,
            c=sharpes,
            s=AE_FORMAT.SCATTER_POINT_SIZE_PORTFOLIO,
            cmap=AE_FORMAT.COLOR_SHARPE_CMAP,
            edgecolors="black",
            linewidths=1.5,
        )

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=AE_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Риск", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Доходность", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Risk-Return", fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE, fontweight="bold"
        )
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Коэффициент Шарпа", fontsize=AE_FORMAT.LABEL_FONT_SIZE)

    @staticmethod
    def _plot_sharpe_comparison(ax, names, sharpes):
        """Визуализация сравнения коэффициентов Шарпа"""
        sharpe_array = np.array(sharpes)
        sharpe_range = sharpe_array.max() - sharpe_array.min() + 0.001

        colors = plt.cm.get_cmap(AE_FORMAT.COLOR_SHARPE_CMAP)(
            (sharpe_array - sharpe_array.min()) / sharpe_range
        )

        bars = ax.bar(names, sharpes, color=colors, edgecolor="black")
        ax.axhline(
            y=1,
            color=AE_FORMAT.COLOR_OPTIMAL_MARKER,
            linestyle="--",
            alpha=0.5,
            label="Шарп = 1",
        )
        ax.set_xlabel("Портфель", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Коэффициент Шарпа", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение Шарпа", fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5, axis="y")

        for bar, value in zip(bars, sharpes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=AE_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_var_comparison(ax, names, var_95s):
        """Визуализация сравнения Value at Risk"""
        bars = ax.bar(
            names, var_95s, color=AE_FORMAT.COLOR_DIVERSIFICATION, edgecolor="black"
        )
        ax.set_xlabel("Портфель", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("VaR (95%)", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Value at Risk", fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE, fontweight="bold"
        )
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5, axis="y")

        for bar, value in zip(bars, var_95s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.01,
                f"{value:.1%}",
                ha="center",
                va="top",
                fontsize=AE_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
                color="white",
            )

    @staticmethod
    def _plot_best_portfolio_summary(ax, portfolios):
        """Визуализация информации о лучшем портфеле"""
        ax.axis("off")

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        top_n = min(AE_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        text = f"🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {best_portfolio.name}\n\n"
        text += f"Доходность: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}\n"
        text += (
            f"Риск: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}\n"
        )
        text += f"Шарп: {AE_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}\n"
        text += (
            f"VaR: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.var_95)}\n\n"
        )
        text += f"📈 ТОП-{top_n} ПОЗИЦИЙ:\n"

        for _, row in top_positions.iterrows():
            ticker = row.get("Тикер", "N/A")
            weight = row.get("Weight", 0)
            score = row.get("AE_Недооцененность", 0)
            text += f"• {ticker}: {AE_FORMAT.PERCENT_FORMAT.format(weight)} (скор: {score:.2f})\n"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=AE_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=AE_FORMAT.COLOR_ANOMALY_BG,
                edgecolor=AE_FORMAT.COLOR_ANOMALY_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_anomaly_analysis(df: pd.DataFrame, filename: str = None):
        """Анализ аномалий и недооцененных акций"""
        if filename is None:
            filename = AE_PATHS["ae_anomaly_file"]

        fig, axes = plt.subplots(2, 2, figsize=AE_FILES.FIGURE_SIZE_ANOMALY)

        AEPortfolioVisualizer._plot_error_distribution(df, axes[0, 0])
        AEPortfolioVisualizer._plot_score_vs_error(df, axes[0, 1])
        AEPortfolioVisualizer._plot_pe_vs_pb(df, axes[1, 0])
        AEPortfolioVisualizer._plot_ae_statistics(df, axes[1, 1])

        plt.suptitle(
            "Анализ аномалий и недооцененных акций",
            fontsize=AE_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=AE_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_error_distribution(df: pd.DataFrame, ax):
        """Визуализация распределения ошибок реконструкции"""
        errors = df["AE_Ошибка_реконструкции"].dropna()

        ax.hist(
            errors,
            bins=AE_FORMAT.HISTOGRAM_BINS,
            edgecolor="black",
            alpha=0.7,
            color=AE_FORMAT.COLOR_ANOMALY_EDGE,
        )
        ax.axvline(
            errors.median(),
            color=AE_FORMAT.COLOR_OPTIMAL_MARKER,
            linestyle="--",
            label=f"Медиана: {errors.median():.4f}",
        )
        ax.axvline(
            errors.quantile(AE_THRESHOLD.Q3_PERCENTILE / 100)
            + AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER
            * (
                errors.quantile(AE_THRESHOLD.Q3_PERCENTILE / 100)
                - errors.quantile(AE_THRESHOLD.Q1_PERCENTILE / 100)
            ),
            color=AE_REC.CATEGORY_COLORS.get(AE_REC.CATEGORY_ANOMALIES),
            linestyle="--",
            label="Порог аномалий",
        )
        ax.set_xlabel("Ошибка реконструкции", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Количество", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Распределение ошибок реконструкции",
            fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5)

    @staticmethod
    def _plot_score_vs_error(df: pd.DataFrame, ax):
        """Визуализация скор недооцененности vs ошибка"""
        scatter = ax.scatter(
            df["AE_Ошибка_реконструкции"],
            df["AE_Недооцененность"],
            c=df["AE_Недооцененность"],
            cmap=AE_FORMAT.COLOR_CONFIDENCE_CMAP,
            s=AE_FORMAT.SCATTER_POINT_SIZE,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        top_mask = df["AE_Топ_недооцененные"] == True
        if top_mask.any():
            ax.scatter(
                df[top_mask]["AE_Ошибка_реконструкции"],
                df[top_mask]["AE_Недооцененность"],
                s=AE_FORMAT.SCATTER_POINT_SIZE_LARGE,
                c=AE_REC.CATEGORY_COLORS.get(AE_REC.CATEGORY_TOP_UNDERVALUED),
                marker="*",
                edgecolors="black",
                linewidths=1,
                label="Топ-недооцененные",
            )

        ax.set_xlabel("Ошибка реконструкции", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Скор недооцененности", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Недооцененность vs Ошибка",
            fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5)

        plt.colorbar(scatter, ax=ax, label="Скор недооцененности")

    @staticmethod
    def _plot_pe_vs_pb(df: pd.DataFrame, ax):
        """Визуализация P/E vs P/B"""
        ax.scatter(
            df["P_B"],
            df["P_E"],
            c=df["AE_Недооцененность"],
            cmap=AE_FORMAT.COLOR_CONFIDENCE_CMAP,
            s=AE_FORMAT.SCATTER_POINT_SIZE,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        top_mask = df["AE_Топ_недооцененные"] == True
        if top_mask.any():
            ax.scatter(
                df[top_mask]["P_B"],
                df[top_mask]["P_E"],
                s=AE_FORMAT.SCATTER_POINT_SIZE_LARGE,
                c=AE_REC.CATEGORY_COLORS.get(AE_REC.CATEGORY_TOP_UNDERVALUED),
                marker="*",
                edgecolors="black",
                linewidths=1,
                label="Топ-недооцененные",
            )

        ax.set_xlabel("P/B", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("P/E", fontsize=AE_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "P/E vs P/B", fontsize=AE_FORMAT.SUBTITLE_FONT_SIZE, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER / 5)

    @staticmethod
    def _plot_ae_statistics(df: pd.DataFrame, ax):
        """Визуализация статистики автоэнкодера"""
        ax.axis("off")

        top_mask = df["AE_Топ_недооцененные"] == True

        text = f"""
        📊 СТАТИСТИКА АВТОЭНКОДЕРА
        
        Всего акций: {len(df)}
        Проанализировано: {df['AE_Ошибка_реконструкции'].notna().sum()}
        
        🚨 АНОМАЛИИ
        Аномалии (IQR): {df['AE_Аномалия'].sum()}
        Сильные аномалии: {df['AE_Сильная_аномалия'].sum()}
        
        🎯 НЕДООЦЕНЕННЫЕ
        Топ-недооцененных: {df['AE_Топ_недооцененные'].sum()}
        Средний скор: {df['AE_Недооцененность'].mean():.3f}
        Медианный скор: {df['AE_Недооцененность'].median():.3f}
        
        📈 ХАРАКТЕРИСТИКИ ТОП-НЕДООЦЕНЕННЫХ
        Средний P/E: {df[top_mask]['P_E'].mean():.1f}
        Средний P/B: {df[top_mask]['P_B'].mean():.2f}
        Средний ROE: {df[top_mask]['ROE'].mean():.1f}%
        """

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=AE_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=AE_FORMAT.COLOR_ANOMALY_BG,
                edgecolor=AE_FORMAT.COLOR_ANOMALY_EDGE,
                alpha=0.9,
            ),
        )


# ==================== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ====================


class AEDataLoader:
    """Загрузка и подготовка данных для автоэнкодера"""

    @staticmethod
    def load_and_prepare_excel_data(file_path):
        """Загрузка и подготовка данных из Excel файла"""
        df = pd.read_excel(file_path, sheet_name="Sheet1")

        for col in AE_COLUMN.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(
                        AE_COLUMN.BILLION_SUFFIX, AE_COLUMN.BILLION_REPLACE, regex=False
                    )
                )
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(
                        AE_COLUMN.MILLION_SUFFIX, AE_COLUMN.MILLION_REPLACE, regex=False
                    )
                )
                df[col] = df[col].astype(str).str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["dividend_yield"] = df["Дивидендная доходность"] / 100

        for old_col, new_col in AE_COLUMN.COLUMN_MAPPING.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        return df


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================


def detect_anomalies_with_ae(df):
    """Обнаружение аномалий и недооцененных акций с помощью автоэнкодера"""

    available_cols = [col for col in AE_FEATURE.DEFAULT_FEATURES if col in df.columns]
    print(f"Используемые признаки: {available_cols}")

    feature_data = []
    valid_indices = []
    tickers = []

    for idx, row in df.iterrows():
        feature_vector = []
        valid = True

        for col in available_cols:
            val = row.get(col, None)
            if pd.isna(val):
                valid = False
                break
            feature_vector.append(float(val))

        if valid and len(feature_vector) == len(available_cols):
            feature_data.append(feature_vector)
            valid_indices.append(idx)
            tickers.append(row.get("Тикер", f"Row_{idx}"))

    if not feature_data:
        print("Недостаточно данных для обучения автоэнкодера")
        return df, None, None

    X = np.array(feature_data)
    print(f"Обучаем автоэнкодер на {len(feature_data)} акциях")

    feature_medians = np.median(X, axis=0)
    feature_means = np.mean(X, axis=0)

    print("\nМедианные значения признаков:")
    for i, col in enumerate(available_cols):
        print(f"{col:<15}: Медиана = {feature_medians[i]:.4f}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    input_size = len(available_cols)
    model = AnomalyDetectorAE(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=AE_ARCH.LEARNING_RATE)

    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    dataloader = DataLoader(dataset, batch_size=AE_ARCH.BATCH_SIZE, shuffle=True)

    print("\nОбучение автоэнкодера...")
    for epoch in range(AE_ARCH.N_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % AE_ARCH.EPOCH_LOG_INTERVAL == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        _, reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        encoded, _ = model(X_tensor)
        encoded_features = encoded.numpy()

    errors_np = reconstruction_errors.numpy()

    error_median = np.median(errors_np)
    error_q1 = np.percentile(errors_np, AE_THRESHOLD.Q1_PERCENTILE)
    error_q3 = np.percentile(errors_np, AE_THRESHOLD.Q3_PERCENTILE)
    error_iqr = error_q3 - error_q1
    anomaly_threshold = error_q3 + AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER * error_iqr
    strong_anomaly_threshold = (
        error_q3 + AE_THRESHOLD.STRONG_ANOMALY_IQR_MULTIPLIER * error_iqr
    )

    is_anomaly = errors_np > anomaly_threshold
    is_strong_anomaly = errors_np > strong_anomaly_threshold

    print(f"\nПорог аномалий: {anomaly_threshold:.6f}")
    print(f"Найдено аномалий: {is_anomaly.sum()} из {len(errors_np)}")

    low_error_mask = errors_np < error_median
    if low_error_mask.sum() > AE_SCORING.GOOD_COMPANIES_MIN_COUNT:
        good_companies_features = X[low_error_mask]
        ideal_profile = np.median(good_companies_features, axis=0)
    else:
        ideal_profile = feature_medians

    undervalued_scores = []

    for i in range(len(X)):
        current_features = X[i]
        error_score = np.exp(-errors_np[i] / error_median)

        fundamental_score = 0
        for j, col in enumerate(available_cols):
            current_val = current_features[j]
            ideal_val = ideal_profile[j]

            if col in AE_FEATURE.LOWER_IS_BETTER_FEATURES:
                if (
                    0
                    < current_val
                    < ideal_val * AE_THRESHOLD.IDEAL_PROFILE_UNDERVALUED_FACTOR
                ):
                    fundamental_score += AE_SCORING.SCORE_STRONG_UNDERVALUED
                elif current_val < ideal_val:
                    fundamental_score += AE_SCORING.SCORE_UNDERVALUED
            elif col in AE_FEATURE.HIGHER_IS_BETTER_FEATURES:
                if (
                    current_val
                    > ideal_val * AE_THRESHOLD.IDEAL_PROFILE_OVERVAULED_FACTOR
                ):
                    fundamental_score += AE_SCORING.SCORE_STRONG_UNDERVALUED
                elif current_val > ideal_val:
                    fundamental_score += AE_SCORING.SCORE_UNDERVALUED

        combined_score = error_score * (
            1
            + fundamental_score
            / (len(available_cols) * AE_SCORING.FUNDAMENTAL_SCORE_FACTOR)
        )
        undervalued_scores.append(combined_score)

    df = AEDataUtils.add_ae_results_to_df(
        df,
        valid_indices,
        errors_np,
        is_anomaly,
        is_strong_anomaly,
        undervalued_scores,
        error_median,
    )

    AEDataUtils.print_top_undervalued(df)

    return df, model, scaler


class AEDataUtils:
    """Утилиты для работы с данными автоэнкодера"""

    @staticmethod
    def add_ae_results_to_df(
        df,
        valid_indices,
        errors_np,
        is_anomaly,
        is_strong_anomaly,
        undervalued_scores,
        error_median,
    ):
        """Добавление результатов автоэнкодера в DataFrame"""

        df["AE_Ошибка_реконструкции"] = np.nan
        df["AE_Ошибка_нормализованная"] = np.nan
        df["AE_Аномалия"] = False
        df["AE_Сильная_аномалия"] = False
        df["AE_Недооцененность"] = np.nan
        df["AE_Ранг_недооцененности"] = np.nan
        df["AE_Топ_недооцененные"] = False

        for i, idx in enumerate(valid_indices):
            df.at[idx, "AE_Ошибка_реконструкции"] = errors_np[i]
            df.at[idx, "AE_Ошибка_нормализованная"] = errors_np[i] / error_median
            df.at[idx, "AE_Аномалия"] = is_anomaly[i]
            df.at[idx, "AE_Сильная_аномалия"] = is_strong_anomaly[i]
            df.at[idx, "AE_Недооцененность"] = undervalued_scores[i]

        if undervalued_scores:
            scores_array = np.array(undervalued_scores)
            for i, idx in enumerate(valid_indices):
                percentile = (
                    (scores_array <= scores_array[i]).sum() / len(scores_array) * 100
                )
                df.at[idx, "AE_Ранг_недооцененности"] = percentile

        if len(undervalued_scores) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
            top_indices = np.argsort(undervalued_scores)[
                -AE_PORTFOLIO.TOP_UNDERVALUED_N :
            ][::-1]
            filtered_top = []

            for i in top_indices:
                if not is_anomaly[i]:
                    filtered_top.append(i)
                if len(filtered_top) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
                    break

            if len(filtered_top) < AE_PORTFOLIO.TOP_UNDERVALUED_N:
                for i in top_indices:
                    if i not in filtered_top:
                        filtered_top.append(i)
                    if len(filtered_top) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
                        break

            for i in filtered_top:
                df.at[valid_indices[i], "AE_Топ_недооцененные"] = True

        return df

    @staticmethod
    def print_top_undervalued(df):
        """Вывод топ-недооцененных акций"""
        print("\n" + AE_FORMAT.SEPARATOR)
        print("ТОП-10 НЕДООЦЕНЕННЫХ АКЦИЙ:")
        print(AE_FORMAT.SEPARATOR)

        undervalued_df = df[df["AE_Недооцененность"].notna()].copy()
        if not undervalued_df.empty:
            undervalued_df = undervalued_df.sort_values(
                "AE_Недооцененность", ascending=False
            )

            print(
                f"{'Тикер':<10} {'Название':<25} {'P/E':<6} {'P/B':<6} {'ДД,%':<6} "
                f"{'ROE,%':<7} {'Скор':<8} {'Ранг':<6}"
            )
            print(AE_FORMAT.SUB_SEPARATOR)

            for _, row in undervalued_df.head(15).iterrows():
                is_top = (
                    AE_FORMAT.STAR_SYMBOL
                    if row.get("AE_Топ_недооцененные", False)
                    else ""
                )
                print(
                    f"{row.get('Тикер', ''):<10} "
                    f"{str(row.get('Название', ''))[:23]:<25} "
                    f"{row.get('P_E', 0):<6.1f} "
                    f"{row.get('P_B', 0):<6.2f} "
                    f"{row.get('dividend_yield', 0)*100:<6.1f} "
                    f"{row.get('ROE', 0):<7.1f} "
                    f"{row.get('AE_Недооцененность', 0):<8.3f} "
                    f"{row.get('AE_Ранг_недооцененности', 0):<6.1f} {is_top}"
                )


def create_portfolios_from_ae_results(df, ae_optimizer):
    """Создание портфелей на основе результатов автоэнкодера"""

    print("\n" + AE_FORMAT.SEPARATOR)
    print("💼 ФОРМИРОВАНИЕ ПОРТФЕЛЕЙ НА ОСНОВЕ АВТОЭНКОДЕРА")
    print(AE_FORMAT.SEPARATOR)

    candidates = df[
        (df["AE_Недооцененность"].notna())
        & (df["AE_Ошибка_реконструкции"].notna())
        & (df["AE_Аномалия"] == False)
    ].copy()

    print(f"Кандидатов после фильтрации аномалий: {len(candidates)}")

    if len(candidates) < AE_PORTFOLIO.MIN_CANDIDATES:
        candidates = df[
            (df["AE_Недооцененность"].notna())
            & (df["AE_Недооцененность"] > df["AE_Недооцененность"].median())
        ].copy()
        print(f"Кандидатов после расширения: {len(candidates)}")

    candidates["AE_Expected_Return"] = candidates.apply(
        ae_optimizer.calculate_expected_return, axis=1
    )
    candidates["AE_Volatility"] = candidates.apply(
        ae_optimizer.calculate_volatility, axis=1
    )

    final_candidates = candidates[
        (candidates["AE_Expected_Return"] > AE_PORTFOLIO.MIN_EXPECTED_RETURN)
        & (candidates["AE_Volatility"] < AE_PORTFOLIO.MAX_VOLATILITY_THRESHOLD)
    ].copy()

    if len(final_candidates) > AE_PORTFOLIO.MAX_CANDIDATES:
        final_candidates = final_candidates.nlargest(
            AE_PORTFOLIO.MAX_CANDIDATES, "AE_Недооцененность"
        )

    print(f"Финальных кандидатов: {len(final_candidates)}")

    portfolios = {}

    if len(final_candidates) >= AE_PORTFOLIO.MIN_PORTFOLIO_SIZE:
        expected_returns = final_candidates["AE_Expected_Return"].values
        cov_matrix = ae_optimizer.create_covariance_matrix(final_candidates)
        opt_result = ae_optimizer.optimize_portfolio(
            expected_returns, cov_matrix, undervalued_boost=True
        )

        pm = AEPortfolioManager(
            "Марковиц-оптимальный",
            final_candidates.reset_index(drop=True),
            opt_result["combined_weights"],
            ae_optimizer,
        )
        portfolios["Марковиц-оптимальный"] = pm
        print(f"✅ Марковиц-оптимальный: Шарп={pm.metrics.sharpe_ratio:.2f}")

    ae_portfolios = ae_optimizer.create_ae_based_portfolios(final_candidates)

    for name, (df_port, weights) in ae_portfolios.items():
        if len(df_port) >= AE_PORTFOLIO.MIN_PORTFOLIO_SIZE:
            pm = AEPortfolioManager(
                name, df_port.reset_index(drop=True), weights, ae_optimizer
            )
            portfolios[name] = pm
            print(f"✅ {name}: Шарп={pm.metrics.sharpe_ratio:.2f}")

    return portfolios, final_candidates


def save_portfolio_results(df, portfolios, candidates, output_path):
    """Сохранение результатов портфельного анализа"""

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=AE_FILES.SHEET_AE_RESULTS, index=False)
        candidates.to_excel(writer, sheet_name=AE_FILES.SHEET_CANDIDATES, index=False)

        if portfolios:
            summary = []
            for name, pm in portfolios.items():
                summary.append(
                    {
                        "Портфель": name,
                        "Доходность": f"{pm.metrics.expected_return:.2%}",
                        "Риск": f"{pm.metrics.risk:.2%}",
                        "Шарп": f"{pm.metrics.sharpe_ratio:.2f}",
                        "VaR": f"{pm.metrics.var_95:.2%}",
                        "Диверсификация": f"{pm.metrics.diversification_score:.1%}",
                        "Позиций": len(pm.df),
                        "Топ_недооцененных": pm.df["AE_Топ_недооцененные"].sum(),
                    }
                )
            pd.DataFrame(summary).to_excel(
                writer, sheet_name=AE_FILES.SHEET_PORTFOLIO_SUMMARY, index=False
            )

            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )
            best_portfolio.df.to_excel(
                writer, sheet_name=AE_FILES.SHEET_BEST_PORTFOLIO, index=False
            )

    print(f"✅ Результаты сохранены в {output_path}")


def main():
    """Основная функция"""

    print(AE_FORMAT.SEPARATOR)
    print("🚀 АВТОЭНКОДЕР + ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ПО МАРКОВИЦУ")
    print(AE_FORMAT.SEPARATOR)

    try:
        print("\n📥 Загрузка данных...")
        df = AEDataLoader.load_and_prepare_excel_data(AE_PATHS["input_file"])
        print(f"Загружено {len(df)} акций")

        print("\n🧠 Обучение автоэнкодера...")
        df_with_ae, model, scaler = detect_anomalies_with_ae(df)

        if model is None:
            print("❌ Ошибка обучения автоэнкодера")
            return

        print("\n📊 Визуализация анализа аномалий...")
        AEPortfolioVisualizer.plot_anomaly_analysis(
            df_with_ae, AE_PATHS["ae_anomaly_file"]
        )

        ae_optimizer = AEPortfolioOptimizer(
            min_weight=AE_PORTFOLIO.MIN_WEIGHT,
            max_weight=AE_PORTFOLIO.MAX_WEIGHT,
            risk_free_rate=AE_PORTFOLIO.RISK_FREE_RATE,
        )

        portfolios, candidates = create_portfolios_from_ae_results(
            df_with_ae, ae_optimizer
        )

        if portfolios:
            print("\n📊 Визуализация портфелей...")

            AEPortfolioVisualizer.plot_portfolio_comparison(
                portfolios, AE_PATHS["ae_portfolio_comparison"]
            )

            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )
            AEPortfolioVisualizer.plot_portfolio_summary(best_portfolio)

        save_portfolio_results(
            df_with_ae, portfolios, candidates, AE_PATHS["output_file"]
        )

        print("\n" + AE_FORMAT.SEPARATOR)
        print("🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
        print(AE_FORMAT.SEPARATOR)

        if portfolios:
            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )

            print(f"\n🏆 РЕКОМЕНДУЕМЫЙ ПОРТФЕЛЬ: {best_portfolio.name}")
            print(
                f"   Ожидаемая доходность: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}"
            )
            print(
                f"   Риск: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}"
            )
            print(
                f"   Коэффициент Шарпа: {AE_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}"
            )

            print(f"\n📈 ТОП-5 ПОЗИЦИЙ В ПОРТФЕЛЕ:")
            top_n = min(AE_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
            top_5 = best_portfolio.get_top_positions(top_n)

            for _, row in top_5.iterrows():
                ticker = row.get("Тикер", "N/A")
                weight = row.get("Weight", 0)
                company = str(row.get("Название", ""))[:30]
                score = row.get("AE_Недооцененность", 0)
                rank = row.get("AE_Ранг_недооцененности", 0)

                print(
                    f"   • {ticker}: {AE_FORMAT.PERCENT_FORMAT.format(weight)} - {company}"
                )
                print(f"     Скор: {score:.3f}, Ранг: {rank:.1f}%")

            print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
            anomaly_alloc = best_portfolio.get_anomaly_allocation()
            for category, weight in anomaly_alloc.items():
                if weight > 0:
                    print(f"   • {category}: {AE_FORMAT.PERCENT_FORMAT.format(weight)}")

        print("\n" + AE_FORMAT.SEPARATOR)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print(AE_FORMAT.SEPARATOR)

    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
