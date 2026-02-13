# metals_portfolio_optimization.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from analysis_commodities_constants import (
    START_DATE,
    END_DATE,
    MARKET_DATA_DIR,
    RESULTS_DIR,
    METALS_DATA_FILES,
    METAL_NAMES,
    COLORS,
    VAR_CONFIDENCE_LEVELS,
    RISK_FREE_RATE,
    TRADING_DAYS,
    NUM_PORTFOLIOS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    WEIGHT_THRESHOLD,
    SIGNIFICANT_WEIGHT_THRESHOLD,
    EF_POINTS,
    PLOT_STYLE,
    BENCHMARK_INDEX,
    OUTPUT_FILES,
    PRECIOUS_METALS,
    INDUSTRIAL_METALS,
    ENERGY,
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


class MetalsDataLoader:
    """Класс для загрузки и предобработки данных по металлам"""

    @staticmethod
    def load_metals_data() -> pd.DataFrame:
        """Загрузка данных из CSV файлов металлов"""
        print(f"   Загрузка данных из: {MARKET_DATA_DIR}")
        data = {}

        for name, file in METALS_DATA_FILES.items():
            file_path = os.path.join(MARKET_DATA_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            # Для некоторых металлов Volume может быть 0 или отсутствовать
            if "Close" in df.columns:
                data[name] = df["Close"]
            else:
                # Если нет Close, используем последнюю доступную колонку
                data[name] = df.iloc[:, 0]

        prices = pd.DataFrame(data)

        # Фильтруем по датам
        mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
        prices = prices[mask]

        # Удаляем строки с пропущенными значениями
        prices = prices.dropna()

        return prices


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


class MetalsCorrelationAnalyzer:
    """Класс для корреляционного анализа металлов"""

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
                        "metal1": matrix.columns[i],
                        "metal2": matrix.columns[j],
                        "correlation": matrix.iloc[i, j],
                    }
                )

        return sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:n]

    def get_correlations_by_group(self) -> pd.DataFrame:
        """Получение корреляций по группам металлов"""
        if self.correlation_matrix is None:
            return pd.DataFrame()

        # Корреляции внутри групп
        precious_corr = (
            self.correlation_matrix.loc[PRECIOUS_METALS, PRECIOUS_METALS].mean().mean()
        )
        industrial_corr = (
            self.correlation_matrix.loc[INDUSTRIAL_METALS, INDUSTRIAL_METALS]
            .mean()
            .mean()
        )

        # Корреляции между группами
        precious_industrial = (
            self.correlation_matrix.loc[PRECIOUS_METALS, INDUSTRIAL_METALS]
            .mean()
            .mean()
        )
        precious_energy = (
            self.correlation_matrix.loc[PRECIOUS_METALS, ENERGY].mean().mean()
        )
        industrial_energy = (
            self.correlation_matrix.loc[INDUSTRIAL_METALS, ENERGY].mean().mean()
        )

        return pd.DataFrame(
            {
                "Group": ["Precious Metals", "Industrial Metals", "Energy"],
                "Intra-group Correlation": [precious_corr, industrial_corr, 1.0],
                "Correlation with Precious": [
                    1.0,
                    precious_industrial,
                    precious_energy,
                ],
                "Correlation with Industrial": [
                    precious_industrial,
                    1.0,
                    industrial_energy,
                ],
                "Correlation with Energy": [precious_energy, industrial_energy, 1.0],
            }
        )

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
            xticklabels=[METAL_NAMES[x] for x in self.correlation_matrix.columns],
            yticklabels=[METAL_NAMES[x] for x in self.correlation_matrix.columns],
        )

        plt.title(
            "Correlation Matrix of Metals & Commodities",
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


class MetalsVisualizer:
    """Класс для визуализации результатов анализа металлов"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def plot_prices_normalized(
        self, prices: pd.DataFrame, show_plot: bool = True
    ) -> None:
        """Построение графика нормализованных цен"""
        plt.figure(figsize=(14, 8))

        normalized_prices = prices / prices.iloc[0] * 100

        for column in normalized_prices.columns:
            plt.plot(
                normalized_prices.index,
                normalized_prices[column],
                label=METAL_NAMES[column],
                color=COLORS.get(column, "#333333"),
                linewidth=2,
            )

        plt.title(
            "Metals Prices Normalized (100 = Start Date)",
            fontsize=PLOT_STYLE["title_fontsize"],
        )
        plt.xlabel("Date", fontsize=PLOT_STYLE["label_fontsize"])
        plt.ylabel("Normalized Price", fontsize=PLOT_STYLE["label_fontsize"])
        plt.legend(loc="upper left", fontsize=PLOT_STYLE["legend_fontsize"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['prices_chart']}.png"
        )
        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_returns_distribution(
        self, returns: pd.DataFrame, show_plot: bool = True
    ) -> None:
        """Построение распределения доходностей"""
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()

        for i, (col, ax) in enumerate(zip(returns.columns, axes)):
            if i >= len(returns.columns):
                break

            ax.hist(
                returns[col],
                bins=50,
                color=COLORS.get(col, "#333333"),
                alpha=0.7,
                edgecolor="black",
            )
            ax.axvline(
                returns[col].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {returns[col].mean():.4f}",
            )
            ax.axvline(0, color="black", linestyle="-", linewidth=1)

            ax.set_title(METAL_NAMES[col], fontsize=12)
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Убираем лишние подграфики
        for j in range(len(returns.columns), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(
            "Distribution of Daily Returns",
            fontsize=PLOT_STYLE["title_fontsize"],
            y=1.02,
        )
        plt.tight_layout()

        save_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['returns_distribution']}.png"
        )
        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_efficient_frontier(
        self,
        returns: pd.DataFrame,
        efficient_portfolios: pd.DataFrame,
        max_sharpe_portfolio: Dict,
        min_vol_portfolio: Dict,
        simulated_results: np.ndarray,
        show_plot: bool = True,
    ) -> None:
        """Построение графика эффективной границы"""
        plt.figure(figsize=PLOT_STYLE["figure_size"]["medium"])

        # Симулированные портфели
        scatter = plt.scatter(
            simulated_results[1, :],
            simulated_results[0, :],
            c=simulated_results[2, :],
            cmap="viridis",
            alpha=0.3,
            s=10,
        )

        # Эффективная граница
        if not efficient_portfolios.empty:
            plt.plot(
                efficient_portfolios["volatility"],
                efficient_portfolios["return"],
                color=COLORS["efficient_frontier"],
                linewidth=3,
                label="Efficient Frontier",
            )

        # Портфель с максимальным Sharpe
        plt.scatter(
            max_sharpe_portfolio["volatility"],
            max_sharpe_portfolio["return"],
            color=COLORS["max_sharpe"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Maximum Sharpe Ratio",
        )

        # Портфель с минимальной волатильностью
        plt.scatter(
            min_vol_portfolio["volatility"],
            min_vol_portfolio["return"],
            color=COLORS["min_vol"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Minimum Volatility",
        )

        # Отдельные металлы
        for i, idx in enumerate(returns.columns):
            ret = returns.mean().iloc[i] * TRADING_DAYS
            vol = returns.std().iloc[i] * np.sqrt(TRADING_DAYS)
            plt.scatter(
                vol,
                ret,
                s=150,
                color=COLORS.get(idx, "#333333"),
                edgecolors="black",
                linewidth=2,
                alpha=0.8,
            )
            plt.annotate(
                METAL_NAMES[idx],
                (vol, ret),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=PLOT_STYLE["annotation_fontsize"],
                fontweight="bold",
            )

        plt.colorbar(scatter, label="Sharpe Ratio")
        plt.xlabel(
            "Expected Volatility (Annual)", fontsize=PLOT_STYLE["label_fontsize"]
        )
        plt.ylabel("Expected Return (Annual)", fontsize=PLOT_STYLE["label_fontsize"])
        plt.title(
            "Markowitz Portfolio Optimization - Metals & Commodities",
            fontsize=PLOT_STYLE["title_fontsize"],
            pad=20,
        )
        plt.legend(loc="upper left", fontsize=PLOT_STYLE["legend_fontsize"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['efficient_frontier']}.png"
        )
        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_weights_comparison(
        self,
        max_sharpe_weights: pd.Series,
        min_vol_weights: pd.Series,
        returns: pd.DataFrame,
        show_plot: bool = True,
    ) -> None:
        """Построение графика сравнения весов портфелей"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_STYLE["figure_size"]["small"])

        # Фильтруем нулевые веса
        max_sharpe_nonzero = max_sharpe_weights[max_sharpe_weights > WEIGHT_THRESHOLD]
        min_vol_nonzero = min_vol_weights[min_vol_weights > WEIGHT_THRESHOLD]

        # Максимальный Sharpe
        if len(max_sharpe_nonzero) > 0:
            colors_max = [
                COLORS.get(idx, "#333333") for idx in max_sharpe_nonzero.index
            ]
            wedges, texts, autotexts = ax1.pie(
                max_sharpe_nonzero.values,
                labels=[METAL_NAMES[x] for x in max_sharpe_nonzero.index],
                autopct="%1.1f%%",
                colors=colors_max,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )
            # Добавляем легенду с весами
            ax1.set_title("Maximum Sharpe Ratio Portfolio", fontsize=14, pad=20)
        else:
            ax1.text(0.5, 0.5, "No significant weights", ha="center", va="center")
            ax1.set_title("Maximum Sharpe Ratio Portfolio", fontsize=14, pad=20)

        # Минимальная волатильность
        if len(min_vol_nonzero) > 0:
            colors_min = [COLORS.get(idx, "#333333") for idx in min_vol_nonzero.index]
            wedges, texts, autotexts = ax2.pie(
                min_vol_nonzero.values,
                labels=[METAL_NAMES[x] for x in min_vol_nonzero.index],
                autopct="%1.1f%%",
                colors=colors_min,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )
            ax2.set_title("Minimum Volatility Portfolio", fontsize=14, pad=20)
        else:
            ax2.text(0.5, 0.5, "No significant weights", ha="center", va="center")
            ax2.set_title("Minimum Volatility Portfolio", fontsize=14, pad=20)

        plt.tight_layout()

        save_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['weights_comparison']}.png"
        )
        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_group_allocation(
        self,
        max_sharpe_metrics: "MetalsPortfolioMetrics",
        min_vol_metrics: "MetalsPortfolioMetrics",
        show_plot: bool = True,
    ) -> None:
        """Построение графика распределения по группам металлов"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Данные для графиков
        groups = ["Precious Metals", "Industrial Metals", "Energy"]
        colors = ["#FFD700", "#B87333", "#3A5F40"]

        # Max Sharpe
        max_sharpe_allocation = [
            max_sharpe_metrics.precious_metals_weight,
            max_sharpe_metrics.industrial_metals_weight,
            max_sharpe_metrics.energy_weight,
        ]

        # Min Vol
        min_vol_allocation = [
            min_vol_metrics.precious_metals_weight,
            min_vol_metrics.industrial_metals_weight,
            min_vol_metrics.energy_weight,
        ]

        ax1.bar(
            groups, max_sharpe_allocation, color=colors, alpha=0.8, edgecolor="black"
        )
        ax1.set_title("Maximum Sharpe Ratio - Group Allocation", fontsize=14)
        ax1.set_ylabel("Weight")
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis="y")

        # Добавляем значения на столбцы
        for i, v in enumerate(max_sharpe_allocation):
            ax1.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

        ax2.bar(groups, min_vol_allocation, color=colors, alpha=0.8, edgecolor="black")
        ax2.set_title("Minimum Volatility - Group Allocation", fontsize=14)
        ax2.set_ylabel("Weight")
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis="y")

        # Добавляем значения на столбцы
        for i, v in enumerate(min_vol_allocation):
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

        plt.suptitle("Portfolio Allocation by Metal Group", fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, "metals_group_allocation.png")
        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()


class MetalsExcelReportGenerator:
    """Класс для генерации Excel отчетов по металлам"""

    @staticmethod
    def save_to_excel(
        results_dir: str,
        correlation_matrix: pd.DataFrame,
        stats: pd.DataFrame,
        skewness_kurtosis: pd.DataFrame,
        max_sharpe_metrics: MetalsPortfolioMetrics,
        min_vol_metrics: MetalsPortfolioMetrics,
        efficient_portfolios: pd.DataFrame,
        returns: pd.DataFrame,
        group_correlations: pd.DataFrame = None,
    ) -> None:
        """Сохранение всех результатов в Excel файл"""

        file_path = os.path.join(results_dir, f"{OUTPUT_FILES['full_report']}.xlsx")

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

            # 1. Статистика по металлам
            stats.to_excel(writer, sheet_name="Metal Statistics", startrow=1)
            worksheet = writer.sheets["Metal Statistics"]
            worksheet.cell(
                row=1, column=1, value="Годовая статистика по металлам (в %)"
            )

            # 2. Skewness и Kurtosis
            skewness_kurtosis.to_excel(
                writer, sheet_name="Returns Distribution", startrow=1
            )
            worksheet = writer.sheets["Returns Distribution"]
            worksheet.cell(
                row=1, column=1, value="Статистика распределения доходностей"
            )

            # 3. Корреляционная матрица
            correlation_matrix.to_excel(
                writer, sheet_name="Correlation Matrix", startrow=1
            )
            worksheet = writer.sheets["Correlation Matrix"]
            worksheet.cell(
                row=1, column=1, value="Корреляционная матрица доходностей металлов"
            )

            # 4. Корреляции по группам
            if group_correlations is not None:
                group_correlations.to_excel(
                    writer, sheet_name="Group Correlations", index=False, startrow=1
                )
                worksheet = writer.sheets["Group Correlations"]
                worksheet.cell(
                    row=1, column=1, value="Корреляции между группами металлов"
                )

            # 5. Веса портфелей
            weights_df = pd.DataFrame(
                {
                    "Metal": max_sharpe_metrics.weights.index,
                    "Max Sharpe Weights": max_sharpe_metrics.weights.values,
                    "Min Volatility Weights": min_vol_metrics.weights.values,
                }
            )
            weights_df = weights_df.round(4)
            weights_df.to_excel(
                writer, sheet_name="Portfolio Weights", index=False, startrow=1
            )
            worksheet = writer.sheets["Portfolio Weights"]
            worksheet.cell(
                row=1, column=1, value="Веса металлов в оптимальных портфелях"
            )

            # 6. Метрики портфелей
            metrics_data = []
            for name, metrics in [
                ("Max Sharpe", max_sharpe_metrics),
                ("Min Volatility", min_vol_metrics),
            ]:
                data = metrics.to_dict()
                data["Portfolio"] = name
                metrics_data.append(data)

            metrics_df = pd.DataFrame(metrics_data)
            metrics_df = metrics_df[
                [
                    "Portfolio",
                    "Expected Return",
                    "Volatility",
                    "Sharpe Ratio",
                    "Beta",
                    "VaR 95%",
                    "CVaR 95%",
                    "VaR 99%",
                    "CVaR 99%",
                    "HHI",
                    "Number of Assets",
                    "Precious Metals Weight",
                    "Industrial Metals Weight",
                    "Energy Weight",
                ]
            ]
            metrics_df = metrics_df.round(4)
            metrics_df.to_excel(
                writer, sheet_name="Portfolio Metrics", index=False, startrow=1
            )
            worksheet = writer.sheets["Portfolio Metrics"]
            worksheet.cell(
                row=1, column=1, value="Метрики оптимальных портфелей металлов"
            )

            # 7. Эффективная граница
            if not efficient_portfolios.empty:
                ef_display = efficient_portfolios[["return", "volatility"]].copy()
                ef_display.columns = ["Expected Return", "Volatility"]
                ef_display = ef_display.round(4)
                ef_display.to_excel(
                    writer, sheet_name="Efficient Frontier", index=False, startrow=1
                )
                worksheet = writer.sheets["Efficient Frontier"]
                worksheet.cell(row=1, column=1, value="Точки эффективной границы")

            # 8. Доходности
            returns_stats = returns.describe().round(6)
            returns_stats.to_excel(writer, sheet_name="Returns Statistics", startrow=1)
            worksheet = writer.sheets["Returns Statistics"]
            worksheet.cell(row=1, column=1, value="Статистика дневных доходностей")

            # 9. Информация о периоде
            info_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Start Date",
                        "End Date",
                        "Risk Free Rate",
                        "Trading Days",
                        "Number of Metals",
                        "Benchmark",
                        "Optimization Date",
                    ],
                    "Value": [
                        START_DATE,
                        END_DATE,
                        f"{RISK_FREE_RATE:.1%}",
                        TRADING_DAYS,
                        len(max_sharpe_metrics.weights),
                        METAL_NAMES.get(BENCHMARK_INDEX, BENCHMARK_INDEX),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                }
            )
            info_df.to_excel(
                writer, sheet_name="Analysis Info", index=False, startrow=1
            )
            worksheet = writer.sheets["Analysis Info"]
            worksheet.cell(row=1, column=1, value="Параметры анализа металлов")

        print(f"\n   Excel отчет сохранен: {file_path}")


class MetalsAnalysisReport:
    """Класс для генерации отчетов в консоли по металлам"""

    @staticmethod
    def print_header(title: str) -> None:
        """Печать заголовка"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def print_section(title: str) -> None:
        """Печать секции"""
        print(f"\n{title}")
        print("-" * 60)

    @staticmethod
    def print_metal_statistics(stats: pd.DataFrame) -> None:
        """Печать статистики по металлам"""
        MetalsAnalysisReport.print_section("СТАТИСТИКА ПО МЕТАЛЛАМ (ГОДОВАЯ)")

        for metal in stats.index:
            name = METAL_NAMES.get(metal, metal)
            ret = stats.loc[metal, "Annual Return (%)"]
            vol = stats.loc[metal, "Annual Volatility (%)"]
            sharpe = stats.loc[metal, "Sharpe Ratio"]
            print(
                f"   {name:<20} Return: {ret:>6.1f}% | Vol: {vol:>6.1f}% | Sharpe: {sharpe:>6.2f}"
            )

    @staticmethod
    def print_portfolio_metrics(name: str, metrics: MetalsPortfolioMetrics) -> None:
        """Печать метрик портфеля металлов"""
        MetalsAnalysisReport.print_section(name)

        # Веса
        significant_weights = metrics.weights[metrics.weights > WEIGHT_THRESHOLD]
        for idx, weight in significant_weights.sort_values(ascending=False).items():
            print(f"   {METAL_NAMES[idx]:<20} {weight:.4f} ({weight*100:.1f}%)")

        # Распределение по группам
        print(f"\n   Распределение по группам:")
        print(f"   Драгоценные металлы: {metrics.precious_metals_weight:.1%}")
        print(f"   Промышленные металлы: {metrics.industrial_metals_weight:.1%}")
        print(f"   Энергоносители: {metrics.energy_weight:.1%}")

        # Метрики
        print(f"\n   Ожидаемая годовая доходность: {metrics.expected_return*100:.2f}%")
        print(f"   Ожидаемая годовая волатильность: {metrics.volatility*100:.2f}%")
        print(f"   Коэффициент Шарпа: {metrics.sharpe_ratio:.4f}")
        print(
            f"   Бета коэффициент (vs {METAL_NAMES.get(BENCHMARK_INDEX, BENCHMARK_INDEX)}): {metrics.beta:.4f}"
        )
        print(
            f"   VaR (95%): {metrics.var_95*100:.2f}%, CVaR (95%): {metrics.cvar_95*100:.2f}%"
        )
        print(
            f"   VaR (99%): {metrics.var_99*100:.2f}%, CVaR (99%): {metrics.cvar_99*100:.2f}%"
        )
        print(f"   Индекс Херфиндаля-Хиршмана: {metrics.hhi:.4f}")
        print(f"   Количество активов (вес > 1%): {metrics.n_assets}")


class MetalsAnalysisPipeline:
    """Основной класс для запуска полного анализа металлов"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_loader = MetalsDataLoader()
        self.stats_calculator = MetalsStatisticsCalculator()
        self.correlation_analyzer = MetalsCorrelationAnalyzer()
        self.risk_calculator = MetalsRiskMetricsCalculator()
        self.visualizer = MetalsVisualizer(self.results_dir)
        self.reporter = MetalsAnalysisReport()
        self.excel_reporter = MetalsExcelReportGenerator()

        self.prices = None
        self.returns = None
        self.stats = None
        self.correlation_matrix = None
        self.optimizer = None

    def run(self) -> Dict[str, Any]:
        """Запуск полного цикла анализа металлов"""

        self.reporter.print_header(
            "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ МЕТАЛЛОВ"
        )

        # 1. Загрузка данных
        print("\n1. Загрузка данных по металлам...")
        self.prices = self.data_loader.load_metals_data()
        print(f"   Загружено {len(self.prices.columns)} металлов/сырьевых товаров")
        print(
            f"   Период: с {self.prices.index[0].date()} по {self.prices.index[-1].date()}"
        )
        print(f"   Торговых дней: {len(self.prices)}")

        # 2. Визуализация цен
        print("\n2. Визуализация цен...")
        self.visualizer.plot_prices_normalized(self.prices, self.show_plots)

        # 3. Расчет доходностей и статистик
        print("\n3. Расчет статистик...")
        self.returns = self.stats_calculator.calculate_returns(self.prices)
        self.stats = self.stats_calculator.calculate_annual_statistics(self.returns)

        # Добавляем максимальную просадку
        max_drawdowns = self.stats_calculator.calculate_all_max_drawdowns(self.prices)
        self.stats["Max Drawdown (%)"] = pd.Series(max_drawdowns) * 100

        # Добавляем skewness и kurtosis
        skew_kurt = self.stats_calculator.calculate_skewness_kurtosis(self.returns)

        print("\n   Статистика по металлам (годовая):")
        print(self.stats.round(2))

        # 4. Распределение доходностей
        print("\n4. Анализ распределения доходностей...")
        self.visualizer.plot_returns_distribution(self.returns, self.show_plots)

        # 5. Корреляционный анализ
        print("\n5. Корреляционный анализ...")
        self.correlation_matrix = self.correlation_analyzer.analyze(self.returns)

        # Сохраняем график
        corr_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['correlation_matrix']}.png"
        )
        self.correlation_analyzer.plot(self.returns, str(corr_path), self.show_plots)

        # Корреляции по группам
        group_correlations = self.correlation_analyzer.get_correlations_by_group()

        # Вывод наиболее коррелированных пар
        print("\n   Наиболее коррелированные пары металлов:")
        top_correlations = self.correlation_analyzer.get_top_correlations(5)
        for pair in top_correlations:
            print(
                f"   {METAL_NAMES[pair['metal1']]:<15} - {METAL_NAMES[pair['metal2']]:<15}: {pair['correlation']:.4f}"
            )

        # 6. Оптимизация портфеля
        print("\n6. Оптимизация портфеля металлов по Марковицу...")
        self.optimizer = MetalsPortfolioOptimizer(self.returns)

        # Портфель с максимальным Sharpe
        max_sharpe_weights = self.optimizer.optimize_max_sharpe()
        max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe = (
            self.optimizer.portfolio_statistics(max_sharpe_weights)
        )

        # Портфель с минимальной волатильностью
        min_vol_weights = self.optimizer.optimize_min_volatility()
        min_vol_ret, min_vol_vol, min_vol_sharpe = self.optimizer.portfolio_statistics(
            min_vol_weights
        )

        # 7. Симуляция портфелей
        print("\n7. Симуляция случайных портфелей...")
        simulated_results, _ = self.optimizer.simulate_portfolios()

        # 8. Эффективная граница
        print("\n8. Построение эффективной границы...")
        efficient_portfolios = self.optimizer.efficient_frontier()
        print(f"   Построено {len(efficient_portfolios)} точек эффективной границы")

        # 9. Расчет метрик риска
        print("\n9. Расчет метрик риска...")

        # Создаем объекты с метриками для портфелей
        max_sharpe_weights_series = pd.Series(
            max_sharpe_weights, index=self.returns.columns
        )
        min_vol_weights_series = pd.Series(min_vol_weights, index=self.returns.columns)

        # Веса по группам
        max_sharpe_group_weights = self.risk_calculator.calculate_group_weights(
            max_sharpe_weights_series
        )
        min_vol_group_weights = self.risk_calculator.calculate_group_weights(
            min_vol_weights_series
        )

        max_sharpe_metrics = MetalsPortfolioMetrics(
            weights=max_sharpe_weights_series,
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, max_sharpe_weights),
            hhi=self.risk_calculator.calculate_hhi(max_sharpe_weights),
            n_assets=len(
                max_sharpe_weights[max_sharpe_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
            precious_metals_weight=max_sharpe_group_weights["precious"],
            industrial_metals_weight=max_sharpe_group_weights["industrial"],
            energy_weight=max_sharpe_group_weights["energy"],
        )

        min_vol_metrics = MetalsPortfolioMetrics(
            weights=min_vol_weights_series,
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, min_vol_weights),
            hhi=self.risk_calculator.calculate_hhi(min_vol_weights),
            n_assets=len(
                min_vol_weights[min_vol_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
            precious_metals_weight=min_vol_group_weights["precious"],
            industrial_metals_weight=min_vol_group_weights["industrial"],
            energy_weight=min_vol_group_weights["energy"],
        )

        # Добавляем VaR и CVaR
        for conf_level in VAR_CONFIDENCE_LEVELS:
            var, cvar = self.risk_calculator.calculate_var_cvar(
                self.returns, max_sharpe_weights, conf_level
            )
            var2, cvar2 = self.risk_calculator.calculate_var_cvar(
                self.returns, min_vol_weights, conf_level
            )

            if conf_level == 0.95:
                max_sharpe_metrics.var_95 = var
                max_sharpe_metrics.cvar_95 = cvar
                min_vol_metrics.var_95 = var2
                min_vol_metrics.cvar_95 = cvar2
            else:
                max_sharpe_metrics.var_99 = var
                max_sharpe_metrics.cvar_99 = cvar
                min_vol_metrics.var_99 = var2
                min_vol_metrics.cvar_99 = cvar2

        # 10. Визуализация результатов
        print("\n10. Визуализация результатов...")

        max_sharpe_portfolio_dict = {
            "return": max_sharpe_ret,
            "volatility": max_sharpe_vol,
            "weights": max_sharpe_metrics.weights,
        }

        min_vol_portfolio_dict = {
            "return": min_vol_ret,
            "volatility": min_vol_vol,
            "weights": min_vol_metrics.weights,
        }

        self.visualizer.plot_efficient_frontier(
            self.returns,
            efficient_portfolios,
            max_sharpe_portfolio_dict,
            min_vol_portfolio_dict,
            simulated_results,
            self.show_plots,
        )

        self.visualizer.plot_weights_comparison(
            max_sharpe_metrics.weights,
            min_vol_metrics.weights,
            self.returns,
            self.show_plots,
        )

        self.visualizer.plot_group_allocation(
            max_sharpe_metrics, min_vol_metrics, self.show_plots
        )

        # 11. Вывод результатов
        self.reporter.print_header("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ПОРТФЕЛЯ МЕТАЛЛОВ")
        self.reporter.print_metal_statistics(self.stats)
        self.reporter.print_portfolio_metrics(
            "Оптимальный портфель (Максимальный Sharpe Ratio)", max_sharpe_metrics
        )
        self.reporter.print_portfolio_metrics(
            "Портфель с минимальной волатильностью", min_vol_metrics
        )

        # 12. Сохранение в Excel
        print("\n11. Сохранение результатов в Excel...")
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.correlation_matrix,
            self.stats,
            skew_kurt,
            max_sharpe_metrics,
            min_vol_metrics,
            efficient_portfolios,
            self.returns,
            group_correlations,
        )

        print(f"\n   Все результаты сохранены в: {self.results_dir}")

        return {
            "prices": self.prices,
            "returns": self.returns,
            "stats": self.stats,
            "correlation_matrix": self.correlation_matrix,
            "max_sharpe_portfolio": max_sharpe_portfolio_dict,
            "min_vol_portfolio": min_vol_portfolio_dict,
            "efficient_portfolios": efficient_portfolios,
            "max_sharpe_metrics": max_sharpe_metrics,
            "min_vol_metrics": min_vol_metrics,
        }


def main(show_plots: bool = True):
    """Основная функция для запуска анализа металлов"""
    print("\n" + "=" * 80)
    print("ЗАПУСК АНАЛИЗА РЫНКА МЕТАЛЛОВ И СЫРЬЕВЫХ ТОВАРОВ")
    print("=" * 80 + "\n")
    print(f"Директория с исходными данными: {MARKET_DATA_DIR}")
    print(f"Директория для сохранения результатов: {RESULTS_DIR}")
    print("\n" + "=" * 80 + "\n")

    pipeline = MetalsAnalysisPipeline(show_plots=show_plots)
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("АНАЛИЗ МЕТАЛЛОВ УСПЕШНО ЗАВЕРШЕН")
    print("=" * 80)
    print(f"\nСозданные файлы в {RESULTS_DIR}:")
    print(f"   1. {OUTPUT_FILES['correlation_matrix']}.png - Корреляционная матрица")
    print(f"   2. {OUTPUT_FILES['prices_chart']}.png - Нормализованные цены")
    print(
        f"   3. {OUTPUT_FILES['returns_distribution']}.png - Распределение доходностей"
    )
    print(f"   4. {OUTPUT_FILES['efficient_frontier']}.png - Эффективная граница")
    print(f"   5. {OUTPUT_FILES['weights_comparison']}.png - Сравнение весов")
    print(f"   6. metals_group_allocation.png - Распределение по группам")
    print(f"   7. {OUTPUT_FILES['full_report']}.xlsx - Полный отчет Excel")

    return results


if __name__ == "__main__":
    results = main(show_plots=True)
