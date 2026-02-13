# portfolio_optimization.py
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

from analysis_indexes_constants import (
    START_DATE,
    END_DATE,
    MARKET_DATA_DIR,
    RESULTS_DIR,
    MARKET_DATA_FILES,
    INDEX_NAMES,
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
)


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


class DataLoader:
    """Класс для загрузки и предобработки данных"""

    @staticmethod
    def load_market_data() -> pd.DataFrame:
        """Загрузка данных из CSV файлов"""
        print(f"   Загрузка данных из: {MARKET_DATA_DIR}")
        data = {}

        for name, file in MARKET_DATA_FILES.items():
            file_path = os.path.join(MARKET_DATA_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            data[name] = df["Close"]

        prices = pd.DataFrame(data)

        # Фильтруем по датам
        mask = (prices.index >= START_DATE) & (prices.index <= END_DATE)
        prices = prices[mask]

        return prices


class StatisticsCalculator:
    """Класс для расчета статистик"""

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Расчет логарифмических доходностей"""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def calculate_annual_statistics(returns: pd.DataFrame) -> pd.DataFrame:
        """Расчет годовых статистик"""
        annual_returns = returns.mean() * TRADING_DAYS
        annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe_ratios = (annual_returns - RISK_FREE_RATE) / annual_volatility

        return pd.DataFrame(
            {
                "Annual Return": annual_returns,
                "Annual Volatility": annual_volatility,
                "Sharpe Ratio": sharpe_ratios,
            }
        )

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Расчет максимальной просадки для одного индекса"""
        wealth_index = prices / prices.iloc[0]
        previous_peaks = wealth_index.expanding().max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    @staticmethod
    def calculate_all_max_drawdowns(prices: pd.DataFrame) -> Dict[str, float]:
        """Расчет максимальных просадок для всех индексов"""
        return {
            col: StatisticsCalculator.calculate_max_drawdown(prices[col])
            for col in prices.columns
        }


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


class PortfolioVisualizer:
    """Класс для визуализации результатов"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

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

        # Отдельные активы
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

            # Укорачиваем название для аннотации
            label = (
                INDEX_NAMES[idx].split()[0]
                if " " in INDEX_NAMES[idx]
                else INDEX_NAMES[idx][:10]
            )
            plt.annotate(
                label,
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
            "Markowitz Portfolio Optimization - Efficient Frontier",
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
            ax1.pie(
                max_sharpe_nonzero.values,
                labels=[INDEX_NAMES[x] for x in max_sharpe_nonzero.index],
                autopct="%1.1f%%",
                colors=colors_max,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )
        else:
            ax1.text(0.5, 0.5, "No significant weights", ha="center", va="center")
        ax1.set_title("Maximum Sharpe Ratio Portfolio", fontsize=14, pad=20)

        # Минимальная волатильность
        if len(min_vol_nonzero) > 0:
            colors_min = [COLORS.get(idx, "#333333") for idx in min_vol_nonzero.index]
            ax2.pie(
                min_vol_nonzero.values,
                labels=[INDEX_NAMES[x] for x in min_vol_nonzero.index],
                autopct="%1.1f%%",
                colors=colors_min,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )
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


class ExcelReportGenerator:
    """Класс для генерации Excel отчетов"""

    @staticmethod
    def save_to_excel(
        results_dir: str,
        correlation_matrix: pd.DataFrame,
        stats: pd.DataFrame,
        max_sharpe_metrics: PortfolioMetrics,
        min_vol_metrics: PortfolioMetrics,
        efficient_portfolios: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> None:
        """Сохранение всех результатов в Excel файл"""

        file_path = os.path.join(results_dir, f"{OUTPUT_FILES['full_report']}.xlsx")

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

            # 1. Статистика по индексам
            stats.to_excel(writer, sheet_name="Index Statistics", startrow=1)
            worksheet = writer.sheets["Index Statistics"]
            worksheet.cell(row=1, column=1, value="Годовая статистика по индексам")

            # 2. Корреляционная матрица
            correlation_matrix.to_excel(
                writer, sheet_name="Correlation Matrix", startrow=1
            )
            worksheet = writer.sheets["Correlation Matrix"]
            worksheet.cell(row=1, column=1, value="Корреляционная матрица доходностей")

            # 3. Веса портфелей
            weights_df = pd.DataFrame(
                {
                    "Index": max_sharpe_metrics.weights.index,
                    "Max Sharpe Weights": max_sharpe_metrics.weights.values,
                    "Min Volatility Weights": min_vol_metrics.weights.values,
                }
            )
            weights_df.to_excel(
                writer, sheet_name="Portfolio Weights", index=False, startrow=1
            )
            worksheet = writer.sheets["Portfolio Weights"]
            worksheet.cell(
                row=1, column=1, value="Веса активов в оптимальных портфелях"
            )

            # 4. Метрики портфелей
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
                ]
            ]
            metrics_df.to_excel(
                writer, sheet_name="Portfolio Metrics", index=False, startrow=1
            )
            worksheet = writer.sheets["Portfolio Metrics"]
            worksheet.cell(row=1, column=1, value="Метрики оптимальных портфелей")

            # 5. Эффективная граница
            ef_display = efficient_portfolios[["return", "volatility"]].copy()
            ef_display.columns = ["Expected Return", "Volatility"]
            ef_display.to_excel(
                writer, sheet_name="Efficient Frontier", index=False, startrow=1
            )
            worksheet = writer.sheets["Efficient Frontier"]
            worksheet.cell(row=1, column=1, value="Точки эффективной границы")

            # 6. Доходности для дополнительного анализа
            returns.describe().to_excel(
                writer, sheet_name="Returns Statistics", startrow=1
            )
            worksheet = writer.sheets["Returns Statistics"]
            worksheet.cell(row=1, column=1, value="Статистика дневных доходностей")

            # 7. Информация о периоде
            info_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Start Date",
                        "End Date",
                        "Risk Free Rate",
                        "Trading Days",
                        "Number of Assets",
                        "Optimization Date",
                    ],
                    "Value": [
                        START_DATE,
                        END_DATE,
                        f"{RISK_FREE_RATE:.1%}",
                        TRADING_DAYS,
                        len(max_sharpe_metrics.weights),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                }
            )
            info_df.to_excel(
                writer, sheet_name="Analysis Info", index=False, startrow=1
            )
            worksheet = writer.sheets["Analysis Info"]
            worksheet.cell(row=1, column=1, value="Параметры анализа")

        print(f"\n   Excel отчет сохранен: {file_path}")


class PortfolioAnalysisReport:
    """Класс для генерации отчетов в консоли"""

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
    def print_portfolio_metrics(name: str, metrics: PortfolioMetrics) -> None:
        """Печать метрик портфеля"""
        PortfolioAnalysisReport.print_section(name)

        # Веса
        significant_weights = metrics.weights[metrics.weights > WEIGHT_THRESHOLD]
        for idx, weight in significant_weights.sort_values(ascending=False).items():
            print(f"   {INDEX_NAMES[idx]:<35} {weight:.4f} ({weight*100:.1f}%)")

        # Метрики
        print(
            f"\n   Ожидаемая годовая доходность: {metrics.expected_return:.4f} "
            f"({metrics.expected_return*100:.1f}%)"
        )
        print(
            f"   Ожидаемая годовая волатильность: {metrics.volatility:.4f} "
            f"({metrics.volatility*100:.1f}%)"
        )
        print(f"   Коэффициент Шарпа: {metrics.sharpe_ratio:.4f}")
        print(f"   Бета коэффициент: {metrics.beta:.4f}")
        print(f"   VaR (95%): {metrics.var_95:.4f}, CVaR (95%): {metrics.cvar_95:.4f}")
        print(f"   VaR (99%): {metrics.var_99:.4f}, CVaR (99%): {metrics.cvar_99:.4f}")
        print(f"   Индекс Херфиндаля-Хиршмана: {metrics.hhi:.4f}")
        print(f"   Количество активов (вес > 1%): {metrics.n_assets}")


class PortfolioAnalysisPipeline:
    """Основной класс для запуска полного анализа"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_loader = DataLoader()
        self.stats_calculator = StatisticsCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_calculator = RiskMetricsCalculator()
        self.visualizer = PortfolioVisualizer(self.results_dir)
        self.reporter = PortfolioAnalysisReport()
        self.excel_reporter = ExcelReportGenerator()

        self.prices = None
        self.returns = None
        self.stats = None
        self.correlation_matrix = None
        self.optimizer = None

    def run(self) -> Dict[str, Any]:
        """Запуск полного цикла анализа"""

        self.reporter.print_header(
            "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ПО МАРКОВИЦУ"
        )

        # 1. Загрузка данных
        print("\n1. Загрузка данных...")
        self.prices = self.data_loader.load_market_data()
        print(f"   Загружено {len(self.prices.columns)} индексов")
        print(
            f"   Период: с {self.prices.index[0].date()} по {self.prices.index[-1].date()}"
        )

        # 2. Расчет доходностей и статистик
        print("\n2. Расчет статистик...")
        self.returns = self.stats_calculator.calculate_returns(self.prices)
        self.stats = self.stats_calculator.calculate_annual_statistics(self.returns)

        # Добавляем максимальную просадку
        max_drawdowns = self.stats_calculator.calculate_all_max_drawdowns(self.prices)
        self.stats["Max Drawdown"] = pd.Series(max_drawdowns)

        print("\n   Статистика по индексам (годовая):")
        print(self.stats.round(4))

        # 3. Корреляционный анализ
        print("\n3. Корреляционный анализ...")
        self.correlation_matrix = self.correlation_analyzer.analyze(self.returns)

        # Сохраняем график
        corr_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['correlation_matrix']}.png"
        )
        self.correlation_analyzer.plot(self.returns, str(corr_path), self.show_plots)

        # Вывод наиболее коррелированных пар
        print("\n   Наиболее коррелированные пары:")
        top_correlations = self.correlation_analyzer.get_top_correlations(5)
        for pair in top_correlations:
            print(
                f"   {INDEX_NAMES[pair['index1']]} - {INDEX_NAMES[pair['index2']]}: "
                f"{pair['correlation']:.4f}"
            )

        # 4. Оптимизация портфеля
        print("\n4. Оптимизация портфеля по Марковицу...")
        self.optimizer = PortfolioOptimizer(self.returns)

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

        # 5. Симуляция портфелей
        print("\n5. Симуляция случайных портфелей...")
        simulated_results, _ = self.optimizer.simulate_portfolios()

        # 6. Эффективная граница
        print("\n6. Построение эффективной границы...")
        efficient_portfolios = self.optimizer.efficient_frontier()

        # 7. Расчет метрик риска
        print("\n7. Расчет метрик риска...")

        # Создаем объекты с метриками для портфелей
        max_sharpe_metrics = PortfolioMetrics(
            weights=pd.Series(max_sharpe_weights, index=self.returns.columns),
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, max_sharpe_weights),
            hhi=self.risk_calculator.calculate_hhi(max_sharpe_weights),
            n_assets=len(
                max_sharpe_weights[max_sharpe_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
        )

        min_vol_metrics = PortfolioMetrics(
            weights=pd.Series(min_vol_weights, index=self.returns.columns),
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, min_vol_weights),
            hhi=self.risk_calculator.calculate_hhi(min_vol_weights),
            n_assets=len(
                min_vol_weights[min_vol_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
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

        # 8. Визуализация
        print("\n8. Визуализация результатов...")

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

        # 9. Вывод результатов
        self.reporter.print_header("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        self.reporter.print_portfolio_metrics(
            "Оптимальный портфель (Максимальный Sharpe Ratio)", max_sharpe_metrics
        )
        self.reporter.print_portfolio_metrics(
            "Портфель с минимальной волатильностью", min_vol_metrics
        )

        # 10. Сохранение в Excel
        print("\n10. Сохранение результатов в Excel...")
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.correlation_matrix,
            self.stats,
            max_sharpe_metrics,
            min_vol_metrics,
            efficient_portfolios,
            self.returns,
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
    """Основная функция для запуска анализа"""
    pipeline = PortfolioAnalysisPipeline(show_plots=show_plots)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    results = main(show_plots=True)
