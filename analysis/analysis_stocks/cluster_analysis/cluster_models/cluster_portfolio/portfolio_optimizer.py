# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================


from dataclasses import dataclass
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from cluster_models.cluster_constants.cluster_constants import (
    CLUSTER,
    CLUSTER_FILES,
    CLUSTER_FORMAT,
    PORTFOLIO_CLUSTER,
)


@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""

    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float


class PortfolioOptimizer:
    """Оптимизация портфеля по Марковицу с учетом кластеров"""

    def __init__(self, min_weight: float = None, max_weight: float = None):
        self.min_weight = min_weight or PORTFOLIO_CLUSTER.MIN_WEIGHT
        self.max_weight = max_weight or PORTFOLIO_CLUSTER.MAX_WEIGHT

    def create_covariance_matrix(
        self,
        df: pd.DataFrame,
        intra_cluster_corr: float = None,
        inter_cluster_corr: float = None,
    ) -> np.ndarray:
        """Создание матрицы ковариации с учетом кластерной структуры"""
        intra_cluster_corr = (
            intra_cluster_corr or PORTFOLIO_CLUSTER.INTRA_CLUSTER_CORRELATION
        )
        inter_cluster_corr = (
            inter_cluster_corr or PORTFOLIO_CLUSTER.INTER_CLUSTER_CORRELATION
        )

        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["Risk"].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    correlation = (
                        intra_cluster_corr
                        if "Cluster" in df.columns
                        and df.iloc[i]["Cluster"] == df.iloc[j]["Cluster"]
                        else inter_cluster_corr
                    )
                    cov_matrix[i, j] = correlation * risks[i] * risks[j]

        return cov_matrix

    def optimize_multi_portfolio(
        self, df: pd.DataFrame, strategies: List[str] = None
    ) -> Dict:
        """Оптимизация нескольких портфелей для разных стратегий"""
        if strategies is None:
            strategies = list(PORTFOLIO_CLUSTER.DEFAULT_STRATEGIES)

        portfolios = {}
        strategy_map = {
            "aggressive": self._optimize_for_max_return,
            "conservative": self._optimize_for_min_risk,
            "balanced": self._optimize_balanced,
            "value": self._optimize_value_portfolio,
            "growth": self._optimize_growth_portfolio,
            "dividend": self._optimize_dividend_portfolio,
            "cluster_based": self._optimize_cluster_based,
        }

        name_map = {
            "aggressive": "Агрессивный",
            "conservative": "Консервативный",
            "balanced": "Сбалансированный",
            "value": "Стоимостной",
            "growth": "Роста",
            "dividend": "Дивидендный",
            "cluster_based": "Кластерный",
        }

        for strategy in strategies:
            try:
                if strategy in strategy_map:
                    weights = strategy_map[strategy](df)
                    portfolios[name_map[strategy]] = weights
            except Exception as e:
                print(f"   ⚠️ Ошибка для стратегии {strategy}: {e}")

        return portfolios

    def _optimize_for_max_return(self, df: pd.DataFrame) -> np.ndarray:
        """Оптимизация для максимальной доходности"""
        expected_returns = df["Expected_Return"].values
        weights = expected_returns / expected_returns.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_for_min_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Оптимизация для минимального риска"""
        risks = df["Risk"].values
        inv_risk = 1 / risks
        weights = inv_risk / inv_risk.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_balanced(self, df: pd.DataFrame) -> np.ndarray:
        """Сбалансированная оптимизация"""
        scores = (df["Value_Score"] + df["Quality_Score"]) / 2
        weights = scores / scores.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_value_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Стоимостной портфель"""
        weights = df["Value_Score"].values / df["Value_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_growth_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Портфель роста"""
        weights = df["Growth_Score"].values / df["Growth_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_dividend_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Дивидендный портфель"""
        weights = df["Income_Score"].values / df["Income_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_cluster_based(self, df: pd.DataFrame) -> np.ndarray:
        """Кластерный портфель - равномерное распределение по кластерам"""
        n = len(df)
        weights = np.zeros(n)
        unique_clusters = df["Cluster"].unique()
        n_clusters = len(unique_clusters)

        for cluster in unique_clusters:
            cluster_indices = df[df["Cluster"] == cluster].index
            cluster_weight = 1 / n_clusters
            per_stock_weight = cluster_weight / len(cluster_indices)
            weights[cluster_indices] = per_stock_weight

        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()


# ==================== КЛАСС ПОРТФЕЛЬНОГО МЕНЕДЖЕРА ====================


class PortfolioManager:
    """Управление портфелем и расчет метрик"""

    def __init__(self, name: str, df: pd.DataFrame, weights: np.ndarray):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        exp_return = np.sum(self.df["Expected_Return"] * self.weights)

        optimizer = PortfolioOptimizer()
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        sharpe = exp_return / risk if risk > 0 else 0

        hhi = np.sum(self.weights**2)
        n = len(self.weights)
        diversification = 1 - (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 0

        return PortfolioMetrics(
            expected_return=exp_return,
            risk=risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        if "Sector" in self.df.columns:
            return self.df.groupby("Sector")["Weight"].sum()
        return pd.Series()

    def get_cluster_allocation(self) -> pd.Series:
        """Распределение по кластерам"""
        if "Cluster" in self.df.columns:
            return self.df.groupby("Cluster")["Weight"].sum()
        return pd.Series()

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or PORTFOLIO_CLUSTER.TOP_POSITIONS_N
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()


# ==================== КЛАСС ВИЗУАЛИЗАТОРА ====================


class PortfolioVisualizer:
    """Визуализация портфелей и результатов"""

    @staticmethod
    def plot_portfolio_comparison(
        portfolios: Dict[str, PortfolioManager],
        filename: str = None,
    ):
        """Сравнение различных портфелей"""
        if not portfolios:
            print("   ⚠️ Нет портфелей для визуализации")
            return

        if filename is None:
            filename = CLUSTER_PATHS["portfolio_comparison"]

        n_portfolios = len(portfolios)
        fig, axes = plt.subplots(2, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_COMPARISON)

        names = []
        returns = []
        risks = []
        sharpe = []

        for name, pm in portfolios.items():
            names.append(name)
            returns.append(pm.metrics.expected_return)
            risks.append(pm.metrics.risk)
            sharpe.append(pm.metrics.sharpe_ratio)

        PortfolioVisualizer._plot_risk_return_comparison(
            axes[0, 0], names, returns, risks, sharpe
        )
        PortfolioVisualizer._plot_sharpe_comparison(axes[0, 1], names, sharpe)
        PortfolioVisualizer._plot_diversification_comparison(
            axes[1, 0], names, portfolios
        )
        PortfolioVisualizer._plot_best_portfolio_summary(axes[1, 1], portfolios)

        plt.suptitle(
            "Сравнение инвестиционных портфелей",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=CLUSTER_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_return_comparison(ax, names, returns, risks, sharpe):
        """Визуализация сравнения риск-доходность"""
        scatter = ax.scatter(
            risks,
            returns,
            c=sharpe,
            cmap=CLUSTER_FORMAT.COLOR_SHARPE_CMAP,
            s=200,
            edgecolors="black",
            linewidths=1.5,
        )

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=CLUSTER_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Риск", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Ожидаемая доходность", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение портфелей: Risk-Return",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Коэффициент Шарпа", fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE)

    @staticmethod
    def _plot_sharpe_comparison(ax, names, sharpe):
        """Визуализация сравнения коэффициентов Шарпа"""
        sharpe_array = np.array(sharpe)
        sharpe_min, sharpe_max = sharpe_array.min(), sharpe_array.max()
        sharpe_range = sharpe_max - sharpe_min + 0.001

        colors = plt.cm.get_cmap(CLUSTER_FORMAT.COLOR_SHARPE_CMAP)(
            (sharpe_array - sharpe_min) / sharpe_range
        )

        bars = ax.bar(names, sharpe, color=colors, edgecolor="black")
        ax.axhline(
            y=1,
            color=CLUSTER_FORMAT.COLOR_SHARPE_TARGET,
            linestyle="--",
            alpha=CLUSTER.SCATTER_ALPHA,
            label="Целевой Шарп = 1",
        )
        ax.set_xlabel("Портфель", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Коэффициент Шарпа", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение коэффициентов Шарпа",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA, axis="y")

        for bar, value in zip(bars, sharpe):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=CLUSTER_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_diversification_comparison(ax, names, portfolios):
        """Визуализация сравнения диверсификации"""
        diversifications = [
            pm.metrics.diversification_score for pm in portfolios.values()
        ]
        bars = ax.bar(
            names,
            diversifications,
            color=CLUSTER_FORMAT.COLOR_DIVERSIFICATION,
            edgecolor="black",
        )
        ax.set_xlabel("Портфель", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Индекс диверсификации", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Уровень диверсификации",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA, axis="y")

        for bar, value in zip(bars, diversifications):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=CLUSTER_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_best_portfolio_summary(ax, portfolios):
        """Визуализация информации о лучшем портфеле"""
        ax.axis("off")

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        top_n = min(PORTFOLIO_CLUSTER.TOP_POSITIONS_RECOMMEND, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        text = f"🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {best_portfolio.name}\n\n"
        text += f"Доходность: {best_portfolio.metrics.expected_return:.1%}\n"
        text += f"Риск: {best_portfolio.metrics.risk:.1%}\n"
        text += f"Шарп: {best_portfolio.metrics.sharpe_ratio:.2f}\n\n"
        text += f"ТОП-{top_n} ПОЗИЦИЙ:\n"

        for _, row in top_positions.iterrows():
            text += f"• {row['Ticker']}: {row['Weight']:.1%}\n"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_cluster_portfolio_allocation(
        portfolio_manager: PortfolioManager, filename: str = None
    ):
        """Визуализация распределения портфеля по кластерам"""
        if filename is None:
            filename = CLUSTER_PATHS["cluster_allocation"]

        fig, axes = plt.subplots(1, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_ALLOCATION)

        cluster_weights = portfolio_manager.get_cluster_allocation()

        if len(cluster_weights) == 0:
            print("   ⚠️ Нет данных о кластерах для визуализации")
            plt.close()
            return

        colors = plt.cm.get_cmap(CLUSTER_FORMAT.COLOR_CLUSTER_CMAP)(
            np.linspace(0, 1, len(cluster_weights))
        )

        explode = [CLUSTER_FORMAT.PIE_EXPLODE_FACTOR] * len(cluster_weights)

        axes[0].pie(
            cluster_weights.values,
            labels=[f"Кластер {int(i)}" for i in cluster_weights.index],
            autopct=CLUSTER_FORMAT.MATPLOTLIB_PERCENT,
            startangle=90,
            colors=colors,
            explode=explode,
        )
        axes[0].set_title(
            "Распределение по кластерам",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )

        axes[1].axis("off")

        text = PortfolioVisualizer._format_cluster_allocation_text(
            portfolio_manager, cluster_weights
        )

        axes[1].text(
            0.05,
            0.95,
            text,
            transform=axes[1].transAxes,
            fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

        plt.suptitle(
            f"Анализ портфеля: {portfolio_manager.name}",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=CLUSTER_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _format_cluster_allocation_text(
        pm: PortfolioManager, cluster_weights: pd.Series
    ) -> str:
        """Форматирование текста для распределения по кластерам"""
        text = f"📊 СОСТАВ ПОРТФЕЛЯ ПО КЛАСТЕРАМ: {pm.name}\n\n"

        for cluster_id in cluster_weights.index:
            cluster_data = pm.df[pm.df["Cluster"] == cluster_id]
            weight = cluster_weights[cluster_id]

            text += f"Кластер {cluster_id} - {weight:.1%}\n"
            text += f"  Компаний: {len(cluster_data)}\n"
            text += f"  Средний P/E: {cluster_data['PE'].mean():.1f}\n"
            text += f"  Средний ROE: {cluster_data['ROE'].mean():.1f}%\n"

            if len(cluster_data) > 0:
                top_n = min(PORTFOLIO_CLUSTER.TOP_IN_CLUSTER_N, len(cluster_data))
                top_in_cluster = cluster_data.nlargest(top_n, "Weight")
                text += f"  Топ: {top_in_cluster.iloc[0]['Ticker']} ({top_in_cluster.iloc[0]['Weight']:.1%})"
                if len(top_in_cluster) > 1:
                    text += f", {top_in_cluster.iloc[1]['Ticker']} ({top_in_cluster.iloc[1]['Weight']:.1%})"
            text += "\n\n"

        return text
