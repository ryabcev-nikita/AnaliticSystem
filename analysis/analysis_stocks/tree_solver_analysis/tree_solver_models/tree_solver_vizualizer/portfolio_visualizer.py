# ==================== КЛАСС ВИЗУАЛИЗАТОРА ====================
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ...tree_solver_models.tree_solver_portfolio.portfolio_manager import (
    PortfolioMetrics,
)
from ...tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FILE_CONSTANTS,
    FORMATTING,
    PORTFOLIO_CONSTANTS,
)
from ...tree_solver_models.tree_solver_market.market_benchmarks import (
    MarketBenchmarks,
)


class PortfolioVisualizer:
    """Визуализация портфеля и результатов анализа"""

    @staticmethod
    def plot_portfolio_summary(
        portfolio_df: pd.DataFrame,
        weights: np.ndarray,
        metrics: PortfolioMetrics,
        benchmarks: MarketBenchmarks,
        filename: str = None,
    ):
        """Сводная визуализация портфеля"""
        if filename is None:
            filename = PATHS["optimal_portfolio"]

        fig, axes = plt.subplots(2, 2, figsize=FILE_CONSTANTS.FIGURE_SIZE_SUMMARY)

        plot_df = portfolio_df.copy()
        plot_df["weights"] = weights
        plot_df = plot_df.reset_index(drop=True)

        n_positions = len(plot_df)

        # 1. Pie chart - распределение весов - ИСПРАВЛЕНО
        top_n = min(PORTFOLIO_CONSTANTS.TOP_PIE_N, len(plot_df))
        if top_n > 0:
            top_indices = np.argsort(weights)[::-1][:top_n]
            top_weights = weights[top_indices]
            top_tickers = plot_df.iloc[top_indices]["Тикер"].values

            other_weight = max(0, 1 - top_weights.sum())
            if other_weight > PORTFOLIO_CONSTANTS.MIN_WEIGHT and len(top_weights) < len(
                weights
            ):
                plot_weights = np.append(top_weights, other_weight)
                plot_labels = np.append(top_tickers, ["Другие"])
            else:
                plot_weights = top_weights
                plot_labels = top_tickers
                if abs(1 - plot_weights.sum()) > 0.01:
                    plot_weights = plot_weights / plot_weights.sum()

            # ИСПРАВЛЕНО: используем MATPLOTLIB_PERCENT для autopct
            axes[0, 0].pie(
                plot_weights,
                labels=plot_labels,
                autopct=FORMATTING.MATPLOTLIB_PERCENT,  # '%1.1f%%'
                startangle=90,
                colors=plt.cm.get_cmap(FORMATTING.COLOR_PIE_CMAP)(
                    range(len(plot_weights))
                ),
            )
            axes[0, 0].set_title(
                f"Топ-{top_n} позиций в портфеле",
                fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )

        # 2. Risk-Return scatter
        axes[0, 1].scatter(
            plot_df["Риск"],
            plot_df["Ожидаемая_доходность"],
            s=weights * 3000,
            alpha=0.6,
            c=FORMATTING.COLOR_PORTFOLIO_MARKER,
            edgecolors="black",
            linewidths=0.5,
        )

        for idx, row in plot_df.iterrows():
            if (
                idx < len(weights)
                and weights[idx] > PORTFOLIO_CONSTANTS.ANNOTATION_WEIGHT_THRESHOLD
            ):
                axes[0, 1].annotate(
                    row["Тикер"],
                    (row["Риск"], row["Ожидаемая_доходность"]),
                    fontsize=FORMATTING.ANNOTATION_FONT_SIZE,
                    alpha=0.8,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        axes[0, 1].axhline(
            y=metrics.expected_return,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Портфель: {metrics.expected_return:.1%}",
        )
        axes[0, 1].axvline(x=metrics.risk, color="r", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel(
            "Риск (волатильность)", fontsize=FORMATTING.AXIS_FONT_SIZE
        )
        axes[0, 1].set_ylabel(
            "Ожидаемая доходность", fontsize=FORMATTING.AXIS_FONT_SIZE
        )
        axes[0, 1].set_title(
            "Risk-Return профиль",
            fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # 3. Секторальное распределение - ИСПРАВЛЕНО
        if len(plot_df) > 0 and "weights" in plot_df.columns:
            sector_weights = plot_df.groupby("Сектор")["weights"].sum()
            if len(sector_weights) > 0:
                sector_weights = sector_weights.sort_values(ascending=False)
                colors = plt.cm.get_cmap(FORMATTING.COLOR_SECTOR_CMAP)(
                    np.linspace(0, 1, len(sector_weights))
                )
                # ИСПРАВЛЕНО: используем MATPLOTLIB_PERCENT для autopct
                axes[1, 0].pie(
                    sector_weights.values,
                    labels=sector_weights.index,
                    autopct=FORMATTING.MATPLOTLIB_PERCENT,  # '%1.1f%%'
                    startangle=90,
                    colors=colors,
                )
                axes[1, 0].set_title(
                    "Распределение по секторам",
                    fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
                    fontweight="bold",
                )

        # 4. Метрики портфеля
        axes[1, 1].axis("off")

        min_weight_val = (
            weights[weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT].min()
            if any(weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT)
            else 0
        )

        metrics_text = PortfolioVisualizer._format_metrics_text(
            metrics, benchmarks, n_positions, weights, min_weight_val
        )

        axes[1, 1].text(
            0.05,
            0.5,
            metrics_text,
            transform=axes[1, 1].transAxes,
            fontsize=FORMATTING.LABEL_FONT_SIZE,
            verticalalignment="center",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=FORMATTING.COLOR_PORTFOLIO_BG,
                edgecolor=FORMATTING.COLOR_PORTFOLIO_MARKER,
                alpha=0.9,
            ),
        )
        axes[1, 1].set_title(
            "Сводка портфеля", fontsize=FORMATTING.SUBTITLE_FONT_SIZE, fontweight="bold"
        )

        plt.suptitle(
            "Оптимальный инвестиционный портфель",
            fontsize=FORMATTING.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _format_metrics_text(
        metrics: PortfolioMetrics,
        benchmarks: MarketBenchmarks,
        n_positions: int,
        weights: np.ndarray,
        min_weight: float,
    ) -> str:
        """Форматирование текста с метриками портфеля"""

        div_yield_str = (
            FORMATTING.PERCENT_FORMAT.format(benchmarks.div_yield_median / 100)
            if pd.notna(benchmarks.div_yield_median) and benchmarks.div_yield_median > 0
            else FORMATTING.NA_STRING
        )
        roe_str = (
            FORMATTING.PERCENT_FORMAT.format(benchmarks.roe_median / 100)
            if pd.notna(benchmarks.roe_median) and benchmarks.roe_median > 0
            else FORMATTING.NA_STRING
        )
        pe_str = (
            FORMATTING.FLOAT_FORMAT_1D.format(benchmarks.pe_median)
            if pd.notna(benchmarks.pe_median) and benchmarks.pe_median > 0
            else FORMATTING.NA_STRING
        )
        pb_str = (
            FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.pb_median)
            if pd.notna(benchmarks.pb_median) and benchmarks.pb_median > 0
            else FORMATTING.NA_STRING
        )

        return (
            "\n        📊 МЕТРИКИ ПОРТФЕЛЯ\n        \n"
            f"        Ожидаемая доходность: {FORMATTING.PERCENT_FORMAT.format(metrics.expected_return)}\n"
            f"        Риск (волатильность): {FORMATTING.PERCENT_FORMAT.format(metrics.risk)}\n"
            f"        Коэффициент Шарпа: {FORMATTING.FLOAT_FORMAT_2D.format(metrics.sharpe_ratio)}\n"
            f"        Индекс диверсификации: {FORMATTING.PERCENT_FORMAT.format(metrics.diversification_score)}\n"
            "        \n"
            "        📈 СОСТАВ\n"
            f"        Количество позиций: {n_positions}\n"
            f"        Максимальная доля: {FORMATTING.PERCENT_FORMAT.format(weights.max())}\n"
            f"        Минимальная доля: {FORMATTING.PERCENT_FORMAT.format(min_weight)}\n"
            "        \n"
            "        📉 РЫНОЧНЫЕ БЕНЧМАРКИ\n"
            f"        Медианный P/E: {pe_str}\n"
            f"        Медианный P/B: {pb_str}\n"
            f"        Медианный ROE: {roe_str}\n"
            f"        Мед. див. доходность: {div_yield_str}\n"
        )

    @staticmethod
    def plot_efficient_frontier(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        optimal_weights: np.ndarray,
        optimal_return: float,
        optimal_risk: float,
        filename: str = None,
    ):
        """Построение границы эффективности"""
        if filename is None:
            filename = PATHS["efficient_frontier"]

        n_assets = len(expected_returns)

        if n_assets < 2:
            print("⚠️ Недостаточно активов для построения границы эффективности")
            return

        n_portfolios = PORTFOLIO_CONSTANTS.N_EFFICIENT_PORTFOLIOS
        returns = []
        risks = []
        sharpe_ratios = []

        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()

            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            returns.append(port_return)
            risks.append(port_risk)
            sharpe_ratios.append(port_return / port_risk if port_risk > 0 else 0)

        plt.figure(figsize=FILE_CONSTANTS.FIGURE_SIZE_FRONTIER)

        scatter = plt.scatter(
            risks,
            returns,
            c=sharpe_ratios,
            cmap=FORMATTING.COLOR_RISK_RETURN_CMAP,
            alpha=0.3,
            s=15,
        )
        plt.colorbar(scatter, label="Коэффициент Шарпа")

        plt.scatter(
            optimal_risk,
            optimal_return,
            c=FORMATTING.COLOR_OPTIMAL_MARKER,
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Оптимальный портфель",
        )

        plt.xlabel("Риск (волатильность)", fontsize=FORMATTING.AXIS_FONT_SIZE)
        plt.ylabel("Ожидаемая доходность", fontsize=FORMATTING.AXIS_FONT_SIZE)
        plt.title(
            "Граница эффективности Марковица",
            fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()
