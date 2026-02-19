import os
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from commodities_models.commodities_constants import (
    COLORS,
    METAL_NAMES,
    OUTPUT_FILES,
    PLOT_STYLE,
    TRADING_DAYS,
    WEIGHT_THRESHOLD,
)
from commodities_models.commodities_metrics import (
    MetalsPortfolioMetrics,
)


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
