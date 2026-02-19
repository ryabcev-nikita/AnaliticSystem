import os
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from indexes_models.indexes_constants import (
    COLORS,
    INDEX_NAMES,
    OUTPUT_FILES,
    PLOT_STYLE,
    TRADING_DAYS,
    WEIGHT_THRESHOLD,
)


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
