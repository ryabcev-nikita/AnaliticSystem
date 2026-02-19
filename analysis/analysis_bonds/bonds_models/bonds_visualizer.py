import os
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from bonds_models.bonds_constants import (
    COLORS,
    OUTPUT_FILES,
    PLOT_STYLE,
    RISK_FREE_RATE,
    RISK_LEVELS,
    SECTORS,
)
from bonds_models.bonds_models import Bond


class BondsPortfolioVisualizer:
    """Класс для визуализации результатов"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def plot_yield_curve(self, bonds_df: pd.DataFrame, show_plot: bool = True):
        """Построение кривой доходности"""
        plt.figure(figsize=PLOT_STYLE["figure_size"]["medium"])

        for risk_level in [0, 1, 2, 3]:
            mask = bonds_df["risk_level"] == risk_level
            subset = bonds_df[mask]
            if len(subset) > 0:
                plt.scatter(
                    subset["years_to_maturity"],
                    subset["current_yield"],
                    label=RISK_LEVELS[risk_level]["name"],
                    color=COLORS[f"risk_{risk_level}"],
                    alpha=0.6,
                    s=50,
                )

        plt.xlabel("Years to Maturity", fontsize=PLOT_STYLE["label_fontsize"])
        plt.ylabel("Current Yield (%)", fontsize=PLOT_STYLE["label_fontsize"])
        plt.title("Yield Curve by Risk Level", fontsize=PLOT_STYLE["title_fontsize"])
        plt.legend(fontsize=PLOT_STYLE["legend_fontsize"])
        plt.grid(True, alpha=0.3)

        # Добавляем линию безрисковой ставки
        plt.axhline(
            y=RISK_FREE_RATE * 100,
            color="red",
            linestyle="--",
            label=f"Risk-Free Rate ({RISK_FREE_RATE*100:.1f}%)",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["yield_curve"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_sector_allocation(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение распределения по секторам"""
        sector_weights = {}
        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                sector = bond.sector
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Сортируем по весу
        sector_weights = dict(
            sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        )

        plt.figure(figsize=PLOT_STYLE["figure_size"]["small"])

        colors = [COLORS.get(s, "#999999") for s in sector_weights.keys()]
        labels = [SECTORS.get(s, {}).get("name", s) for s in sector_weights.keys()]

        wedges, texts, autotexts = plt.pie(
            sector_weights.values(),
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )

        plt.title(
            "Portfolio Sector Allocation", fontsize=PLOT_STYLE["title_fontsize"], pad=20
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["sector_allocation"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_currency_allocation(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение распределения по валютам"""
        currency_weights = {}
        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                currency = bond.currency
                currency_weights[currency] = currency_weights.get(currency, 0) + weight

        plt.figure(figsize=PLOT_STYLE["figure_size"]["small"])

        colors = [COLORS.get(c, "#999999") for c in currency_weights.keys()]

        wedges, texts, autotexts = plt.pie(
            currency_weights.values(),
            labels=[c.upper() for c in currency_weights.keys()],
            autopct="%1.1f%%",
            colors=colors,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )

        plt.title(
            "Portfolio Currency Allocation",
            fontsize=PLOT_STYLE["title_fontsize"],
            pad=20,
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.results_dir, OUTPUT_FILES["currency_allocation"] + ".png"
            ),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_maturity_profile(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение профиля погашений"""
        maturities = []
        bond_weights = []

        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                maturities.append(bond.years_to_maturity)
                bond_weights.append(weight * 100)

        plt.figure(figsize=PLOT_STYLE["figure_size"]["medium"])

        plt.bar(range(len(maturities)), bond_weights, color="steelblue", alpha=0.7)
        plt.xlabel("Bonds", fontsize=PLOT_STYLE["label_fontsize"])
        plt.ylabel("Weight (%)", fontsize=PLOT_STYLE["label_fontsize"])
        plt.title("Portfolio Maturity Profile", fontsize=PLOT_STYLE["title_fontsize"])
        plt.xticks(
            range(len(maturities)),
            [f"{m:.1f}y" for m in maturities],
            rotation=45,
            ha="right",
        )
        plt.grid(True, alpha=0.3, axis="y")

        # Добавляем среднюю дюрацию
        avg_duration = np.average(maturities, weights=bond_weights)
        plt.axhline(
            y=avg_duration,
            color="red",
            linestyle="--",
            label=f"Average Duration: {avg_duration:.2f}y",
        )
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["maturity_profile"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_risk_analysis(self, portfolio_stats: Dict, show_plot: bool = True):
        """Построение анализа рисков"""
        fig, axes = plt.subplots(1, 2, figsize=PLOT_STYLE["figure_size"]["medium"])

        # Радарная диаграмма (упрощенно - столбцы)
        metrics = ["Yield", "Duration", "Diversification", "Risk Score", "Convexity"]
        values = [
            portfolio_stats["yield"] * 100,
            portfolio_stats["duration"],
            portfolio_stats["diversification"] * 100,
            10 - portfolio_stats["risk_score"],  # Инвертируем для наглядности
            portfolio_stats["convexity"] * 10,
        ]

        ax1 = axes[0]
        x_pos = np.arange(len(metrics))
        ax1.bar(
            x_pos, values, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics, rotation=45, ha="right")
        ax1.set_ylabel("Score")
        ax1.set_title("Portfolio Risk Metrics")
        ax1.grid(True, alpha=0.3, axis="y")

        # Круговая диаграмма распределения риска
        ax2 = axes[1]
        risk_labels = ["Yield", "Duration", "Concentration", "Credit"]
        risk_values = [
            portfolio_stats["yield"] * 30,
            portfolio_stats["duration"] * 5,
            (1 - portfolio_stats["diversification"]) * 50,
            portfolio_stats["risk_score"] * 5,
        ]

        ax2.pie(
            risk_values,
            labels=risk_labels,
            autopct="%1.1f%%",
            colors=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
        )
        ax2.set_title("Risk Distribution")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["risk_analysis"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()
