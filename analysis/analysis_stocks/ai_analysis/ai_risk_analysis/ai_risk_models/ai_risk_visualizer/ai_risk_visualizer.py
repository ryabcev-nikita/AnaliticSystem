# ==================== КЛАСС ВИЗУАЛИЗАЦИИ ====================


from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ...ai_risk_models.ai_risk_constants.ai_risk_constants import (
    NN_FEATURE,
    NN_FILES,
    NN_FORMAT,
    RISK_CAT,
)
from ...ai_risk_models.ai_risk_loader.path_config import NN_RISK_PATHS

from ...ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_optimizer import (
    NNRiskPortfolioOptimizer,
)
from ...ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_manager import (
    NNRiskPortfolioManager,
)


class NNRiskPortfolioVisualizer:
    """Визуализация портфелей на основе нейросетевой оценки риска"""

    @staticmethod
    def plot_portfolio_summary(
        portfolio_manager: NNRiskPortfolioManager,
        filename: str = None,
    ):
        """Сводная визуализация портфеля"""
        if filename is None:
            filename = f"{NN_RISK_PATHS['nn_risk_portfolio_base']}_{portfolio_manager.name}.png"

        fig, axes = plt.subplots(2, 2, figsize=NN_FILES.FIGURE_SIZE_SUMMARY)

        NNRiskPortfolioVisualizer._plot_risk_allocation(portfolio_manager, axes[0, 0])
        NNRiskPortfolioVisualizer._plot_risk_return_scatter(
            portfolio_manager, axes[0, 1]
        )
        NNRiskPortfolioVisualizer._plot_risk_contribution(portfolio_manager, axes[1, 0])
        NNRiskPortfolioVisualizer._plot_portfolio_metrics(portfolio_manager, axes[1, 1])

        plt.suptitle(
            f"Портфель на основе нейросетевой оценки риска: {portfolio_manager.name}",
            fontsize=NN_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_allocation(pm: NNRiskPortfolioManager, ax):
        """Визуализация распределения по категориям риска"""
        risk_allocation = pm.get_risk_category_allocation()

        if len(risk_allocation) > 0:
            colors = {
                RISK_CAT.RISK_A_NAME: NN_FORMAT.COLOR_RISK_A,
                RISK_CAT.RISK_B_NAME: NN_FORMAT.COLOR_RISK_B,
                RISK_CAT.RISK_C_NAME: NN_FORMAT.COLOR_RISK_C,
                RISK_CAT.RISK_D_NAME: NN_FORMAT.COLOR_RISK_D,
            }

            plot_colors = [
                colors.get(cat, NN_FORMAT.COLOR_RISK_DEFAULT)
                for cat in risk_allocation.index
            ]

            ax.pie(
                risk_allocation.values,
                labels=risk_allocation.index,
                autopct=NN_FORMAT.MATPLOTLIB_PERCENT,
                startangle=90,
                colors=plot_colors,
                explode=[NN_FORMAT.PIE_EXPLODE_FACTOR] * len(risk_allocation),
            )
            ax.set_title(
                "Распределение по категориям риска",
                fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_risk_return_scatter(pm: NNRiskPortfolioManager, ax):
        """Визуализация риск-доходность позиций"""
        scatter = ax.scatter(
            pm.df["NN_Volatility"],
            pm.df["NN_Expected_Return"],
            s=pm.weights * 3000,
            c=pm.df["NN_Уверенность"],
            cmap=NN_FORMAT.COLOR_CONFIDENCE_CMAP,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        top_n = min(NN_FORMAT.TOP_POSITIONS_SUMMARY, len(pm.df))
        top_positions = pm.get_top_positions(top_n)

        for _, row in top_positions.iterrows():
            ax.annotate(
                row.get("Тикер", "N/A"),
                (row["NN_Volatility"], row["NN_Expected_Return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=NN_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.scatter(
            pm.metrics.risk,
            pm.metrics.expected_return,
            s=300,
            c=NN_FORMAT.COLOR_OPTIMAL_MARKER,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label=f"Портфель (Шарп: {pm.metrics.sharpe_ratio:.2f})",
        )

        ax.set_xlabel("Волатильность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Ожидаемая доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Risk-Return профиль позиций",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label="Уверенность нейросети")

    @staticmethod
    def _plot_risk_contribution(pm: NNRiskPortfolioManager, ax):
        """Визуализация вклада в риск портфеля"""
        risk_contrib = pm.get_risk_contribution()

        if len(risk_contrib) > 0:
            top_n = min(NN_FORMAT.TOP_RISK_CONTRIBUTION, len(risk_contrib))
            top_risk = risk_contrib.nlargest(top_n)

            colors = plt.cm.get_cmap(NN_FORMAT.COLOR_RISK_CONTRIBUTION_CMAP)(
                np.linspace(0.2, 0.8, len(top_risk))
            )
            bars = ax.barh(
                range(len(top_risk)), top_risk.values, color=colors, edgecolor="black"
            )

            ax.set_yticks(range(len(top_risk)))
            ax.set_yticklabels(top_risk.index)
            ax.set_xlabel("Вклад в риск портфеля", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
            ax.set_title(
                "Распределение риска по акциям",
                fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )
            ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="x")

            for bar, value in zip(bars, top_risk.values):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    ha="left",
                    va="center",
                    fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                )

    @staticmethod
    def _plot_portfolio_metrics(pm: NNRiskPortfolioManager, ax):
        """Визуализация метрик портфеля"""
        ax.axis("off")

        risk_alloc = pm.get_risk_category_allocation()
        risk_alloc_str = ""
        for cat, weight in risk_alloc.items():
            risk_alloc_str += f"{cat}: {weight:.1%}\n"

        metrics_text = f"""
        📊 МЕТРИКИ ПОРТФЕЛЯ: {pm.name}
        
        Ожидаемая доходность: {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.expected_return)}
        Риск (волатильность): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.risk)}
        Коэффициент Шарпа: {NN_FORMAT.FLOAT_FORMAT_2D.format(pm.metrics.sharpe_ratio)}
        
        📈 РИСК-МЕТРИКИ
        VaR (95%): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.var_95)}
        CVaR (95%): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.cvar_95)}
        
        📊 ДИВЕРСИФИКАЦИЯ
        Индекс диверсификации: {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.diversification_score)}
        Количество позиций: {len(pm.df)}
        Макс. доля: {NN_FORMAT.PERCENT_FORMAT.format(pm.weights.max())}
        
        🤖 НЕЙРОСЕТЕВАЯ ОЦЕНКА
        Средняя уверенность: {NN_FORMAT.PERCENT_FORMAT.format(pm.df['NN_Уверенность'].mean())}
        {risk_alloc_str}
        """

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=NN_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=NN_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=NN_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_portfolio_comparison(
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Сравнение различных портфелей"""
        if not portfolios:
            return

        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_portfolio_comparison"]

        fig, axes = plt.subplots(2, 2, figsize=NN_FILES.FIGURE_SIZE_COMPARISON)

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

        NNRiskPortfolioVisualizer._plot_risk_return_comparison(
            axes[0, 0], names, returns, risks, sharpes
        )
        NNRiskPortfolioVisualizer._plot_sharpe_comparison(axes[0, 1], names, sharpes)
        NNRiskPortfolioVisualizer._plot_var_comparison(axes[1, 0], names, var_95s)
        NNRiskPortfolioVisualizer._plot_best_portfolio_summary(axes[1, 1], portfolios)

        plt.suptitle(
            "Сравнение портфелей на основе нейросетевой оценки риска",
            fontsize=NN_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_return_comparison(ax, names, returns, risks, sharpes):
        """Визуализация сравнения риск-доходность"""
        scatter = ax.scatter(
            risks,
            returns,
            c=sharpes,
            s=300,
            cmap=NN_FORMAT.COLOR_CONFIDENCE_CMAP,
            edgecolors="black",
            linewidths=1.5,
        )

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=NN_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Риск", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение портфелей: Risk-Return",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Коэффициент Шарпа", fontsize=NN_FORMAT.LABEL_FONT_SIZE)

    @staticmethod
    def _plot_sharpe_comparison(ax, names, sharpes):
        """Визуализация сравнения коэффициентов Шарпа"""
        sharpe_array = np.array(sharpes)
        sharpe_range = sharpe_array.max() - sharpe_array.min() + 0.001

        colors = plt.cm.get_cmap(NN_FORMAT.COLOR_CONFIDENCE_CMAP)(
            (sharpe_array - sharpe_array.min()) / sharpe_range
        )

        bars = ax.bar(names, sharpes, color=colors, edgecolor="black")
        ax.axhline(
            y=1,
            color=NN_FORMAT.COLOR_OPTIMAL_MARKER,
            linestyle="--",
            alpha=0.5,
            label="Целевой Шарп = 1",
        )
        ax.set_xlabel("Портфель", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Коэффициент Шарпа", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение коэффициентов Шарпа",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="y")

        for bar, value in zip(bars, sharpes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_var_comparison(ax, names, var_95s):
        """Визуализация сравнения Value at Risk"""
        bars = ax.bar(names, var_95s, color=NN_FORMAT.COLOR_RISK_B, edgecolor="black")
        ax.set_xlabel("Портфель", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("VaR (95%)", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение Value at Risk",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="y")

        for bar, value in zip(bars, var_95s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.01,
                f"{value:.1%}",
                ha="center",
                va="top",
                fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
                color="white",
            )

    @staticmethod
    def _plot_best_portfolio_summary(ax, portfolios):
        """Визуализация информации о лучшем портфеле"""
        ax.axis("off")

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        top_n = min(NN_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        text = f"🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {best_portfolio.name}\n\n"
        text += f"Доходность: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}\n"
        text += (
            f"Риск: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}\n"
        )
        text += f"Шарп: {NN_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}\n"
        text += (
            f"VaR: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.var_95)}\n\n"
        )
        text += f"📈 ТОП-{top_n} ПОЗИЦИЙ:\n"

        for _, row in top_positions.iterrows():
            ticker = row.get("Тикер", "N/A")
            weight = row.get("Weight", 0)
            text += f"• {ticker}: {NN_FORMAT.PERCENT_FORMAT.format(weight)}\n"
            risk_cat = row.get("NN_Категория_текст", "N/A")
            if risk_cat != "N/A":
                text += f"  {risk_cat}\n"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=NN_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=NN_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=NN_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_efficient_frontier(
        optimizer: NNRiskPortfolioOptimizer,
        df: pd.DataFrame,
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Построение границы эффективности"""
        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_efficient_frontier"]

        expected_returns = df["NN_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(df)

        n_portfolios = NN_FILES.N_EFFICIENT_PORTFOLIOS
        n_assets = len(df)

        returns = []
        risks = []
        sharpes = []

        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()

            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            port_sharpe = (
                (port_return - optimizer.risk_free_rate) / port_risk
                if port_risk > 0
                else 0
            )

            returns.append(port_return)
            risks.append(port_risk)
            sharpes.append(port_sharpe)

        plt.figure(figsize=NN_FILES.FIGURE_SIZE_FRONTIER)

        scatter = plt.scatter(
            risks,
            returns,
            c=sharpes,
            cmap=NN_FORMAT.COLOR_EFFICIENT_CMAP,
            alpha=0.2,
            s=10,
        )

        plt.colorbar(scatter, label="Коэффициент Шарпа")

        colors = plt.cm.get_cmap(NN_FORMAT.COLOR_PORTFOLIO_CMAP)(
            np.linspace(0, 1, len(portfolios))
        )

        for (name, pm), color in zip(portfolios.items(), colors):
            plt.scatter(
                pm.metrics.risk,
                pm.metrics.expected_return,
                s=300,
                c=[color],
                marker="*",
                edgecolors="black",
                linewidths=2,
                label=f"{name} (Шарп: {pm.metrics.sharpe_ratio:.2f})",
            )

        market_weights = np.array([1 / n_assets] * n_assets)
        market_return = np.sum(expected_returns * market_weights)
        market_risk = np.sqrt(
            np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        )

        plt.scatter(
            market_risk,
            market_return,
            s=200,
            c=NN_FORMAT.COLOR_MARKET_MARKER,
            marker="s",
            edgecolors="black",
            linewidths=2,
            label="Равновзвешенный",
        )

        plt.xlabel("Риск (волатильность)", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        plt.ylabel("Ожидаемая доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        plt.title(
            "Граница эффективности Марковица",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)

        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()
