# ==================== КЛАСС ВИЗУАЛИЗАЦИИ ====================


from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_FILES,
    AE_FORMAT,
    AE_REC,
    AE_THRESHOLD,
)
from ai_anomaly_detector_models.ai_anomaly_portfolio.ai_anomaly_portfolio_manager import (
    AEPortfolioManager,
)

from ai_anomaly_detector_models.ai_anomaly_loader.path_config import (
    AE_PATHS,
)


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
