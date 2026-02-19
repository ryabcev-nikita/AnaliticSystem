from datetime import datetime
import os

import pandas as pd

from indexes_models.indexes_constants import PortfolioMetrics
from indexes_models.indexes_constants import (
    END_DATE,
    INDEX_NAMES,
    OUTPUT_FILES,
    RISK_FREE_RATE,
    START_DATE,
    TRADING_DAYS,
    WEIGHT_THRESHOLD,
)


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
