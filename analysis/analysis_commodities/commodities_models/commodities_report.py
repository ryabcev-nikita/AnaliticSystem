from datetime import datetime
import os

import pandas as pd

from commodities_models.commodities_constants import (
    BENCHMARK_INDEX,
    END_DATE,
    METAL_NAMES,
    OUTPUT_FILES,
    RISK_FREE_RATE,
    START_DATE,
    TRADING_DAYS,
    WEIGHT_THRESHOLD,
)
from commodities_models.commodities_metrics import MetalsPortfolioMetrics


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
