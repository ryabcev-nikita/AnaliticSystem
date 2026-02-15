# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================


from typing import List
import pandas as pd
from ...tree_solver_models.tree_solver_portfolio.portfolio_manager import (
    PortfolioManager,
)
from ...tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FILE_CONSTANTS,
    FORMATTING,
    PORTFOLIO_CONSTANTS,
    REPORT,
)
from ...tree_solver_models.tree_solver_market.market_benchmarks import MarketBenchmarks


class ReportGenerator:
    """Генерация отчетов и рекомендаций"""

    @staticmethod
    def generate_portfolio_report(
        portfolio_manager: PortfolioManager,
        benchmarks: MarketBenchmarks,
        filename: str = None,
    ):
        """Генерация Excel отчета"""
        if filename is None:
            filename = PATHS["portfolio_report"]

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Состав портфеля
                portfolio_df = portfolio_manager.df.copy()
                portfolio_df = portfolio_df.sort_values("weights", ascending=False)

                portfolio_display = portfolio_df[
                    [
                        "Тикер",
                        "Название",
                        "Сектор",
                        "weights",
                        "Ожидаемая_доходность",
                        "Риск",
                        "P/E",
                        "P/B",
                        "ROE",
                        "Дивидендная доходность",
                        "Predicted_Оценка_текст",
                        "Predicted_Уверенность",
                    ]
                ].copy()

                portfolio_display.columns = [
                    REPORT.COLUMN_TICKER,
                    REPORT.COLUMN_NAME,
                    REPORT.COLUMN_SECTOR,
                    REPORT.COLUMN_WEIGHT,
                    REPORT.COLUMN_EXPECTED_RETURN,
                    REPORT.COLUMN_RISK,
                    REPORT.COLUMN_PE,
                    REPORT.COLUMN_PB,
                    REPORT.COLUMN_ROE,
                    REPORT.COLUMN_DIV_YIELD,
                    REPORT.COLUMN_RATING,
                    REPORT.COLUMN_CONFIDENCE,
                ]

                portfolio_display.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_PORTFOLIO, index=False
                )

                # Секторальное распределение
                sector_weights = portfolio_manager.get_sector_allocation()
                if len(sector_weights) > 0:
                    sector_df = pd.DataFrame(
                        {"Сектор": sector_weights.index, "Доля": sector_weights.values}
                    )
                    sector_df.to_excel(
                        writer, sheet_name=FILE_CONSTANTS.SHEET_SECTORS, index=False
                    )

                # Метрики портфеля
                min_weight_val = (
                    portfolio_manager.weights[
                        portfolio_manager.weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT
                    ].min()
                    if any(portfolio_manager.weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT)
                    else 0
                )

                metrics_df = pd.DataFrame(
                    {
                        "Метрика": [
                            REPORT.METRIC_EXPECTED_RETURN,
                            REPORT.METRIC_RISK,
                            REPORT.METRIC_SHARPE,
                            REPORT.METRIC_DIVERSIFICATION,
                            REPORT.METRIC_N_POSITIONS,
                            REPORT.METRIC_MAX_WEIGHT,
                            REPORT.METRIC_MIN_WEIGHT,
                        ],
                        "Значение": [
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.expected_return
                            ),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.risk
                            ),
                            FORMATTING.FLOAT_FORMAT_2D.format(
                                portfolio_manager.metrics.sharpe_ratio
                            ),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.diversification_score
                            ),
                            len(portfolio_manager.df),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.weights.max()
                            ),
                            FORMATTING.PERCENT_FORMAT.format(min_weight_val),
                        ],
                    }
                )
                metrics_df.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_METRICS, index=False
                )

                # Рыночные бенчмарки
                benchmarks_df = pd.DataFrame(
                    {
                        "Мультипликатор": [
                            REPORT.BENCHMARK_PE,
                            REPORT.BENCHMARK_PB,
                            REPORT.BENCHMARK_PS,
                            REPORT.BENCHMARK_ROE,
                            REPORT.BENCHMARK_DIV_YIELD,
                            REPORT.BENCHMARK_DEBT,
                            REPORT.BENCHMARK_BETA,
                        ],
                        "Значение": ReportGenerator._format_benchmark_values(
                            benchmarks
                        ),
                    }
                )
                benchmarks_df.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_BENCHMARKS, index=False
                )

            print(f"   ✅ Отчет сохранен: {filename}")

        except Exception as e:
            print(f"   ❌ Ошибка при сохранении отчета: {e}")

    @staticmethod
    def _format_benchmark_values(benchmarks: MarketBenchmarks) -> List[str]:
        """Форматирование значений бенчмарков"""
        return [
            (
                FORMATTING.FLOAT_FORMAT_1D.format(benchmarks.pe_median)
                if pd.notna(benchmarks.pe_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.pb_median)
                if pd.notna(benchmarks.pb_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.ps_median)
                if pd.notna(benchmarks.ps_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.roe_median / 100)
                if pd.notna(benchmarks.roe_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.div_yield_median / 100)
                if pd.notna(benchmarks.div_yield_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.debt_capital_median / 100)
                if pd.notna(benchmarks.debt_capital_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.beta_median)
                if pd.notna(benchmarks.beta_median)
                else FORMATTING.NA_STRING
            ),
        ]

    @staticmethod
    def print_recommendations(portfolio_manager: PortfolioManager):
        """Вывод рекомендаций в консоль"""
        print(FORMATTING.SEPARATOR)
        print("🎯 КЛЮЧЕВЫЕ ИНВЕСТИЦИОННЫЕ РЕКОМЕНДАЦИИ")
        print(FORMATTING.SEPARATOR)

        # Топ-3 рекомендации
        top_n = min(
            PORTFOLIO_CONSTANTS.TOP_RECOMMENDATIONS_N, len(portfolio_manager.df)
        )
        top_positions = portfolio_manager.get_top_positions(top_n)

        print(f"\n🔹 ТОП-{top_n} АКЦИИ ДЛЯ ПОКУПКИ:")
        for _, row in top_positions.iterrows():
            print(f"   • {row['Тикер']} - {row['Название']}")
            print(
                f"     Доля: {FORMATTING.PERCENT_FORMAT.format(row['weights'])} | "
                f"Доходность: {FORMATTING.PERCENT_FORMAT.format(row['Ожидаемая_доходность'])} | "
                f"Риск: {FORMATTING.PERCENT_FORMAT.format(row['Риск'])}"
            )
            print(
                f"     Оценка: {row['Predicted_Оценка_текст']} "
                f"(уверенность: {FORMATTING.PERCENT_FORMAT.format(row['Predicted_Уверенность'])})"
            )

        print("\n🔸 ДИВЕРСИФИКАЦИЯ:")
        print(
            f"   • Индекс диверсификации: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.metrics.diversification_score)}"
        )
        sector_allocation = portfolio_manager.get_sector_allocation()
        print(f"   • Количество секторов: {len(sector_allocation)}")
        if len(sector_allocation) > 0:
            print(
                f"   • Максимальная доля сектора: {FORMATTING.PERCENT_FORMAT.format(sector_allocation.max())}"
            )

        print("\n🔹 РИСК-МЕНЕДЖМЕНТ:")
        print(
            f"   • Коэффициент Шарпа: {FORMATTING.FLOAT_FORMAT_2D.format(portfolio_manager.metrics.sharpe_ratio)} "
            f"(выше 1 - отлично)"
        )
        print(
            f"   • Ожидаемая волатильность: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.metrics.risk)}"
        )
        print(
            f"   • Максимальная доля одной акции: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.weights.max())} "
            f"(лимит {FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.MAX_WEIGHT)})"
        )

        print("\n🔸 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
        print("   • Ребалансировка портфеля каждые 3-6 месяцев")
        print("   • Мониторинг фундаментальных показателей ежеквартально")
        print(
            f"   • Stop-loss: {FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.STOP_LOSS_THRESHOLD)} от цены покупки для каждой позиции"
        )
        print(
            f"   • Take-profit: +{FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.TAKE_PROFIT_THRESHOLD)} для недооцененных акций"
        )

        print(FORMATTING.SEPARATOR)
