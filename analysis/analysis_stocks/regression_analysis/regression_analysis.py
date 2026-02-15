"""
Основной скрипт для запуска анализа акций
"""

import pandas as pd
from .regression_models.regression_constants.multiplicator_constants import PATHS
from .regression_models.regression_loader.data_processor import (
    StockDataProcessor,
)
from .regression_models.regression_analyzer.regression_analyzer import (
    RegressionAnalyzer,
)
from .regression_models.regression_portfolio.portfolio_optimizer import (
    PortfolioOptimizerForRegression,
)
from .regression_models.regression_report.report_generator import (
    ReportGenerator,
)
import warnings
import traceback
import sys

warnings.filterwarnings("ignore")


def create_model_regression_analysis():
    """Основная функция для запуска анализа"""

    try:
        print("=" * 80)
        print("РЕГРЕССИОННЫЙ АНАЛИЗ АКЦИЙ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ")
        print("Версия с улучшенной обработкой выбросов и робастной регрессией")
        print("=" * 80)

        # Шаг 1: Обработка данных
        print("\n[1] ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ")
        print("-" * 50)
        try:
            processor = StockDataProcessor(PATHS["input_file"])
            processor.process(method="iqr")  # используем IQR для удаления выбросов
            data = processor.get_processed_data()
            print(f"\n✅ Обработано {len(data)} акций")
        except Exception as e:
            print(f"❌ Ошибка при обработке данных: {e}")
            traceback.print_exc()
            return

        # Шаг 2: Регрессионный анализ
        print("\n[2] РЕГРЕССИОННЫЙ АНАЛИЗ")
        print("-" * 50)
        try:
            analyzer = RegressionAnalyzer(data, PATHS["output_dir"])
            robust_models = analyzer.run_all_analyses()
            print(f"\n✅ Построено {len(robust_models)} робастных моделей")
        except Exception as e:
            print(f"❌ Ошибка при регрессионном анализе: {e}")
            traceback.print_exc()
            robust_models = {}

        # Шаг 3: Оптимизация портфеля
        print("\n[3] ОПТИМИЗАЦИЯ ПОРТФЕЛЯ")
        print("-" * 50)
        try:
            optimizer = PortfolioOptimizerForRegression(
                data, robust_models, PATHS["output_dir"]
            )
            optimal_portfolio, portfolio_metrics, selected_stocks = optimizer.optimize()

            if optimal_portfolio is not None:
                print(f"\n✅ Портфель оптимизирован успешно")
            else:
                print(f"\n⚠️ Портфель не удалось оптимизировать")

        except Exception as e:
            print(f"❌ Ошибка при оптимизации портфеля: {e}")
            traceback.print_exc()
            optimal_portfolio, portfolio_metrics, selected_stocks = (
                None,
                None,
                pd.DataFrame(),
            )

        # Шаг 4: Генерация отчета
        print("\n[4] ГЕНЕРАЦИЯ ИТОГОВОГО ОТЧЕТА")
        print("-" * 50)
        try:
            reporter = ReportGenerator(
                getattr(analyzer, "models", {}),  # OLS модели
                robust_models,  # робастные модели
                selected_stocks if "selected_stocks" in locals() else pd.DataFrame(),
                (
                    optimal_portfolio
                    if "optimal_portfolio" in locals()
                    else pd.DataFrame()
                ),
                portfolio_metrics if "portfolio_metrics" in locals() else {},
                PATHS["output_dir"],
            )
            reporter.save_report()
            print(f"\n✅ Отчет сгенерирован успешно")

        except Exception as e:
            print(f"❌ Ошибка при генерации отчета: {e}")
            traceback.print_exc()

        print("\n" + "=" * 80)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print(f"📁 Результаты сохранены в директории: {PATHS['output_dir']}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Анализ прерван пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_model_regression_analysis()
