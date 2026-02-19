# metals_portfolio_optimization.py
import os
import pandas as pd
from typing import Dict, Any
import warnings

from commodities_models.commodities_optimizer import MetalsPortfolioOptimizer
from commodities_models.commodities_analyzer import MetalsCorrelationAnalyzer
from commodities_models.commodities_loader import MetalsDataLoader
from commodities_models.commodities_metrics import (
    MetalsPortfolioMetrics,
    MetalsRiskMetricsCalculator,
)
from commodities_models.commodities_report import (
    MetalsAnalysisReport,
    MetalsExcelReportGenerator,
)
from commodities_models.commodities_statistics import MetalsStatisticsCalculator
from commodities_models.commodities_visualizer import MetalsVisualizer

warnings.filterwarnings("ignore")

from commodities_models.commodities_constants import (
    MARKET_DATA_DIR,
    RESULTS_DIR,
    METAL_NAMES,
    VAR_CONFIDENCE_LEVELS,
    SIGNIFICANT_WEIGHT_THRESHOLD,
    OUTPUT_FILES,
)


class MetalsAnalysisPipeline:
    """Основной класс для запуска полного анализа металлов"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_loader = MetalsDataLoader()
        self.stats_calculator = MetalsStatisticsCalculator()
        self.correlation_analyzer = MetalsCorrelationAnalyzer()
        self.risk_calculator = MetalsRiskMetricsCalculator()
        self.visualizer = MetalsVisualizer(self.results_dir)
        self.reporter = MetalsAnalysisReport()
        self.excel_reporter = MetalsExcelReportGenerator()

        self.prices = None
        self.returns = None
        self.stats = None
        self.correlation_matrix = None
        self.optimizer = None

    def run(self) -> Dict[str, Any]:
        """Запуск полного цикла анализа металлов"""

        self.reporter.print_header(
            "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ МЕТАЛЛОВ"
        )

        # 1. Загрузка данных
        print("\n1. Загрузка данных по металлам...")
        self.prices = self.data_loader.load_metals_data()
        print(f"   Загружено {len(self.prices.columns)} металлов/сырьевых товаров")
        print(
            f"   Период: с {self.prices.index[0].date()} по {self.prices.index[-1].date()}"
        )
        print(f"   Торговых дней: {len(self.prices)}")

        # 2. Визуализация цен
        print("\n2. Визуализация цен...")
        self.visualizer.plot_prices_normalized(self.prices, self.show_plots)

        # 3. Расчет доходностей и статистик
        print("\n3. Расчет статистик...")
        self.returns = self.stats_calculator.calculate_returns(self.prices)
        self.stats = self.stats_calculator.calculate_annual_statistics(self.returns)

        # Добавляем максимальную просадку
        max_drawdowns = self.stats_calculator.calculate_all_max_drawdowns(self.prices)
        self.stats["Max Drawdown (%)"] = pd.Series(max_drawdowns) * 100

        # Добавляем skewness и kurtosis
        skew_kurt = self.stats_calculator.calculate_skewness_kurtosis(self.returns)

        print("\n   Статистика по металлам (годовая):")
        print(self.stats.round(2))

        # 4. Распределение доходностей
        print("\n4. Анализ распределения доходностей...")
        self.visualizer.plot_returns_distribution(self.returns, self.show_plots)

        # 5. Корреляционный анализ
        print("\n5. Корреляционный анализ...")
        self.correlation_matrix = self.correlation_analyzer.analyze(self.returns)

        # Сохраняем график
        corr_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['correlation_matrix']}.png"
        )
        self.correlation_analyzer.plot(self.returns, str(corr_path), self.show_plots)

        # Корреляции по группам
        group_correlations = self.correlation_analyzer.get_correlations_by_group()

        # Вывод наиболее коррелированных пар
        print("\n   Наиболее коррелированные пары металлов:")
        top_correlations = self.correlation_analyzer.get_top_correlations(5)
        for pair in top_correlations:
            print(
                f"   {METAL_NAMES[pair['metal1']]:<15} - {METAL_NAMES[pair['metal2']]:<15}: {pair['correlation']:.4f}"
            )

        # 6. Оптимизация портфеля
        print("\n6. Оптимизация портфеля металлов по Марковицу...")
        self.optimizer = MetalsPortfolioOptimizer(self.returns)

        # Портфель с максимальным Sharpe
        max_sharpe_weights = self.optimizer.optimize_max_sharpe()
        max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe = (
            self.optimizer.portfolio_statistics(max_sharpe_weights)
        )

        # Портфель с минимальной волатильностью
        min_vol_weights = self.optimizer.optimize_min_volatility()
        min_vol_ret, min_vol_vol, min_vol_sharpe = self.optimizer.portfolio_statistics(
            min_vol_weights
        )

        # 7. Симуляция портфелей
        print("\n7. Симуляция случайных портфелей...")
        simulated_results, _ = self.optimizer.simulate_portfolios()

        # 8. Эффективная граница
        print("\n8. Построение эффективной границы...")
        efficient_portfolios = self.optimizer.efficient_frontier()
        print(f"   Построено {len(efficient_portfolios)} точек эффективной границы")

        # 9. Расчет метрик риска
        print("\n9. Расчет метрик риска...")

        # Создаем объекты с метриками для портфелей
        max_sharpe_weights_series = pd.Series(
            max_sharpe_weights, index=self.returns.columns
        )
        min_vol_weights_series = pd.Series(min_vol_weights, index=self.returns.columns)

        # Веса по группам
        max_sharpe_group_weights = self.risk_calculator.calculate_group_weights(
            max_sharpe_weights_series
        )
        min_vol_group_weights = self.risk_calculator.calculate_group_weights(
            min_vol_weights_series
        )

        max_sharpe_metrics = MetalsPortfolioMetrics(
            weights=max_sharpe_weights_series,
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, max_sharpe_weights),
            hhi=self.risk_calculator.calculate_hhi(max_sharpe_weights),
            n_assets=len(
                max_sharpe_weights[max_sharpe_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
            precious_metals_weight=max_sharpe_group_weights["precious"],
            industrial_metals_weight=max_sharpe_group_weights["industrial"],
            energy_weight=max_sharpe_group_weights["energy"],
        )

        min_vol_metrics = MetalsPortfolioMetrics(
            weights=min_vol_weights_series,
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, min_vol_weights),
            hhi=self.risk_calculator.calculate_hhi(min_vol_weights),
            n_assets=len(
                min_vol_weights[min_vol_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
            precious_metals_weight=min_vol_group_weights["precious"],
            industrial_metals_weight=min_vol_group_weights["industrial"],
            energy_weight=min_vol_group_weights["energy"],
        )

        # Добавляем VaR и CVaR
        for conf_level in VAR_CONFIDENCE_LEVELS:
            var, cvar = self.risk_calculator.calculate_var_cvar(
                self.returns, max_sharpe_weights, conf_level
            )
            var2, cvar2 = self.risk_calculator.calculate_var_cvar(
                self.returns, min_vol_weights, conf_level
            )

            if conf_level == 0.95:
                max_sharpe_metrics.var_95 = var
                max_sharpe_metrics.cvar_95 = cvar
                min_vol_metrics.var_95 = var2
                min_vol_metrics.cvar_95 = cvar2
            else:
                max_sharpe_metrics.var_99 = var
                max_sharpe_metrics.cvar_99 = cvar
                min_vol_metrics.var_99 = var2
                min_vol_metrics.cvar_99 = cvar2

        # 10. Визуализация результатов
        print("\n10. Визуализация результатов...")

        max_sharpe_portfolio_dict = {
            "return": max_sharpe_ret,
            "volatility": max_sharpe_vol,
            "weights": max_sharpe_metrics.weights,
        }

        min_vol_portfolio_dict = {
            "return": min_vol_ret,
            "volatility": min_vol_vol,
            "weights": min_vol_metrics.weights,
        }

        self.visualizer.plot_efficient_frontier(
            self.returns,
            efficient_portfolios,
            max_sharpe_portfolio_dict,
            min_vol_portfolio_dict,
            simulated_results,
            self.show_plots,
        )

        self.visualizer.plot_weights_comparison(
            max_sharpe_metrics.weights,
            min_vol_metrics.weights,
            self.returns,
            self.show_plots,
        )

        self.visualizer.plot_group_allocation(
            max_sharpe_metrics, min_vol_metrics, self.show_plots
        )

        # 11. Вывод результатов
        self.reporter.print_header("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ПОРТФЕЛЯ МЕТАЛЛОВ")
        self.reporter.print_metal_statistics(self.stats)
        self.reporter.print_portfolio_metrics(
            "Оптимальный портфель (Максимальный Sharpe Ratio)", max_sharpe_metrics
        )
        self.reporter.print_portfolio_metrics(
            "Портфель с минимальной волатильностью", min_vol_metrics
        )

        # 12. Сохранение в Excel
        print("\n11. Сохранение результатов в Excel...")
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.correlation_matrix,
            self.stats,
            skew_kurt,
            max_sharpe_metrics,
            min_vol_metrics,
            efficient_portfolios,
            self.returns,
            group_correlations,
        )

        print(f"\n   Все результаты сохранены в: {self.results_dir}")

        return {
            "prices": self.prices,
            "returns": self.returns,
            "stats": self.stats,
            "correlation_matrix": self.correlation_matrix,
            "max_sharpe_portfolio": max_sharpe_portfolio_dict,
            "min_vol_portfolio": min_vol_portfolio_dict,
            "efficient_portfolios": efficient_portfolios,
            "max_sharpe_metrics": max_sharpe_metrics,
            "min_vol_metrics": min_vol_metrics,
        }


def main(show_plots: bool = True):
    """Основная функция для запуска анализа металлов"""
    print("\n" + "=" * 80)
    print("ЗАПУСК АНАЛИЗА РЫНКА МЕТАЛЛОВ И СЫРЬЕВЫХ ТОВАРОВ")
    print("=" * 80 + "\n")
    print(f"Директория с исходными данными: {MARKET_DATA_DIR}")
    print(f"Директория для сохранения результатов: {RESULTS_DIR}")
    print("\n" + "=" * 80 + "\n")

    pipeline = MetalsAnalysisPipeline(show_plots=show_plots)
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("АНАЛИЗ МЕТАЛЛОВ УСПЕШНО ЗАВЕРШЕН")
    print("=" * 80)
    print(f"\nСозданные файлы в {RESULTS_DIR}:")
    print(f"   1. {OUTPUT_FILES['correlation_matrix']}.png - Корреляционная матрица")
    print(f"   2. {OUTPUT_FILES['prices_chart']}.png - Нормализованные цены")
    print(
        f"   3. {OUTPUT_FILES['returns_distribution']}.png - Распределение доходностей"
    )
    print(f"   4. {OUTPUT_FILES['efficient_frontier']}.png - Эффективная граница")
    print(f"   5. {OUTPUT_FILES['weights_comparison']}.png - Сравнение весов")
    print(f"   6. metals_group_allocation.png - Распределение по группам")
    print(f"   7. {OUTPUT_FILES['full_report']}.xlsx - Полный отчет Excel")

    return results


if __name__ == "__main__":
    results = main(show_plots=True)
