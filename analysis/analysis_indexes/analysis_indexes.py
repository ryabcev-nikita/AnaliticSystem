# portfolio_optimization.py
import os
import pandas as pd
from typing import Dict, Any
import warnings

from indexes_models.indexes_loader import DataLoader
from indexes_models.indexes_metrics import RiskMetricsCalculator
from indexes_models.indexes_report import ExcelReportGenerator, PortfolioAnalysisReport
from indexes_models.indexes_statistics import StatisticsCalculator
from indexes_models.indexes_visualizer import (
    PortfolioVisualizer,
)
from indexes_models.indexes_optimizer import CorrelationAnalyzer, PortfolioOptimizer
from indexes_models.indexes_constants import (
    RESULTS_DIR,
    INDEX_NAMES,
    VAR_CONFIDENCE_LEVELS,
    SIGNIFICANT_WEIGHT_THRESHOLD,
    OUTPUT_FILES,
    PortfolioMetrics,
)

warnings.filterwarnings("ignore")


class PortfolioAnalysisPipeline:
    """Основной класс для запуска полного анализа"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        self.data_loader = DataLoader()
        self.stats_calculator = StatisticsCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_calculator = RiskMetricsCalculator()
        self.visualizer = PortfolioVisualizer(self.results_dir)
        self.reporter = PortfolioAnalysisReport()
        self.excel_reporter = ExcelReportGenerator()

        self.prices = None
        self.returns = None
        self.stats = None
        self.correlation_matrix = None
        self.optimizer = None

    def run(self) -> Dict[str, Any]:
        """Запуск полного цикла анализа"""

        self.reporter.print_header(
            "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ПО МАРКОВИЦУ"
        )

        # 1. Загрузка данных
        print("\n1. Загрузка данных...")
        self.prices = self.data_loader.load_market_data()
        print(f"   Загружено {len(self.prices.columns)} индексов")
        print(
            f"   Период: с {self.prices.index[0].date()} по {self.prices.index[-1].date()}"
        )

        # 2. Расчет доходностей и статистик
        print("\n2. Расчет статистик...")
        self.returns = self.stats_calculator.calculate_returns(self.prices)
        self.stats = self.stats_calculator.calculate_annual_statistics(self.returns)

        # Добавляем максимальную просадку
        max_drawdowns = self.stats_calculator.calculate_all_max_drawdowns(self.prices)
        self.stats["Max Drawdown"] = pd.Series(max_drawdowns)

        print("\n   Статистика по индексам (годовая):")
        print(self.stats.round(4))

        # 3. Корреляционный анализ
        print("\n3. Корреляционный анализ...")
        self.correlation_matrix = self.correlation_analyzer.analyze(self.returns)

        # Сохраняем график
        corr_path = os.path.join(
            self.results_dir, f"{OUTPUT_FILES['correlation_matrix']}.png"
        )
        self.correlation_analyzer.plot(self.returns, str(corr_path), self.show_plots)

        # Вывод наиболее коррелированных пар
        print("\n   Наиболее коррелированные пары:")
        top_correlations = self.correlation_analyzer.get_top_correlations(5)
        for pair in top_correlations:
            print(
                f"   {INDEX_NAMES[pair['index1']]} - {INDEX_NAMES[pair['index2']]}: "
                f"{pair['correlation']:.4f}"
            )

        # 4. Оптимизация портфеля
        print("\n4. Оптимизация портфеля по Марковицу...")
        self.optimizer = PortfolioOptimizer(self.returns)

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

        # 5. Симуляция портфелей
        print("\n5. Симуляция случайных портфелей...")
        simulated_results, _ = self.optimizer.simulate_portfolios()

        # 6. Эффективная граница
        print("\n6. Построение эффективной границы...")
        efficient_portfolios = self.optimizer.efficient_frontier()

        # 7. Расчет метрик риска
        print("\n7. Расчет метрик риска...")

        # Создаем объекты с метриками для портфелей
        max_sharpe_metrics = PortfolioMetrics(
            weights=pd.Series(max_sharpe_weights, index=self.returns.columns),
            expected_return=max_sharpe_ret,
            volatility=max_sharpe_vol,
            sharpe_ratio=max_sharpe_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, max_sharpe_weights),
            hhi=self.risk_calculator.calculate_hhi(max_sharpe_weights),
            n_assets=len(
                max_sharpe_weights[max_sharpe_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
        )

        min_vol_metrics = PortfolioMetrics(
            weights=pd.Series(min_vol_weights, index=self.returns.columns),
            expected_return=min_vol_ret,
            volatility=min_vol_vol,
            sharpe_ratio=min_vol_sharpe,
            beta=self.risk_calculator.calculate_beta(self.returns, min_vol_weights),
            hhi=self.risk_calculator.calculate_hhi(min_vol_weights),
            n_assets=len(
                min_vol_weights[min_vol_weights > SIGNIFICANT_WEIGHT_THRESHOLD]
            ),
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

        # 8. Визуализация
        print("\n8. Визуализация результатов...")

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

        # 9. Вывод результатов
        self.reporter.print_header("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        self.reporter.print_portfolio_metrics(
            "Оптимальный портфель (Максимальный Sharpe Ratio)", max_sharpe_metrics
        )
        self.reporter.print_portfolio_metrics(
            "Портфель с минимальной волатильностью", min_vol_metrics
        )

        # 10. Сохранение в Excel
        print("\n10. Сохранение результатов в Excel...")
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.correlation_matrix,
            self.stats,
            max_sharpe_metrics,
            min_vol_metrics,
            efficient_portfolios,
            self.returns,
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


def create_model_analysis_indexes(show_plots: bool = True):
    """Основная функция для запуска анализа"""
    pipeline = PortfolioAnalysisPipeline(show_plots=show_plots)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    results = create_model_analysis_indexes(show_plots=True)
