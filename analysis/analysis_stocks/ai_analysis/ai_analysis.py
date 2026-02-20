# file: analysis/analysis_stocks/ai_analysis/ai_analysis.py
"""
Главный модуль для запуска анализа
"""
import sys
import numpy as np
from ai_models.ai_config import AnalysisConfig
from ai_models.ai_loader import DataLoader, DataPreprocessor
from ai_models.ai_analyzer import UndervaluedDetector
from ai_models.ai_optimizer import MarkowitzOptimizer
from ai_models.ai_report import ReportGenerator
from ai_models.ai_visualizer import ResultVisualizer


class StockAnalysisPipeline:
    """Основной пайплайн анализа акций"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация пайплайна
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.detector = UndervaluedDetector(config)
        self.optimizer = MarkowitzOptimizer(config)
        self.visualizer = ResultVisualizer(config)
        self.report_generator = ReportGenerator(config)
        
        # Данные для хранения результатов
        self.raw_data = None
        self.filtered_data = None
        self.undervalued_stocks = None
        self.portfolio_results = None
    
    def run(self):
        """Запуск полного пайплайна анализа"""
        try:
            print("="*60)
            print("ЗАПУСК АНАЛИЗА АКЦИЙ")
            print("="*60)
            
            # Шаг 1: Загрузка данных
            self.raw_data = self.data_loader.load_data()
            
            # Шаг 2: Предобработка
            print("\nПредобработка данных...")
            data_with_growth = self.preprocessor.calculate_growth_rate(self.raw_data)
            self.filtered_data = self.preprocessor.filter_companies(data_with_growth)
            
            # Шаг 3: Обучение нейронной сети
            X, y, _ = self.preprocessor.prepare_features(self.filtered_data)
            history = self.detector.train(X, y)
            
            # Шаг 4: Определение недооцененных акций
            print("\nОпределение недооцененных акций...")
            self.undervalued_stocks = self.detector.get_undervalued_stocks(
                self.filtered_data, top_n=20
            )
            
            # Шаг 5: Оптимизация портфеля с ограничениями
            print("\nОптимизация портфеля...")
            returns = self.undervalued_stocks['expected_return'].values
            betas = self.undervalued_stocks['beta'].fillna(1).values
            cov_matrix = self.optimizer.build_covariance_matrix(betas)
            
            # Оптимизация с ограничениями из конфига
            self.portfolio_results = self.optimizer.optimize(
                returns, 
                cov_matrix,
                min_weight=self.config.PORTFOLIO_MIN_WEIGHT,
                max_weight=self.config.PORTFOLIO_MAX_WEIGHT,
                min_assets=self.config.PORTFOLIO_MIN_ASSETS,
                max_assets=self.config.PORTFOLIO_MAX_ASSETS
            )
            
            # Шаг 6: Визуализация
            print("\nПостроение графиков...")
            all_portfolios, _ = self.optimizer.get_all_portfolios()
            self.visualizer.plot_efficient_frontier(
                all_portfolios,
                self.portfolio_results['max_sharpe'],
                self.portfolio_results['min_volatility']
            )
            self.visualizer.plot_nn_history(history)
            
            # Шаг 7: Сохранение результатов
            print("\nСохранение результатов...")
            self.report_generator.save_selected_stocks(self.undervalued_stocks)
            self.report_generator.save_portfolio_weights(
                self.undervalued_stocks['ticker'].tolist(),
                self.portfolio_results['max_sharpe'],
                self.portfolio_results['min_volatility']
            )
            
            train_loss, val_loss = self.detector.get_final_losses()
            self.report_generator.generate_text_report(
                self.undervalued_stocks,
                train_loss,
                val_loss,
                self.portfolio_results['max_sharpe'],
                self.portfolio_results['min_volatility']
            )
            
            # Итоговая статистика
            self._print_summary()
            
        except Exception as e:
            print(f"\nОшибка при выполнении анализа: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _print_summary(self):
        """Вывод итоговой статистики"""
        print("\n" + "="*50)
        print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
        print("="*50)
        print("\nСозданы файлы:")
        print(f"1. {self.config.OUTPUT_SELECTED_STOCKS} - отобранные недооцененные акции")
        print(f"2. {self.config.OUTPUT_PORTFOLIO_WEIGHTS} - веса оптимальных портфелей")
        print(f"3. {self.config.OUTPUT_EFFICIENT_FRONTIER} - график эффективной границы")
        print(f"4. {self.config.OUTPUT_NN_HISTORY} - история обучения нейронной сети")
        print(f"5. {self.config.OUTPUT_SUMMARY} - текстовый отчет с результатами")
        
        print("\nСтатистика портфелей:")
        ms = self.portfolio_results['max_sharpe']
        mv = self.portfolio_results['min_volatility']
        print(f"Max Sharpe: Доходность={ms.expected_return*100:.2f}%, "
              f"Волатильность={ms.volatility*100:.2f}%, Sharpe={ms.sharpe_ratio:.2f}")
        print(f"           Акций={ms.num_assets}, Макс.доля={ms.max_weight:.2%}, Мин.доля={ms.min_weight:.2%}")
        print(f"Min Vol:   Доходность={mv.expected_return*100:.2f}%, "
              f"Волатильность={mv.volatility*100:.2f}%, Sharpe={mv.sharpe_ratio:.2f}")
        print(f"           Акций={mv.num_assets}, Макс.доля={mv.max_weight:.2%}, Мин.доля={mv.min_weight:.2%}")


def main():
    """Точка входа в программу"""
    config = AnalysisConfig()
    pipeline = StockAnalysisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()