# bonds_portfolio_optimization.py
import os
import pandas as pd
from typing import Dict, Any
import warnings

from bonds_models.bonds_report import BondsExcelReportGenerator
from bonds_models.bonds_optimizer import BondsPortfolioOptimizer
from bonds_models.bonds_scorer import BondScorer
from bonds_models.bonds_models import BondsAnalyzer, BondsDataLoader
from bonds_models.bonds_visualizer import BondsPortfolioVisualizer
from bonds_models.bonds_constants import (
    DATA_DIR,
    RESULTS_DIR,
    RISK_LEVELS,
)

warnings.filterwarnings("ignore")

# Создаем директорию для результатов
os.makedirs(RESULTS_DIR, exist_ok=True)


class BondsPortfolioAnalysisPipeline:
    """Основной класс для запуска анализа"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR

        self.bonds_df = None
        self.bonds_list = None
        self.analyzer = None
        self.scorer = None
        self.optimizer = None
        self.visualizer = BondsPortfolioVisualizer(self.results_dir)

    def run(self, file_path: str) -> Dict[str, Any]:
        """Запуск полного цикла анализа"""

        print("\n" + "=" * 80)
        print("АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ОБЛИГАЦИЙ")
        print("=" * 80)

        # 1. Загрузка данных
        print("\n1. Загрузка данных об облигациях...")
        loader = BondsDataLoader()
        raw_df = loader.load_bonds_data(file_path)
        print(f"   Загружено {len(raw_df)} облигаций")

        # 2. Анализ облигаций
        print("\n2. Анализ облигаций и расчет метрик...")
        self.analyzer = BondsAnalyzer(raw_df)
        self.bonds_df = self.analyzer.get_bonds_dataframe()
        self.bonds_list = self.analyzer.bonds_list
        print(f"   Обработано {len(self.bonds_list)} облигаций")

        # 3. Статистика
        stats_by_risk = self.analyzer.get_statistics_by_risk()
        stats_by_currency = self.analyzer.get_statistics_by_currency()

        print("\n   Статистика по уровням риска:")
        print(stats_by_risk)
        print("\n   Статистика по валютам:")
        print(stats_by_currency)

        # 4. Скоринг облигаций
        print("\n3. Скоринг облигаций...")
        self.scorer = BondScorer(self.bonds_list)
        top_bonds = self.scorer.get_top_bonds(50)
        print(f"   Отобрано топ-50 облигаций для оптимизации")

        # 5. Визуализация кривой доходности
        print("\n4. Построение кривой доходности...")
        self.visualizer.plot_yield_curve(self.bonds_df, self.show_plots)

        # 6. Оптимизация портфеля
        print("\n5. Оптимизация портфеля...")
        self.optimizer = BondsPortfolioOptimizer(top_bonds)
        optimal_weights, portfolio_stats = self.optimizer.optimize_portfolio()

        print(f"\n   Результаты оптимизации:")
        print(f"   Количество облигаций: {portfolio_stats['n_bonds']}")
        print(f"   Средняя доходность: {portfolio_stats['yield']*100:.2f}%")
        print(f"   Средняя дюрация: {portfolio_stats['duration']:.2f} лет")
        print(f"   Средний риск: {portfolio_stats['risk_score']:.2f}")
        print(f"   Диверсификация: {portfolio_stats['diversification']*100:.1f}%")

        # 7. Получение портфелей для разных уровней риска
        print("\n6. Формирование портфелей по уровням риска...")
        risk_portfolios = {}
        for level in [0, 1, 2, 3]:
            portfolio = self.optimizer.get_portfolio_by_risk_profile(level)
            if portfolio:
                risk_portfolios[level] = portfolio
                print(
                    f"   Риск-уровень {level} ({RISK_LEVELS[level]['name']}): "
                    f"{portfolio['statistics']['n_bonds']} облигаций, "
                    f"доходность {portfolio['statistics']['yield']*100:.2f}%, "
                    f"дюрация {portfolio['statistics']['duration']:.2f} лет"
                )

        # 8. Визуализация портфеля
        print("\n7. Визуализация портфеля...")
        selected_bonds = [top_bonds[i] for i in self.optimizer.selected_indices]
        selected_weights = optimal_weights[optimal_weights > 0]

        self.visualizer.plot_sector_allocation(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_currency_allocation(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_maturity_profile(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_risk_analysis(portfolio_stats, self.show_plots)

        # 9. Сохранение результатов в Excel
        print("\n8. Сохранение результатов...")

        # Добавляем скоринговые оценки в DataFrame
        score_data = []
        for bond in self.bonds_list:
            score_data.append(
                {
                    "ticker": bond.ticker,
                    "total_score": bond.total_score,
                    "yield_score": bond.yield_score,
                    "risk_score": bond.risk_score,
                    "liquidity_score": bond.liquidity_score_norm,
                    "duration_score": bond.duration_score,
                    "sector_score": bond.sector_score,
                    "currency_score": bond.currency_score,
                }
            )

        scores_df = pd.DataFrame(score_data)
        self.bonds_df = self.bonds_df.merge(scores_df, on="ticker", how="left")
        self.bonds_df = self.bonds_df.sort_values("total_score", ascending=False)

        self.excel_reporter = BondsExcelReportGenerator()
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.bonds_df,
            selected_bonds,
            selected_weights,
            portfolio_stats,
            stats_by_risk,
            stats_by_currency,
            risk_portfolios,
        )

        print(f"\n   Все результаты сохранены в: {self.results_dir}")

        return {
            "bonds_df": self.bonds_df,
            "bonds_list": self.bonds_list,
            "optimal_portfolio": {
                "bonds": selected_bonds,
                "weights": selected_weights,
                "statistics": portfolio_stats,
            },
            "risk_portfolios": risk_portfolios,
            "statistics": {"by_risk": stats_by_risk, "by_currency": stats_by_currency},
        }


def create_model_analysis_bonds():
    file_path = os.path.join(DATA_DIR, "bonds_data.xlsx")
    pipeline = BondsPortfolioAnalysisPipeline()
    results = pipeline.run(file_path)

    return results


if __name__ == "__main__":
    results = create_model_analysis_bonds()
