import warnings

import pandas as pd
from tree_solver_models.tree_solver_analyzer.decision_tree_model import (
    DecisionTreeModel,
)
from tree_solver_models.tree_solver_analyzer.fundamental_analyzer import (
    FundamentalAnalyzer,
)
from tree_solver_models.tree_solver_analyzer.multiplier_analyzer import (
    MultiplierAnalyzer,
)
from tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FORMATTING,
    PORTFOLIO_CONSTANTS,
    TARGET_MAPPING,
)
from tree_solver_models.tree_solver_loader.data_loader import (
    DataLoader,
)
from tree_solver_models.tree_solver_loader.path_config import (
    PATHS,
)
from tree_solver_models.tree_solver_market.market_analyzer import (
    MarketAnalyzer,
)
from tree_solver_models.tree_solver_portfolio.portfolio_manager import (
    PortfolioManager,
)
from tree_solver_models.tree_solver_portfolio.portfolio_optimizer_tree import (
    PortfolioOptimizerTree,
)
from tree_solver_models.tree_solver_report.report_generator import (
    ReportGenerator,
)
from tree_solver_models.tree_solver_vizualizer.portfolio_visualizer import (
    PortfolioVisualizer,
)


warnings.filterwarnings("ignore")


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================
def main():
    """Основной пайплайн анализа и оптимизации портфеля"""

    print("🚀 Запуск анализа фундаментальных показателей и оптимизации портфеля...")

    # Шаг 1: Загрузка данных
    print("📥 Загрузка данных...")
    loader = DataLoader()
    df = loader.load_and_clean_data(PATHS["file_path"])
    print(df)
    # Шаг 2: Расчет рыночных бенчмарков
    print("📊 Расчет рыночных бенчмарков...")
    market_benchmarks = MarketAnalyzer.calculate_benchmarks(df)

    print(f"\n📈 МЕДИАННЫЕ ЗНАЧЕНИЯ МУЛЬТИПЛИКАТОРОВ:")
    print(f"   P/E: {FORMATTING.FLOAT_FORMAT_1D.format(market_benchmarks.pe_median)}")
    print(f"   P/B: {FORMATTING.FLOAT_FORMAT_2D.format(market_benchmarks.pb_median)}")
    print(f"   P/S: {FORMATTING.FLOAT_FORMAT_2D.format(market_benchmarks.ps_median)}")
    print(
        f"   ROE: {FORMATTING.PERCENT_FORMAT.format(market_benchmarks.roe_median / 100)}"
    )
    print(
        f"   Дивидендная доходность: {FORMATTING.PERCENT_FORMAT.format(market_benchmarks.div_yield_median / 100)}"
    )
    print(
        f"   Долг/Капитал: {FORMATTING.PERCENT_FORMAT.format(market_benchmarks.debt_capital_median / 100)}"
    )
    print(
        f"   Бета: {FORMATTING.FLOAT_FORMAT_2D.format(market_benchmarks.beta_median)}"
    )

    # Шаг 3: Обучение дерева решений
    print("\n🌳 Обучение модели дерева решений...")
    dt_model = DecisionTreeModel()
    training_results = dt_model.train(df)

    print(
        f"   Точность на обучающей выборке: {FORMATTING.PERCENT_FORMAT.format(training_results['train_accuracy'])}"
    )
    print(
        f"   Точность на тестовой выборке: {FORMATTING.PERCENT_FORMAT.format(training_results['test_accuracy'])}"
    )

    print("\n🔍 Важность признаков в модели:")
    feature_importance = sorted(
        training_results["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    for feature, importance in feature_importance[:5]:
        if importance > 0.01:
            print(f"   {feature}: {FORMATTING.PERCENT_FORMAT.format(importance)}")

    # Шаг 4: Предсказание для всех акций
    print("\n🎯 Оценка всех акций...")
    df = dt_model.predict(df)

    # Шаг 5: Фундаментальный анализ
    print("📉 Расчет фундаментальной доходности и риска...")
    fundamental_analyzer = FundamentalAnalyzer(market_benchmarks)

    df["Ожидаемая_доходность"] = df.apply(
        fundamental_analyzer.calculate_expected_return, axis=1
    )
    df["Риск"] = df.apply(fundamental_analyzer.calculate_risk, axis=1)

    # Шаг 6: Отбор кандидатов в портфель
    print("🎯 Отбор кандидатов в портфель...")

    candidates = df[
        (
            df["Predicted_Оценка"].isin(
                [TARGET_MAPPING.STRONG_UNDERVALUED, TARGET_MAPPING.UNDERVALUED]
            )
        )
        & (df["Рыночная капитализация"].fillna(0) > PORTFOLIO_CONSTANTS.MIN_MARKET_CAP)
        & (
            df["Ожидаемая_доходность"].fillna(0)
            > PORTFOLIO_CONSTANTS.MIN_EXPECTED_RETURN
        )
        & (df["Риск"].fillna(1) < PORTFOLIO_CONSTANTS.MAX_RISK)
    ].copy()

    if len(candidates) < 5:
        print("   ⚠️ Мало кандидатов, расширяем критерии...")
        candidates = df[
            (
                df["Predicted_Оценка"].isin(
                    [
                        TARGET_MAPPING.STRONG_UNDERVALUED,
                        TARGET_MAPPING.UNDERVALUED,
                        TARGET_MAPPING.FAIR_VALUE,
                    ]
                )
            )
            & (
                df["Рыночная капитализация"].fillna(0)
                > PORTFOLIO_CONSTANTS.MIN_MARKET_CAP_LOOSE
            )
            & (
                df["Ожидаемая_доходность"].fillna(0)
                > PORTFOLIO_CONSTANTS.MIN_EXPECTED_RETURN_LOOSE
            )
        ].copy()

    if len(candidates) > PORTFOLIO_CONSTANTS.MAX_CANDIDATES:
        candidates = candidates.nlargest(
            PORTFOLIO_CONSTANTS.MAX_CANDIDATES, "Predicted_Уверенность"
        )

    print(f"   Отобрано кандидатов: {len(candidates)}")

    if len(candidates) < 2:
        print("❌ Недостаточно кандидатов для формирования портфеля!")
        return None

    # Шаг 7: Оптимизация портфеля
    print("📐 Оптимизация портфеля по Марковицу...")
    optimizer = PortfolioOptimizerTree(
        min_weight=PORTFOLIO_CONSTANTS.MIN_WEIGHT,
        max_weight=(
            PORTFOLIO_CONSTANTS.MAX_WEIGHT_LOOSE
            if len(candidates) < 10
            else PORTFOLIO_CONSTANTS.MAX_WEIGHT
        ),
    )

    try:
        cov_matrix = optimizer.create_covariance_matrix(candidates)

        optimization_result = optimizer.optimize(
            candidates["Ожидаемая_доходность"].values, cov_matrix
        )

        # Шаг 8: Создание портфеля
        print("💼 Формирование итогового портфеля...")
        portfolio_manager = PortfolioManager(
            candidates, optimization_result["combined_weights"]
        )

        # Шаг 9: Визуализация
        print("📊 Создание визуализации...")
        visualizer = PortfolioVisualizer()

        dt_model.plot_tree(PATHS["decision_tree"])

        visualizer.plot_portfolio_summary(
            candidates,
            optimization_result["combined_weights"],
            portfolio_manager.metrics,
            market_benchmarks,
            PATHS["optimal_portfolio"],
        )

        visualizer.plot_efficient_frontier(
            candidates["Ожидаемая_доходность"].values,
            cov_matrix,
            optimization_result["combined_weights"],
            portfolio_manager.metrics.expected_return,
            portfolio_manager.metrics.risk,
            PATHS["efficient_frontier"],
        )

        # Шаг 10: Генерация отчетов
        print("📄 Генерация отчетов...")
        report_generator = ReportGenerator()
        report_generator.generate_portfolio_report(
            portfolio_manager, market_benchmarks, PATHS["portfolio_report"]
        )

        # Шаг 11: Вывод рекомендаций
        report_generator.print_recommendations(portfolio_manager)

        print("\n✅ Анализ завершен! Результаты сохранены в:")
        print(f"   • {PATHS['portfolio_report']} - детальный отчет")
        print(f"   • {PATHS['optimal_portfolio']} - визуализация портфеля")
        print(f"   • {PATHS['efficient_frontier']} - граница эффективности")
        print(f"   • {PATHS['decision_tree']} - дерево решений")

        return portfolio_manager

    except Exception as e:
        print(f"❌ Ошибка при оптимизации портфеля: {e}")
        return None


# ==================== ЗАПУСК С ДОПОЛНИТЕЛЬНЫМ АНАЛИЗОМ ====================

if __name__ == "__main__":
    portfolio = main()

    if portfolio is not None:
        print(FORMATTING.SEPARATOR)
        print("📊 ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ МУЛЬТИПЛИКАТОРОВ")
        print(FORMATTING.SEPARATOR)

        loader = DataLoader()
        df_full = loader.load_and_clean_data(PATHS["file_path"])
        df_full["Сектор"] = df_full["Название"].apply(MarketAnalyzer.assign_sector)

        multiplier_analyzer = MultiplierAnalyzer()
        sector_multipliers = multiplier_analyzer.analyze_sector_multipliers(df_full)

        print("\n📈 МУЛЬТИПЛИКАТОРЫ ПО СЕКТОРАМ:")
        print(sector_multipliers.round(2).to_string(index=False))

        best_values = multiplier_analyzer.find_best_values(df_full)
        print("\n🏆 ЛУЧШИЕ ЗНАЧЕНИЯ НА РЫНКЕ:")
        for key, value in best_values.items():
            if pd.notna(value):
                if "доходность" in key.lower():
                    print(f"   {key}: {FORMATTING.PERCENT_FORMAT.format(value / 100)}")
                elif "капитализация" in key.lower():
                    print(
                        f"   {key}: {FORMATTING.BILLIONS_FORMAT.format(value / CONVERSION.BILLION)}"
                    )
                else:
                    print(f"   {key}: {FORMATTING.FLOAT_FORMAT_2D.format(value)}")
