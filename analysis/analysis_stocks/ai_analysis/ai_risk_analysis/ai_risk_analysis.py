# ==================== ИНТЕГРИРОВАННЫЙ ПАЙПЛАЙН ====================
import os

import pandas as pd
from .ai_risk_models.ai_risk_analyzer.ai_risk_analyzer import (
    NeuralRiskAssessor,
)
from .ai_risk_models.ai_risk_constants.ai_risk_constants import (
    NN_ARCH,
    NN_FORMAT,
    NN_PORTFOLIO,
)
from .ai_risk_models.ai_risk_loader.path_config import (
    NN_RISK_PATHS,
)
from .ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_manager import (
    NNRiskPortfolioManager,
)
from .ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_optimizer import (
    NNRiskPortfolioOptimizer,
)
from .ai_risk_models.ai_risk_report.report_generator import (
    NNRiskReportGenerator,
)
from .ai_risk_models.ai_risk_visualizer.ai_risk_visualizer import (
    NNRiskPortfolioVisualizer,
)


def create_model_ai_risk_analysis():
    """Полный пайплайн: обучение нейросети + оптимизация портфеля"""

    print(NN_FORMAT.SEPARATOR)
    print("🚀 ЗАПУСК НЕЙРОСЕТЕВОГО АНАЛИЗА РИСКОВ И ОПТИМИЗАЦИИ ПОРТФЕЛЯ")
    print(NN_FORMAT.SEPARATOR)

    print("\n📥 Загрузка данных...")

    if not os.path.exists(NN_RISK_PATHS["input_file"]):
        print(f"❌ Файл не найден: {NN_RISK_PATHS['input_file']}")
        return None, None

    df = pd.read_excel(NN_RISK_PATHS["input_file"])
    print(f"   Загружено {len(df)} компаний")

    ai_risk_analyzer = NeuralRiskAssessor(n_folds=NN_ARCH.N_FOLDS)
    print("\n🧠 Обучение нейросетевой модели оценки риска...")
    df_with_risk, models, scaler = ai_risk_analyzer.train_risk_assessment_ensemble(df)

    if models is None:
        print("❌ Ошибка обучения нейросети!")
        return df_with_risk, None

    print("\n📊 Подготовка данных для оптимизации портфеля...")

    optimizer = NNRiskPortfolioOptimizer(
        min_weight=NN_PORTFOLIO.MIN_WEIGHT,
        max_weight=NN_PORTFOLIO.MAX_WEIGHT,
        risk_free_rate=NN_PORTFOLIO.RISK_FREE_RATE,
    )

    df_with_risk["NN_Expected_Return"] = df_with_risk.apply(
        optimizer.calculate_expected_return, axis=1
    )
    df_with_risk["NN_Volatility"] = df_with_risk.apply(
        optimizer.calculate_volatility, axis=1
    )

    print("\n💼 Создание портфелей на основе категорий риска...")

    candidates = df_with_risk[
        (df_with_risk["NN_Категория_риска"].notna())
        & (df_with_risk["NN_Уверенность"] > NN_PORTFOLIO.MIN_CONFIDENCE)
        & (df_with_risk["NN_Expected_Return"] > NN_PORTFOLIO.MIN_EXPECTED_RETURN)
        & (df_with_risk["NN_Volatility"] < NN_PORTFOLIO.MAX_VOLATILITY_THRESHOLD)
    ].copy()

    if len(candidates) > NN_PORTFOLIO.MAX_CANDIDATES:
        candidates["NN_Score"] = (
            (1 - candidates["NN_Категория_риска"] / 3) * NN_PORTFOLIO.RISK_SCORE_WEIGHT
            + candidates["NN_Уверенность"] * NN_PORTFOLIO.CONFIDENCE_WEIGHT
            + (candidates["NN_Expected_Return"] / NN_PORTFOLIO.RETURN_NORMALIZATION)
            * NN_PORTFOLIO.RETURN_WEIGHT
        )
        candidates = candidates.nlargest(NN_PORTFOLIO.MAX_CANDIDATES, "NN_Score")

    print(f"   Отобрано кандидатов: {len(candidates)}")

    if len(candidates) < 3:
        print("❌ Недостаточно кандидатов для формирования портфеля")
        return df_with_risk, None

    portfolios = {}

    try:
        expected_returns = candidates["NN_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(candidates)
        opt_result = optimizer.optimize_portfolio(expected_returns, cov_matrix)

        pm_optimized = NNRiskPortfolioManager(
            "Марковиц-оптимальный",
            candidates.reset_index(drop=True),
            opt_result["combined_weights"],
            optimizer,
        )
        portfolios["Марковиц-оптимальный"] = pm_optimized
        print(
            f"   ✅ Марковиц-оптимальный: Шарп={pm_optimized.metrics.sharpe_ratio:.2f}"
        )
    except Exception as e:
        print(f"   ⚠️ Ошибка оптимизации Марковица: {e}")

    risk_portfolios = optimizer.optimize_risk_based_portfolios(candidates)

    for name, (df_port, weights) in risk_portfolios.items():
        try:
            pm = NNRiskPortfolioManager(
                name, df_port.reset_index(drop=True), weights, optimizer
            )
            portfolios[name] = pm
            print(f"   ✅ {name}: Шарп={pm.metrics.sharpe_ratio:.2f}")
        except Exception as e:
            print(f"   ⚠️ Ошибка создания {name}: {e}")

    print("\n📊 Создание визуализаций...")

    if portfolios:
        NNRiskPortfolioVisualizer.plot_portfolio_comparison(
            portfolios, NN_RISK_PATHS["ai_risk_portfolio_comparison"]
        )

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        NNRiskPortfolioVisualizer.plot_portfolio_summary(best_portfolio)

        NNRiskPortfolioVisualizer.plot_efficient_frontier(
            optimizer,
            candidates,
            portfolios,
            NN_RISK_PATHS["ai_risk_efficient_frontier"],
        )

    print("\n📄 Сохранение результатов...")

    NNRiskReportGenerator.generate_full_report(
        df_with_risk, candidates, portfolios, NN_RISK_PATHS["ai_risk_portfolio_results"]
    )

    print("\n" + NN_FORMAT.SEPARATOR)
    print("🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print(NN_FORMAT.SEPARATOR)

    if portfolios:
        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)

        print(f"\n🏆 РЕКОМЕНДУЕМЫЙ ПОРТФЕЛЬ: {best_portfolio.name}")
        print(
            f"   Ожидаемая доходность: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}"
        )
        print(
            f"   Риск: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}"
        )
        print(
            f"   Коэффициент Шарпа: {NN_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}"
        )
        print(
            f"   VaR (95%): {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.var_95)}"
        )

        print(f"\n📈 ТОП-{NN_FORMAT.TOP_POSITIONS_BEST} ПОЗИЦИЙ В ПОРТФЕЛЕ:")
        top_n = min(NN_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        for _, row in top_positions.iterrows():
            ticker = row.get("Тикер", "N/A")
            weight = row.get("Weight", 0)
            company = str(row.get("Название", ""))[:30]
            risk_cat = row.get("NN_Категория_текст", "N/A")
            confidence = row.get("NN_Уверенность", 0)

            print(
                f"   • {ticker}: {NN_FORMAT.PERCENT_FORMAT.format(weight)} - {company}"
            )
            print(
                f"     Риск: {risk_cat}, Уверенность: {NN_FORMAT.PERCENT_FORMAT.format(confidence)}"
            )

        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ РИСКА:")
        risk_alloc = best_portfolio.get_risk_category_allocation()
        for category, weight in risk_alloc.items():
            print(f"   • {category}: {NN_FORMAT.PERCENT_FORMAT.format(weight)}")

    print("\n" + NN_FORMAT.SEPARATOR)
    print("✅ Анализ завершен!")
    print(NN_FORMAT.SEPARATOR)


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    try:
        create_model_ai_risk_analysis()
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении анализа: {str(e)}")
        import traceback

        traceback.print_exc()
