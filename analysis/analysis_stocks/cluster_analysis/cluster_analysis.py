import os
import warnings

# Импорт констант
from cluster_models.cluster_analyzer.cluster_analyzer import ClusterAnalyzer
from cluster_models.cluster_analyzer.fundamental_analyzer import FundamentalAnalyzer
from cluster_models.cluster_loader.data_loader import DataLoader
from cluster_models.cluster_loader.path_config import CLUSTER_PATHS
from cluster_models.cluster_portfolio.portfolio_optimizer import (
    PortfolioManager,
    PortfolioOptimizer,
    PortfolioVisualizer,
)
from cluster_models.cluster_report.report_generator import ReportGenerator
from cluster_models.cluster_constants.cluster_constants import (
    CLUSTER,
    PORTFOLIO_CLUSTER,
    CLUSTER_FILES,
    CLUSTER_FORMAT,
)

warnings.filterwarnings("ignore")

# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================


def create_model_cluster_analysis_with_portfolio():
    """Полный анализ с кластеризацией и оптимизацией портфелей"""

    print(CLUSTER_FORMAT.SEPARATOR)
    print("🚀 ЗАПУСК КЛАСТЕРНОГО АНАЛИЗА И ОПТИМИЗАЦИИ ПОРТФЕЛЯ")
    print(CLUSTER_FORMAT.SEPARATOR)

    # Шаг 1: Загрузка данных
    print("\n📥 Загрузка данных...")

    if not os.path.exists(CLUSTER_PATHS["data_path"]):
        print(f"❌ Файл не найден: {CLUSTER_PATHS['data_path']}")
        return None, None

    loader = DataLoader()
    df = loader.load_and_clean_data(CLUSTER_PATHS["data_path"])

    print(f"   Загружено компаний: {len(df)}")

    # Шаг 2: Подготовка данных для кластеризации
    print("\n🔬 Проведение кластерного анализа...")

    cluster_features = list(CLUSTER.DEFAULT_CLUSTER_FEATURES)
    available_features = [f for f in cluster_features if f in df.columns]

    print(f"   Признаки для кластеризации: {available_features}")

    df_cluster = df[df[available_features].notna().all(axis=1)].copy()
    print(f"   Доступно для кластеризации: {len(df_cluster)} компаний")

    if len(df_cluster) < CLUSTER.MIN_DATA_FOR_CLUSTERING:
        print("❌ Недостаточно данных для кластеризации")
        return None, None

    cluster_analyzer = ClusterAnalyzer()
    optimal_k, _ = cluster_analyzer.find_optimal_clusters(
        df_cluster, available_features, max_clusters=CLUSTER.MAX_CLUSTERS
    )

    df_clustered = cluster_analyzer.fit_predict(
        df_cluster, available_features, n_clusters=optimal_k
    )

    cluster_profiles = cluster_analyzer.analyze_clusters(df_clustered)

    print(f"   Создано кластеров: {len(cluster_profiles)}")
    for profile in cluster_profiles:
        print(
            f"   Кластер {profile.cluster_id}: {profile.size} компаний - {profile.description}"
        )

    cluster_analyzer.plot_clusters(df_clustered, CLUSTER_PATHS["cluster_analysis"])

    # Шаг 3: Объединение результатов
    df_with_clusters = df.merge(
        df_clustered[["Ticker", "Cluster", "PCA1", "PCA2"]], on="Ticker", how="left"
    )

    # Шаг 4: Фундаментальный анализ
    print("\n📊 Расчет фундаментальных метрик...")

    df_with_clusters["Value_Score"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_value_score, axis=1
    )
    df_with_clusters["Quality_Score"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_quality_score, axis=1
    )
    df_with_clusters["Growth_Score"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_growth_score, axis=1
    )
    df_with_clusters["Income_Score"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_income_score, axis=1
    )
    df_with_clusters["Expected_Return"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_expected_return, axis=1
    )
    df_with_clusters["Risk"] = df_with_clusters.apply(
        FundamentalAnalyzer.calculate_risk, axis=1
    )

    # Шаг 5: Отбор кандидатов
    print("\n🎯 Отбор кандидатов в портфели...")

    candidates = df_with_clusters[
        (df_with_clusters["Cluster"].notna())
        & (df_with_clusters["Market_Cap"].fillna(0) > PORTFOLIO_CLUSTER.MIN_MARKET_CAP)
        & (
            df_with_clusters["Expected_Return"].fillna(0)
            > PORTFOLIO_CLUSTER.MIN_EXPECTED_RETURN
        )
        & (df_with_clusters["Risk"].fillna(1) < PORTFOLIO_CLUSTER.MAX_RISK_THRESHOLD)
    ].copy()

    if len(candidates) == 0:
        print("⚠️ Нет кандидатов, расширяем критерии...")
        candidates = df_with_clusters[
            (df_with_clusters["Cluster"].notna())
            & (
                df_with_clusters["Market_Cap"].fillna(0)
                > PORTFOLIO_CLUSTER.MIN_MARKET_CAP_LOOSE
            )
        ].copy()

    if len(candidates) > PORTFOLIO_CLUSTER.MAX_CANDIDATES:
        candidates["Total_Score"] = (
            candidates["Value_Score"] * PORTFOLIO_CLUSTER.VALUE_SCORE_WEIGHT
            + candidates["Quality_Score"] * PORTFOLIO_CLUSTER.QUALITY_SCORE_WEIGHT
            + candidates["Income_Score"] * PORTFOLIO_CLUSTER.INCOME_SCORE_WEIGHT
            + candidates["Expected_Return"]
            * PORTFOLIO_CLUSTER.RETURN_SCORE_MULTIPLIER
            * PORTFOLIO_CLUSTER.RETURN_SCORE_WEIGHT
        )
        candidates = candidates.nlargest(
            PORTFOLIO_CLUSTER.MAX_CANDIDATES, "Total_Score"
        )

    print(f"   Отобрано кандидатов: {len(candidates)}")

    if len(candidates) < CLUSTER.MIN_CLUSTERS:
        print("❌ Недостаточно кандидатов для формирования портфеля")
        return None, None

    # Шаг 6: Оптимизация портфелей
    print("\n📐 Оптимизация портфелей по различным стратегиям...")

    optimizer = PortfolioOptimizer(
        min_weight=PORTFOLIO_CLUSTER.MIN_WEIGHT_LOOSE,
        max_weight=PORTFOLIO_CLUSTER.MAX_WEIGHT_LOOSE,
    )

    portfolio_managers = {}
    weights_dict = optimizer.optimize_multi_portfolio(
        candidates.reset_index(drop=True), list(PORTFOLIO_CLUSTER.DEFAULT_STRATEGIES)
    )

    for port_name, weights in weights_dict.items():
        pm = PortfolioManager(port_name, candidates.reset_index(drop=True), weights)
        portfolio_managers[port_name] = pm
        print(
            f"   ✅ {port_name}: Шарп={pm.metrics.sharpe_ratio:.2f}, "
            f"Дох={pm.metrics.expected_return:.1%}, Риск={pm.metrics.risk:.1%}"
        )

    # Шаг 7: Визуализация
    print("\n📊 Создание визуализаций...")

    if portfolio_managers:
        PortfolioVisualizer.plot_portfolio_comparison(
            portfolio_managers, CLUSTER_PATHS["portfolio_comparison"]
        )

        if "Кластерный" in portfolio_managers:
            PortfolioVisualizer.plot_cluster_portfolio_allocation(
                portfolio_managers["Кластерный"], CLUSTER_PATHS["cluster_allocation"]
            )

    # Шаг 8: Генерация отчетов
    print("\n📄 Генерация отчетов...")

    ReportGenerator.generate_full_report(
        portfolio_managers,
        cluster_profiles,
        df_with_clusters,
        CLUSTER_PATHS["investment_cluster_report"],
    )

    df_with_clusters.to_excel(CLUSTER_PATHS["clustered_companies"], index=False)

    # Шаг 9: Итоговые рекомендации
    print("\n" + CLUSTER_FORMAT.SEPARATOR)
    print("🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print(CLUSTER_FORMAT.SEPARATOR)

    if portfolio_managers:
        best_portfolio = max(
            portfolio_managers.values(), key=lambda p: p.metrics.sharpe_ratio
        )

        print(f"\n🏆 РЕКОМЕНДУЕМЫЙ ПОРТФЕЛЬ: {best_portfolio.name}")
        print(f"   Ожидаемая доходность: {best_portfolio.metrics.expected_return:.1%}")
        print(f"   Риск: {best_portfolio.metrics.risk:.1%}")
        print(f"   Коэффициент Шарпа: {best_portfolio.metrics.sharpe_ratio:.2f}")

        print(
            f"\n📈 ТОП-{PORTFOLIO_CLUSTER.TOP_POSITIONS_RECOMMEND} ПОЗИЦИЙ В ПОРТФЕЛЕ:"
        )
        top_n = min(PORTFOLIO_CLUSTER.TOP_POSITIONS_RECOMMEND, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        for _, row in top_positions.iterrows():
            ticker = row.get("Ticker", "N/A")
            weight = row.get("Weight", 0)
            company = row.get("Company", "")[:30]
            pe = row.get("PE", 0)
            pb = row.get("PB", 0)
            roe = row.get("ROE", 0)
            print(f"   • {ticker}: {weight:.1%} - {company}")
            print(f"     P/E: {pe:.1f}, P/B: {pb:.2f}, ROE: {roe:.1f}%")

    print("\n" + CLUSTER_FORMAT.SEPARATOR)
    print("✅ Анализ завершен! Результаты сохранены в:")
    print(f"   • {CLUSTER_FILES.INVESTMENT_CLUSTER_REPORT} - полный отчет")
    print(f"   • {CLUSTER_FILES.CLUSTERED_COMPANIES_FILE} - компании с кластерами")
    print(f"   • {CLUSTER_FILES.CLUSTER_ANALYSIS_FILE} - визуализация кластеров")
    print(f"   • {CLUSTER_FILES.PORTFOLIO_COMPARISON_FILE} - сравнение портфелей")
    print(CLUSTER_FORMAT.SEPARATOR)

    return portfolio_managers, cluster_profiles


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    portfolios, clusters = create_model_cluster_analysis_with_portfolio()
