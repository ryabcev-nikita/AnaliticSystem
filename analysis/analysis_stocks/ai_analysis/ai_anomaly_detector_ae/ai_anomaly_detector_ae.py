from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import RobustScaler
import torch

from ai_anomaly_detector_models.ai_anomaly_analyzer.ai_anomaly_analyzer import (
    AEPortfolioOptimizer,
    AnomalyDetectorAE,
)
from ai_anomaly_detector_models.ai_anomaly_loader.ai_anomaly_loader import (
    AEDataLoader,
)
from ai_anomaly_detector_models.ai_anomaly_loader.path_config import AE_PATHS
from ai_anomaly_detector_models.ai_anomaly_portfolio.ai_anomaly_portfolio_manager import (
    AEPortfolioManager,
)
from ai_anomaly_detector_models.ai_anomaly_visualizer.ai_anomaly_visualizer import (
    AEPortfolioVisualizer,
)
from ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_ARCH,
    AE_FEATURE,
    AE_FILES,
    AE_FORMAT,
    AE_PORTFOLIO,
    AE_SCORING,
    AE_THRESHOLD,
)
from ai_anomaly_detector_models.ai_anomaly_utils.ai_anomaly_utils import AEDataUtils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


@dataclass
class StockRecommendation:
    """Рекомендация по акции"""

    ticker: str
    name: str
    undervalued_score: float
    anomaly_score: float
    expected_return: float
    volatility: float
    allocation_max: float
    risk_category: str


def detect_anomalies_with_ae(df):
    """Обнаружение аномалий и недооцененных акций с помощью автоэнкодера"""

    available_cols = [col for col in AE_FEATURE.DEFAULT_FEATURES if col in df.columns]
    print(f"Используемые признаки: {available_cols}")

    feature_data = []
    valid_indices = []
    tickers = []

    for idx, row in df.iterrows():
        feature_vector = []
        valid = True

        for col in available_cols:
            val = row.get(col, None)
            if pd.isna(val):
                valid = False
                break
            feature_vector.append(float(val))

        if valid and len(feature_vector) == len(available_cols):
            feature_data.append(feature_vector)
            valid_indices.append(idx)
            tickers.append(row.get("Тикер", f"Row_{idx}"))

    if not feature_data:
        print("Недостаточно данных для обучения автоэнкодера")
        return df, None, None

    X = np.array(feature_data)
    print(f"Обучаем автоэнкодер на {len(feature_data)} акциях")

    feature_medians = np.median(X, axis=0)
    feature_means = np.mean(X, axis=0)

    print("\nМедианные значения признаков:")
    for i, col in enumerate(available_cols):
        print(f"{col:<15}: Медиана = {feature_medians[i]:.4f}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    input_size = len(available_cols)
    model = AnomalyDetectorAE(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=AE_ARCH.LEARNING_RATE)

    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    dataloader = DataLoader(dataset, batch_size=AE_ARCH.BATCH_SIZE, shuffle=True)

    print("\nОбучение автоэнкодера...")
    for epoch in range(AE_ARCH.N_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % AE_ARCH.EPOCH_LOG_INTERVAL == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        _, reconstructed = model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        encoded, _ = model(X_tensor)
        encoded_features = encoded.numpy()

    errors_np = reconstruction_errors.numpy()

    error_median = np.median(errors_np)
    error_q1 = np.percentile(errors_np, AE_THRESHOLD.Q1_PERCENTILE)
    error_q3 = np.percentile(errors_np, AE_THRESHOLD.Q3_PERCENTILE)
    error_iqr = error_q3 - error_q1
    anomaly_threshold = error_q3 + AE_THRESHOLD.ANOMALY_IQR_MULTIPLIER * error_iqr
    strong_anomaly_threshold = (
        error_q3 + AE_THRESHOLD.STRONG_ANOMALY_IQR_MULTIPLIER * error_iqr
    )

    is_anomaly = errors_np > anomaly_threshold
    is_strong_anomaly = errors_np > strong_anomaly_threshold

    print(f"\nПорог аномалий: {anomaly_threshold:.6f}")
    print(f"Найдено аномалий: {is_anomaly.sum()} из {len(errors_np)}")

    low_error_mask = errors_np < error_median
    if low_error_mask.sum() > AE_SCORING.GOOD_COMPANIES_MIN_COUNT:
        good_companies_features = X[low_error_mask]
        ideal_profile = np.median(good_companies_features, axis=0)
    else:
        ideal_profile = feature_medians

    undervalued_scores = []

    for i in range(len(X)):
        current_features = X[i]
        error_score = np.exp(-errors_np[i] / error_median)

        fundamental_score = 0
        for j, col in enumerate(available_cols):
            current_val = current_features[j]
            ideal_val = ideal_profile[j]

            if col in AE_FEATURE.LOWER_IS_BETTER_FEATURES:
                if (
                    0
                    < current_val
                    < ideal_val * AE_THRESHOLD.IDEAL_PROFILE_UNDERVALUED_FACTOR
                ):
                    fundamental_score += AE_SCORING.SCORE_STRONG_UNDERVALUED
                elif current_val < ideal_val:
                    fundamental_score += AE_SCORING.SCORE_UNDERVALUED
            elif col in AE_FEATURE.HIGHER_IS_BETTER_FEATURES:
                if (
                    current_val
                    > ideal_val * AE_THRESHOLD.IDEAL_PROFILE_OVERVAULED_FACTOR
                ):
                    fundamental_score += AE_SCORING.SCORE_STRONG_UNDERVALUED
                elif current_val > ideal_val:
                    fundamental_score += AE_SCORING.SCORE_UNDERVALUED

        combined_score = error_score * (
            1
            + fundamental_score
            / (len(available_cols) * AE_SCORING.FUNDAMENTAL_SCORE_FACTOR)
        )
        undervalued_scores.append(combined_score)

    df = AEDataUtils.add_ae_results_to_df(
        df,
        valid_indices,
        errors_np,
        is_anomaly,
        is_strong_anomaly,
        undervalued_scores,
        error_median,
    )

    AEDataUtils.print_top_undervalued(df)

    return df, model, scaler


def create_portfolios_from_ae_results(df, ae_optimizer):
    """Создание портфелей на основе результатов автоэнкодера"""

    print("\n" + AE_FORMAT.SEPARATOR)
    print("💼 ФОРМИРОВАНИЕ ПОРТФЕЛЕЙ НА ОСНОВЕ АВТОЭНКОДЕРА")
    print(AE_FORMAT.SEPARATOR)

    candidates = df[
        (df["AE_Недооцененность"].notna())
        & (df["AE_Ошибка_реконструкции"].notna())
        & (df["AE_Аномалия"] == False)
    ].copy()

    print(f"Кандидатов после фильтрации аномалий: {len(candidates)}")

    if len(candidates) < AE_PORTFOLIO.MIN_CANDIDATES:
        candidates = df[
            (df["AE_Недооцененность"].notna())
            & (df["AE_Недооцененность"] > df["AE_Недооцененность"].median())
        ].copy()
        print(f"Кандидатов после расширения: {len(candidates)}")

    candidates["AE_Expected_Return"] = candidates.apply(
        ae_optimizer.calculate_expected_return, axis=1
    )
    candidates["AE_Volatility"] = candidates.apply(
        ae_optimizer.calculate_volatility, axis=1
    )

    final_candidates = candidates[
        (candidates["AE_Expected_Return"] > AE_PORTFOLIO.MIN_EXPECTED_RETURN)
        & (candidates["AE_Volatility"] < AE_PORTFOLIO.MAX_VOLATILITY_THRESHOLD)
    ].copy()

    if len(final_candidates) > AE_PORTFOLIO.MAX_CANDIDATES:
        final_candidates = final_candidates.nlargest(
            AE_PORTFOLIO.MAX_CANDIDATES, "AE_Недооцененность"
        )

    print(f"Финальных кандидатов: {len(final_candidates)}")

    portfolios = {}

    if len(final_candidates) >= AE_PORTFOLIO.MIN_PORTFOLIO_SIZE:
        expected_returns = final_candidates["AE_Expected_Return"].values
        cov_matrix = ae_optimizer.create_covariance_matrix(final_candidates)
        opt_result = ae_optimizer.optimize_portfolio(
            expected_returns, cov_matrix, undervalued_boost=True
        )

        pm = AEPortfolioManager(
            "Марковиц-оптимальный",
            final_candidates.reset_index(drop=True),
            opt_result["combined_weights"],
            ae_optimizer,
        )
        portfolios["Марковиц-оптимальный"] = pm
        print(f"✅ Марковиц-оптимальный: Шарп={pm.metrics.sharpe_ratio:.2f}")

    ae_portfolios = ae_optimizer.create_ae_based_portfolios(final_candidates)

    for name, (df_port, weights) in ae_portfolios.items():
        if len(df_port) >= AE_PORTFOLIO.MIN_PORTFOLIO_SIZE:
            pm = AEPortfolioManager(
                name, df_port.reset_index(drop=True), weights, ae_optimizer
            )
            portfolios[name] = pm
            print(f"✅ {name}: Шарп={pm.metrics.sharpe_ratio:.2f}")

    return portfolios, final_candidates


def save_portfolio_results(df, portfolios, candidates, output_path):
    """Сохранение результатов портфельного анализа"""

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=AE_FILES.SHEET_AE_RESULTS, index=False)
        candidates.to_excel(writer, sheet_name=AE_FILES.SHEET_CANDIDATES, index=False)

        if portfolios:
            summary = []
            for name, pm in portfolios.items():
                summary.append(
                    {
                        "Портфель": name,
                        "Доходность": f"{pm.metrics.expected_return:.2%}",
                        "Риск": f"{pm.metrics.risk:.2%}",
                        "Шарп": f"{pm.metrics.sharpe_ratio:.2f}",
                        "VaR": f"{pm.metrics.var_95:.2%}",
                        "Диверсификация": f"{pm.metrics.diversification_score:.1%}",
                        "Позиций": len(pm.df),
                        "Топ_недооцененных": pm.df["AE_Топ_недооцененные"].sum(),
                    }
                )
            pd.DataFrame(summary).to_excel(
                writer, sheet_name=AE_FILES.SHEET_PORTFOLIO_SUMMARY, index=False
            )

            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )
            best_portfolio.df.to_excel(
                writer, sheet_name=AE_FILES.SHEET_BEST_PORTFOLIO, index=False
            )

    print(f"✅ Результаты сохранены в {output_path}")


def create_model_ai_anomaly_detector_ae():
    """Основная функция"""

    print(AE_FORMAT.SEPARATOR)
    print("🚀 АВТОЭНКОДЕР + ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ПО МАРКОВИЦУ")
    print(AE_FORMAT.SEPARATOR)

    try:
        print("\n📥 Загрузка данных...")
        df = AEDataLoader.load_and_prepare_excel_data(AE_PATHS["input_file"])
        print(f"Загружено {len(df)} акций")

        print("\n🧠 Обучение автоэнкодера...")
        df_with_ae, model, scaler = detect_anomalies_with_ae(df)

        if model is None:
            print("❌ Ошибка обучения автоэнкодера")
            return

        print("\n📊 Визуализация анализа аномалий...")
        AEPortfolioVisualizer.plot_anomaly_analysis(
            df_with_ae, AE_PATHS["ae_anomaly_file"]
        )

        ae_optimizer = AEPortfolioOptimizer(
            min_weight=AE_PORTFOLIO.MIN_WEIGHT,
            max_weight=AE_PORTFOLIO.MAX_WEIGHT,
            risk_free_rate=AE_PORTFOLIO.RISK_FREE_RATE,
        )

        portfolios, candidates = create_portfolios_from_ae_results(
            df_with_ae, ae_optimizer
        )

        if portfolios:
            print("\n📊 Визуализация портфелей...")

            AEPortfolioVisualizer.plot_portfolio_comparison(
                portfolios, AE_PATHS["ae_portfolio_comparison"]
            )

            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )
            AEPortfolioVisualizer.plot_portfolio_summary(best_portfolio)

        save_portfolio_results(
            df_with_ae, portfolios, candidates, AE_PATHS["output_file"]
        )

        print("\n" + AE_FORMAT.SEPARATOR)
        print("🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
        print(AE_FORMAT.SEPARATOR)

        if portfolios:
            best_portfolio = max(
                portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
            )

            print(f"\n🏆 РЕКОМЕНДУЕМЫЙ ПОРТФЕЛЬ: {best_portfolio.name}")
            print(
                f"   Ожидаемая доходность: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}"
            )
            print(
                f"   Риск: {AE_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}"
            )
            print(
                f"   Коэффициент Шарпа: {AE_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}"
            )

            print(f"\n📈 ТОП-5 ПОЗИЦИЙ В ПОРТФЕЛЕ:")
            top_n = min(AE_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
            top_5 = best_portfolio.get_top_positions(top_n)

            for _, row in top_5.iterrows():
                ticker = row.get("Тикер", "N/A")
                weight = row.get("Weight", 0)
                company = str(row.get("Название", ""))[:30]
                score = row.get("AE_Недооцененность", 0)
                rank = row.get("AE_Ранг_недооцененности", 0)

                print(
                    f"   • {ticker}: {AE_FORMAT.PERCENT_FORMAT.format(weight)} - {company}"
                )
                print(f"     Скор: {score:.3f}, Ранг: {rank:.1f}%")

            print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
            anomaly_alloc = best_portfolio.get_anomaly_allocation()
            for category, weight in anomaly_alloc.items():
                if weight > 0:
                    print(f"   • {category}: {AE_FORMAT.PERCENT_FORMAT.format(weight)}")

        print("\n" + AE_FORMAT.SEPARATOR)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print(AE_FORMAT.SEPARATOR)

    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    create_model_ai_anomaly_detector_ae()
