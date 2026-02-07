import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")


# Ваш класс автоэнкодера остается без изменений
class AnomalyDetectorAE(nn.Module):
    """Автоэнкодер для обнаружения аномалий в мультипликаторах"""

    def __init__(self, input_size):
        super().__init__()
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Скрытое представление
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_and_prepare_excel_data(file_path):
    """Загрузка и подготовка данных из Excel файла"""
    # Загружаем данные
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Преобразуем числовые значения (убираем 'млрд', 'млн' и т.д.)
    numeric_columns = [
        "Рыночная капитализация",
        "Выручка",
        "Чистая прибыль",
        "EBITDA",
        "Свободный денежный поток",
        "Дивиденд на акцию",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("млрд", "e9", regex=False)
            df[col] = df[col].astype(str).str.replace("млн", "e6", regex=False)
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Создаем нужные столбцы (адаптируем под ваши данные)
    df["dividend_yield"] = (
        df["Дивидендная доходность"] / 100
    )  # Преобразуем проценты в доли

    # Переименовываем столбцы для соответствия с кодом
    column_mapping = {
        "P/E": "P_E",
        "P/B": "P_B",
        "P/S": "P_S",
        "EV/EBITDA": "EV_EBITDA",
        "Debt/Capital": "debt_capital",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    return df


def detect_anomalies_with_ae(df):
    """Обнаружение аномалий и недооцененных акций с помощью автоэнкодера"""

    # Подготовка данных
    feature_cols = [
        "dividend_yield",
        "P_E",
        "P_B",
        "P_S",
        "NPM",
        "EV_EBITDA",
        "ROE",
        "debt_capital",
    ]

    # Проверяем, какие столбцы существуют
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Используемые признаки: {available_cols}")

    # Нормализация исходных данных (для справедливого сравнения)
    from sklearn.preprocessing import RobustScaler

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

    # Рассчитываем медианные значения признаков для всего датасета
    # (пригодится для сравнения)
    feature_medians = np.median(X, axis=0)
    feature_means = np.mean(X, axis=0)

    print("\nМедианные значения признаков по всему датасету:")
    for i, col in enumerate(available_cols):
        print(
            f"{col:<15}: Медиана = {feature_medians[i]:.4f}, Среднее = {feature_means[i]:.4f}"
        )

    # Нормализация для автоэнкодера
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение автоэнкодера
    input_size = len(available_cols)
    model = AnomalyDetectorAE(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Обучение
    n_epochs = 100
    print("\nНачинаем обучение автоэнкодера...")
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}, Reconstruction Loss: {total_loss/len(dataloader):.6f}"
            )

    # Вычисление ошибок реконструкции
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        _, reconstructed = model(X_tensor)

        # Ошибка реконструкции
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        # Получаем скрытые представления
        encoded, _ = model(X_tensor)
        encoded_features = encoded.numpy()

    # АНАЛИЗ ОШИБОК: используем медиану и IQR вместо mean и std
    errors_np = reconstruction_errors.numpy()

    # Медианный подход для определения аномалий
    error_median = np.median(errors_np)
    error_q1 = np.percentile(errors_np, 25)  # 25-й перцентиль
    error_q3 = np.percentile(errors_np, 75)  # 75-й перцентиль
    error_iqr = error_q3 - error_q1  # Межквартильный размах

    # Определяем аномалии через IQR метод (более устойчивый)
    anomaly_threshold_iqr = error_q3 + 1.5 * error_iqr
    mild_anomaly_threshold = error_q3 + 3 * error_iqr

    # Также считаем классический подход для сравнения
    error_mean = errors_np.mean()
    error_std = errors_np.std()
    anomaly_threshold_classic = error_mean + 2 * error_std

    # Определяем аномалии
    is_anomaly_iqr = errors_np > anomaly_threshold_iqr
    is_anomaly_classic = errors_np > anomaly_threshold_classic

    print(f"\n{'='*60}")
    print("СТАТИСТИКА ОШИБОК РЕКОНСТРУКЦИИ:")
    print(f"{'='*60}")
    print(f"Медиана ошибок:        {error_median:.6f}")
    print(f"Q1 (25-й перцентиль):  {error_q1:.6f}")
    print(f"Q3 (75-й перцентиль):  {error_q3:.6f}")
    print(f"IQR (Q3-Q1):           {error_iqr:.6f}")
    print(f"Средняя ошибка:        {error_mean:.6f}")
    print(f"Стандартное отклонение: {error_std:.6f}")
    print(f"\nПороги аномалий:")
    print(f"IQR метод (Q3 + 1.5*IQR): {anomaly_threshold_iqr:.6f}")
    print(f"Классический (mean + 2*std): {anomaly_threshold_classic:.6f}")
    print(f"Сильные аномалии (Q3 + 3*IQR): {mild_anomaly_threshold:.6f}")
    print(f"\nНайдено аномалий (IQR метод): {is_anomaly_iqr.sum()} из {len(errors_np)}")
    print(
        f"Найдено аномалий (Классический): {is_anomaly_classic.sum()} из {len(errors_np)}"
    )

    # Используем IQR метод как основной (более устойчивый)
    is_anomaly = is_anomaly_iqr

    # НОРМАЛИЗАЦИЯ ДЛЯ НЕДООЦЕНЕННОСТИ: учитываем медианные значения
    # Создаем "идеальный профиль" на основе медиан хороших компаний
    # Считаем медианы только для компаний с низкой ошибкой реконструкции
    low_error_mask = errors_np < error_median  # Компании с ошибкой ниже медианы
    if low_error_mask.sum() > 5:  # Если достаточно компаний
        good_companies_features = X[low_error_mask]
        ideal_profile = np.median(good_companies_features, axis=0)
        print(f"\nИдеальный профиль (медиана хороших компаний):")
        for i, col in enumerate(available_cols):
            print(f"{col:<15}: {ideal_profile[i]:.4f}")
    else:
        ideal_profile = feature_medians

    # Находим недооцененные акции с учетом медианных значений
    undervalued_scores = []
    undervalued_details = []

    for i, idx in enumerate(valid_indices):
        row = df.iloc[idx]
        current_features = X[i]

        # 1. ОЦЕНКА ПО ОШИБКЕ РЕКОНСТРУКЦИИ
        # Инвертируем ошибку: чем меньше ошибка, тем лучше
        # Используем экспоненциальное преобразование для чувствительности
        error_score = np.exp(-errors_np[i] / error_median)

        # 2. ОЦЕНКА ПО ФУНДАМЕНТАЛЬНЫМ ПОКАЗАТЕЛЯМ (относительно медиан)
        fundamental_score = 0
        deviation_penalty = 0

        # Для каждого признака сравниваем с медианой/идеальным значением
        for j, col in enumerate(available_cols):
            current_val = current_features[j]
            ideal_val = ideal_profile[j]

            # Определяем "хороший" диапазон для каждого показателя
            if col == "P_E":
                # P/E: чем меньше, тем лучше, но не отрицательный
                if 0 < current_val < 15:  # Абсолютный хороший диапазон
                    fundamental_score += 1
                # Также сравниваем с медианой
                if current_val < ideal_val * 0.8:  # Лучше медианы на 20%
                    fundamental_score += 0.5

            elif col == "P_B":
                if 0.5 < current_val < 2:  # Хороший диапазон
                    fundamental_score += 1
                if current_val < ideal_val * 0.9:
                    fundamental_score += 0.5

            elif col == "dividend_yield":
                if current_val > 0.05:  # Высокая доходность
                    fundamental_score += 1
                if current_val > ideal_val * 1.2:  # Выше медианы на 20%
                    fundamental_score += 0.5

            elif col == "ROE":
                if current_val > 0.1:  # ROE > 10%
                    fundamental_score += 1
                if current_val > ideal_val * 1.1:
                    fundamental_score += 0.5

            elif col == "NPM":
                if current_val > 0.05:  # Маржа > 5%
                    fundamental_score += 1

            elif col == "debt_capital":
                if current_val < 0.5:  # Долг < 50% капитала
                    fundamental_score += 1
                if current_val < ideal_val * 0.8:
                    fundamental_score += 0.5

        # 3. ОЦЕНКА ОТНОСИТЕЛЬНО СРЕДНИХ ЗНАЧЕНИЙ
        mean_deviation_score = 0
        for j, col in enumerate(available_cols):
            current_val = current_features[j]
            mean_val = feature_means[j]

            # Для "чем меньше, тем лучше" показателей
            if col in ["P_E", "P_B", "P_S", "EV_EBITDA", "debt_capital"]:
                if current_val < mean_val:
                    mean_deviation_score += 1
            # Для "чем больше, тем лучше" показателей
            elif col in ["dividend_yield", "ROE", "NPM"]:
                if current_val > mean_val:
                    mean_deviation_score += 1

        # 4. КОМБИНИРОВАННЫЙ СКОР
        # Базовый скор от ошибки реконструкции
        base_score = error_score

        # Бонус за хорошие фундаментальные показатели
        fundamental_bonus = fundamental_score / (len(available_cols) * 1.5)

        # Бонус за отклонение от среднего в правильную сторону
        mean_bonus = mean_deviation_score / len(available_cols)

        # Итоговый скор
        combined_score = base_score * (1 + fundamental_bonus + mean_bonus * 0.5)

        undervalued_scores.append(combined_score)

        # Сохраняем детали для отладки
        undervalued_details.append(
            {
                "ticker": tickers[i],
                "error": errors_np[i],
                "error_score": error_score,
                "fundamental_score": fundamental_score,
                "mean_deviation": mean_deviation_score,
                "final_score": combined_score,
            }
        )

    # Добавляем результаты в DataFrame
    df["AE_Ошибка_реконструкции"] = np.nan
    df["AE_Ошибка_нормализованная"] = np.nan
    df["AE_Аномалия"] = False
    df["AE_Сильная_аномалия"] = False
    df["AE_Недооцененность"] = np.nan
    df["AE_Ранг_недооцененности"] = np.nan
    df["AE_Скрытые_признаки"] = None
    df["AE_Топ_недооцененные"] = False

    for i, idx in enumerate(valid_indices):
        df.at[idx, "AE_Ошибка_реконструкции"] = errors_np[i]
        df.at[idx, "AE_Ошибка_нормализованная"] = errors_np[i] / error_median
        df.at[idx, "AE_Аномалия"] = is_anomaly_iqr[i]
        df.at[idx, "AE_Сильная_аномалия"] = errors_np[i] > mild_anomaly_threshold
        df.at[idx, "AE_Недооцененность"] = undervalued_scores[i]
        df.at[idx, "AE_Скрытые_признаки"] = str(encoded_features[i])

    # Добавляем ранг недооцененности (процентиль)
    if undervalued_scores:
        scores_array = np.array(undervalued_scores)
        for i, idx in enumerate(valid_indices):
            # Процентиль: какой процент акций имеет меньший скор
            percentile = (
                (scores_array <= scores_array[i]).sum() / len(scores_array) * 100
            )
            df.at[idx, "AE_Ранг_недооцененности"] = percentile

    # Определяем топ-10 недооцененных акций
    if len(undervalued_scores) >= 10:
        # Используем процентиль для определения топа
        top_indices = np.argsort(undervalued_scores)[-10:][::-1]

        # Также фильтруем: не должно быть аномалий
        filtered_top_indices = []
        for i in top_indices:
            if not is_anomaly_iqr[i]:  # Исключаем аномалии
                filtered_top_indices.append(i)
            if len(filtered_top_indices) >= 10:
                break

        # Если после фильтрации меньше 10, добавляем лучшие из оставшихся
        if len(filtered_top_indices) < 10:
            for i in top_indices:
                if i not in filtered_top_indices:
                    filtered_top_indices.append(i)
                if len(filtered_top_indices) >= 10:
                    break

        for i in filtered_top_indices:
            idx = valid_indices[i]
            df.at[idx, "AE_Топ_недооцененные"] = True

    # Выводим результаты
    print("\n" + "=" * 80)
    print("ТОП-10 НЕДООЦЕНЕННЫХ АКЦИЙ (с учетом медианных значений):")
    print("=" * 80)

    # Сортируем недооцененные акции
    undervalued_df = df[df["AE_Недооцененность"].notna()].copy()
    if not undervalued_df.empty:
        undervalued_df = undervalued_df.sort_values(
            "AE_Недооцененность", ascending=False
        )

        print(
            f"{'Тикер':<10} {'Название':<25} {'P/E':<6} {'P/B':<6} {'ДД,%':<6} {'ROE,%':<7} {'Ошибка':<10} {'Ранг,%':<7} {'Скор':<8}"
        )
        print("-" * 80)

        for _, row in undervalued_df.head(
            15
        ).iterrows():  # Показываем топ-15 для наглядности
            is_top = "★" if row.get("AE_Топ_недооцененные", False) else ""
            is_anom = "⚠" if row.get("AE_Аномалия", False) else ""

            print(
                f"{row.get('Тикер', ''):<10} "
                f"{str(row.get('Название', ''))[:23]:<25} "
                f"{row.get('P_E', 0):<6.1f} "
                f"{row.get('P_B', 0):<6.2f} "
                f"{(row.get('dividend_yield', 0)*100 if 'dividend_yield' in row else row.get('Дивидендная доходность', 0)):<6.1f} "
                f"{row.get('ROE', 0):<7.1f} "
                f"{row.get('AE_Ошибка_реконструкции', 0):<10.6f} "
                f"{row.get('AE_Ранг_недооцененности', 0):<7.1f} "
                f"{row.get('AE_Недооцененность', 0):<8.3f} {is_top}{is_anom}"
            )

    print("\n" + "=" * 80)
    print("АНАЛИЗ АНОМАЛИЙ:")
    print("=" * 80)

    # Группируем аномалии по типу
    if is_anomaly_iqr.any():
        print("\nАномалии, обнаруженные IQR методом:")
        anomalies_iqr_indices = np.where(is_anomaly_iqr)[0]

        for i in anomalies_iqr_indices[:10]:  # Показываем первые 10
            idx = valid_indices[i]
            row = df.iloc[idx]

            # Анализируем, какие признаки вызывают аномалию
            anomaly_reasons = []
            current_features = X[i]

            for j, col in enumerate(available_cols):
                current_val = current_features[j]
                median_val = feature_medians[j]
                q1_val = np.percentile(X[:, j], 25)
                q3_val = np.percentile(X[:, j], 75)

                # Проверяем, является ли значение выбросом для этого признака
                iqr = q3_val - q1_val
                is_feature_outlier = (current_val < q1_val - 1.5 * iqr) or (
                    current_val > q3_val + 1.5 * iqr
                )

                if is_feature_outlier:
                    direction = "выше" if current_val > q3_val + 1.5 * iqr else "ниже"
                    anomaly_reasons.append(
                        f"{col}({current_val:.2f} vs med:{median_val:.2f})"
                    )

            reasons_str = ", ".join(anomaly_reasons[:3])  # Берем 3 главные причины
            print(
                f"{row.get('Тикер', ''):<10} {str(row.get('Название', ''))[:25]:<25} "
                f"Ошибка: {errors_np[i]:.6f} | Причины: {reasons_str}"
            )

    # Статистика по методам
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МЕТОДОВ ОБНАРУЖЕНИЯ АНОМАЛИЙ:")
    print("=" * 80)

    common_anomalies = (is_anomaly_iqr & is_anomaly_classic).sum()
    only_iqr = (is_anomaly_iqr & ~is_anomaly_classic).sum()
    only_classic = (~is_anomaly_iqr & is_anomaly_classic).sum()

    print(f"Общие аномалии (оба метода): {common_anomalies}")
    print(f"Только IQR метод: {only_iqr}")
    print(f"Только классический метод: {only_classic}")

    if only_classic > 0:
        print("\nАномалии, обнаруженные только классическим методом:")
        classic_only_indices = np.where(~is_anomaly_iqr & is_anomaly_classic)[0]
        for i in classic_only_indices[:5]:
            idx = valid_indices[i]
            row = df.iloc[idx]
            print(
                f"  {row.get('Тикер', '')}: ошибка={errors_np[i]:.6f} "
                f"(mean={error_mean:.6f}, std={error_std:.6f}, threshold={anomaly_threshold_classic:.6f})"
            )

    return df, model, scaler


def save_results(df, output_path):
    """Сохранение результатов анализа"""
    # Сохраняем только нужные столбцы
    result_columns = [
        "Тикер",
        "Название",
        "P_E",
        "P_B",
        "EV_EBITDA",
        "Дивидендная доходность",
        "AE_Ошибка_реконструкции",
        "AE_Аномалия",
        "AE_Недооцененность",
        "AE_Топ_недооцененные",
    ]

    # Фильтруем существующие столбцы
    existing_columns = [col for col in result_columns if col in df.columns]
    result_df = df[existing_columns].copy()

    # Сохраняем в Excel
    result_df.to_excel(output_path, index=False)
    print(f"\nРезультаты сохранены в файл: {output_path}")

    return result_df


# Основной скрипт
def main():
    # Задаем пути к файлам
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = f"{parent_dir}/../data/fundamentals_shares.xlsx"
    output_file = f"{parent_dir}/../data/fundamentals_analysis_results.xlsx"

    try:
        # 1. Загружаем данные из Excel
        print("Загрузка данных из Excel...")
        df = load_and_prepare_excel_data(input_file)
        print(f"Загружено {len(df)} акций")

        # 2. Выполняем анализ с автоэнкодером
        print("\nЗапуск анализа с автоэнкодером...")
        df_with_analysis, model, scaler = detect_anomalies_with_ae(df)

        # 3. Сохраняем результаты
        save_results(df_with_analysis, output_file)

        # 4. Дополнительная информация
        print("\n" + "=" * 80)
        print("КРАТКАЯ СВОДКА:")
        print("=" * 80)

        total_stocks = len(df)
        analyzed_stocks = df_with_analysis["AE_Ошибка_реконструкции"].notna().sum()
        top_undervalued = df_with_analysis["AE_Топ_недооцененные"].sum()
        anomalies = df_with_analysis["AE_Аномалия"].sum()

        print(f"Всего акций: {total_stocks}")
        print(f"Проанализировано: {analyzed_stocks}")
        print(f"Топ недооцененных: {top_undervalued}")
        print(f"Аномалий обнаружено: {anomalies}")

        if top_undervalued > 0:
            print("\nРекомендации для инвестирования:")
            top_stocks = df_with_analysis[
                df_with_analysis["AE_Топ_недооцененные"] == True
            ]
            for _, row in top_stocks.iterrows():
                print(f"- {row['Тикер']}: {row['Название']}")

    except Exception as e:
        print(f"Ошибка при выполнении анализа: {str(e)}")


if __name__ == "__main__":
    main()
