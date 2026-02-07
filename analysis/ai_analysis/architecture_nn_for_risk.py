import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def create_advanced_risk_assessment_nn(input_shape, num_classes=4):
    """Создание усовершенствованного ансамбля нейросетей для оценки риска"""

    # 1. Основная сеть с регуляризацией
    def create_model_1():
        model = keras.Sequential(
            [
                layers.Dense(
                    128,
                    activation="relu",
                    input_shape=(input_shape,),
                    kernel_regularizer=regularizers.l2(0.001),
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(
                    64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model

    # 2. Сеть с residual connections
    def create_model_2():
        inputs = layers.Input(shape=(input_shape,))

        # Первый блок
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Residual connection
        residual = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(residual)
        x = layers.Add()([residual, x])
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(32, activation="relu")(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    # 3. Широкая сеть
    def create_model_3():
        model = keras.Sequential(
            [
                layers.Dense(256, activation="relu", input_shape=(input_shape,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model

    # Создаем модели
    models = [create_model_1(), create_model_2(), create_model_3()]

    return models


def prepare_features_for_nn(df):
    """Подготовка признаков для нейросети с учетом beta коэффициента"""

    # Базовые признаки (обязательные)
    basic_features = [
        "dividend_yield",  # Дивидендная доходность
        "P_E",  # P/E
        "P_B",  # P/B
        "P_S",  # P/S
        "NPM",  # Чистая маржа прибыли
        "EV_EBITDA",  # EV/EBITDA
        "ROE",  # Рентабельность капитала
        "debt_capital",  # Долг/Капитал
        "EPS",  # Прибыль на акцию
    ]

    # Рыночные риски
    risk_features = [
        "Бета",  # Beta коэффициент
        "P_FCF",  # Цена/Свободный денежный поток
        "ROA",  # Рентабельность активов
        "ROIC",  # Рентабельность инвестиций
        "Debt_EBITDA",  # Долг/EBITDA
    ]

    # Проверяем наличие столбцов
    available_basic = [col for col in basic_features if col in df.columns]
    available_risk = [col for col in risk_features if col in df.columns]

    # Если beta нет в данных, попробуем найти под другими названиями
    beta_aliases = ["Бета", "Beta", "beta", "Бета коэффициент", "Бетта", "БЕТА"]
    beta_col = None
    for alias in beta_aliases:
        if alias in df.columns:
            beta_col = alias
            break

    all_features = available_basic + available_risk
    print(f"Используется {len(all_features)} признаков для нейросети:")
    for i, feat in enumerate(all_features):
        print(f"  {i+1}. {feat}")

    if beta_col:
        print(f"✓ Beta коэффициент найден в столбце: '{beta_col}'")
    else:
        print("✗ Beta коэффициент не найден в данных")

    # Собираем данные
    X = []
    tickers = []
    valid_indices = []

    for idx, row in df.iterrows():
        feature_vector = []
        valid = True

        # Базовые признаки
        for col in available_basic:
            val = row.get(col, None)
            if pd.isna(val):
                valid = False
                break
            try:
                feature_vector.append(float(val))
            except (ValueError, TypeError):
                valid = False
                break

        # Если нет базовых признаков, пропускаем
        if not valid or len(feature_vector) < len(available_basic):
            continue

        # Рыночные риски (добавляем с обработкой NaN)
        for col in available_risk:
            val = row.get(col, None)
            if pd.isna(val):
                # Заполняем медианным значением по столбцу
                col_median = (
                    df[col].median() if col in df.columns and not df[col].empty else 0
                )
                feature_vector.append(col_median)
            else:
                try:
                    feature_vector.append(float(val))
                except (ValueError, TypeError):
                    # Если преобразование не удалось, используем медиану
                    col_median = (
                        df[col].median()
                        if col in df.columns and not df[col].empty
                        else 0
                    )
                    feature_vector.append(col_median)

        # Добавляем дополнительный риск на основе beta
        if beta_col and beta_col in row:
            beta_val = row[beta_col]
            if not pd.isna(beta_val):
                try:
                    beta_float = float(beta_val)
                    # Нормализуем beta для нейросети (от 0 до 1)
                    beta_normalized = max(min(beta_float, 3.0), -1.0) / 3.0
                    feature_vector.append(beta_normalized)
                except (ValueError, TypeError):
                    feature_vector.append(0.5)  # Среднее значение по умолчанию
            else:
                feature_vector.append(0.5)  # Среднее значение
        else:
            feature_vector.append(0.5)  # По умолчанию

        # Проверяем, что вектор признаков имеет правильную длину
        if len(feature_vector) == (len(available_basic) + len(available_risk) + 1):
            X.append(feature_vector)
            tickers.append(row.get("Тикер", f"Row_{idx}"))
            valid_indices.append(idx)

    if not X:
        print("❌ Недостаточно данных для обучения нейросети")
        return None, None, None, None

    X = np.array(X)
    print(f"✅ Подготовлено {len(X)} акций с {X.shape[1]} признаками")

    return X, tickers, valid_indices, all_features


def calculate_risk_categories(df, use_ae_scores=True):
    """Рассчет категорий риска для обучения нейросети"""

    y = []
    category_details = []

    for idx, row in df.iterrows():
        # Собираем показатели риска
        risk_factors = []

        # 1. P/E риск (чем выше P/E, тем выше риск переоценки)
        p_e = row.get("P_E", 20)
        try:
            p_e_float = float(p_e)
            if p_e_float <= 0:
                p_e_risk = 3  # Отрицательный P/E = очень высокий риск
            elif p_e_float < 10:
                p_e_risk = 0  # Низкий риск
            elif p_e_float < 20:
                p_e_risk = 1  # Средний риск
            elif p_e_float < 30:
                p_e_risk = 2  # Высокий риск
            else:
                p_e_risk = 3  # Очень высокий риск
        except (ValueError, TypeError):
            p_e_risk = 2  # Средне-высокий риск по умолчанию

        risk_factors.append(p_e_risk)

        # 2. Debt/Capital риск
        debt_cap = row.get("debt_capital", 0.5)
        try:
            debt_cap_float = float(debt_cap)
            if debt_cap_float < 0.3:
                debt_risk = 0
            elif debt_cap_float < 0.5:
                debt_risk = 1
            elif debt_cap_float < 0.7:
                debt_risk = 2
            else:
                debt_risk = 3
        except (ValueError, TypeError):
            debt_risk = 2

        risk_factors.append(debt_risk)

        # 3. ROE риск (чем ниже ROE, тем выше риск)
        roe = row.get("ROE", 0.1)
        try:
            roe_float = float(roe)
            if roe_float > 0.15:
                roe_risk = 0
            elif roe_float > 0.10:
                roe_risk = 1
            elif roe_float > 0.05:
                roe_risk = 2
            else:
                roe_risk = 3
        except (ValueError, TypeError):
            roe_risk = 2

        risk_factors.append(roe_risk)

        # 4. Beta риск (если есть)
        beta = row.get("Бета", 1.0)
        try:
            beta_float = float(beta)
            if beta_float < 0.7:
                beta_risk = 0
            elif beta_float < 1.0:
                beta_risk = 1
            elif beta_float < 1.3:
                beta_risk = 2
            else:
                beta_risk = 3
        except (ValueError, TypeError):
            beta_risk = 1  # Средний риск по умолчанию

        risk_factors.append(beta_risk)

        # 5. Дивидендная доходность (низкая = выше риск)
        div_yield = row.get("dividend_yield", row.get("Дивидендная доходность", 0))
        try:
            if isinstance(div_yield, (int, float)):
                div_yield_float = div_yield
            else:
                # Пробуем преобразовать строку
                div_yield_float = float(div_yield) / 100  # Если в процентах

            if div_yield_float > 0.08:
                div_risk = 0
            elif div_yield_float > 0.05:
                div_risk = 1
            elif div_yield_float > 0.02:
                div_risk = 2
            else:
                div_risk = 3
        except (ValueError, TypeError):
            div_risk = 2

        risk_factors.append(div_risk)

        # 6. Аномалии от автоэнкодера (если есть)
        if use_ae_scores and "AE_Аномалия" in row:
            try:
                ae_anomaly = bool(row["AE_Аномалия"])
                ae_strong = row.get("AE_Сильная_аномалия", False)

                if ae_strong:
                    ae_risk = 3
                elif ae_anomaly:
                    ae_risk = 2
                else:
                    ae_risk = 0
            except:
                ae_risk = 0
        else:
            ae_risk = 0

        risk_factors.append(ae_risk)

        # Усредняем риски (игнорируем нули от отсутствующих факторов)
        valid_risks = [r for r in risk_factors if r is not None]
        if valid_risks:
            avg_risk = np.mean(valid_risks)
        else:
            avg_risk = 2  # Средний риск по умолчанию

        # Определяем категорию
        if avg_risk < 1.0:
            category = 0  # 'A: Низкий риск'
        elif avg_risk < 1.8:
            category = 1  # 'B: Средний риск'
        elif avg_risk < 2.5:
            category = 2  # 'C: Высокий риск'
        else:
            category = 3  # 'D: Очень высокий риск'

        y.append(category)
        category_details.append(
            {
                "avg_risk": avg_risk,
                "p_e_risk": p_e_risk,
                "debt_risk": debt_risk,
                "roe_risk": roe_risk,
                "beta_risk": beta_risk,
                "div_risk": div_risk,
                "ae_risk": ae_risk,
            }
        )

    return np.array(y), category_details


def train_risk_assessment_ensemble(df, n_folds=3, use_ae_results=True):
    """Обучение ансамбля нейросетей для оценки риска"""

    print("=" * 80)
    print("НАЧАЛО ОБУЧЕНИЯ НЕЙРОСЕТИ ДЛЯ ОЦЕНКИ РИСКА")
    print("=" * 80)

    # 1. Подготовка признаков
    X, tickers, valid_indices, feature_names = prepare_features_for_nn(df)

    if X is None:
        print("❌ Ошибка подготовки данных!")
        return df, None, None

    # 2. Рассчет целевых категорий
    y, risk_details = calculate_risk_categories(df, use_ae_results)

    # Берем только те индексы, для которых есть признаки
    y_filtered = y[valid_indices]

    # Проверяем баланс классов
    unique, counts = np.unique(y_filtered, return_counts=True)
    print("\n📊 Распределение категорий риска:")

    # Создаем словарь для отображения
    actual_categories = {}
    for cat, count in zip(unique, counts):
        if cat == 0:
            cat_name = "A: Низкий риск"
        elif cat == 1:
            cat_name = "B: Средний риск"
        elif cat == 2:
            cat_name = "C: Высокий риск"
        elif cat == 3:
            cat_name = "D: Очень высокий риск"
        else:
            cat_name = f"Категория {cat}"

        actual_categories[cat] = cat_name
        print(f"  {cat_name}: {count} акций ({count/len(y_filtered)*100:.1f}%)")

    # 3. Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. One-hot кодирование целей
    num_classes = len(actual_categories)
    print(f"\n🎯 Количество классов для классификации: {num_classes}")

    if num_classes < 2:
        print("❌ Недостаточно классов для классификации! Нужно минимум 2 класса.")
        return df, None, None

    # Создаем маппинг для классов (чтобы они шли подряд от 0)
    class_mapping = {
        old: new for new, old in enumerate(sorted(actual_categories.keys()))
    }
    y_mapped = np.array([class_mapping[cat] for cat in y_filtered])

    # One-hot кодирование
    y_categorical = to_categorical(y_mapped, num_classes=num_classes)

    # Обновляем названия категорий
    actual_category_names = [
        actual_categories[old] for old in sorted(actual_categories.keys())
    ]

    # 5. Стратифицированная кросс-валидация
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Создаем ансамбль моделей (с правильным количеством выходов)
    models = create_advanced_risk_assessment_nn(X.shape[1], num_classes=num_classes)

    # Для хранения предсказаний
    ensemble_predictions = np.zeros_like(y_categorical)
    fold_metrics = []

    print("\n" + "=" * 80)
    print(f"НАЧИНАЕМ КРОСС-ВАЛИДАЦИЮ ({n_folds} фолдов)")
    print("=" * 80)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_mapped)):
        print(f"\n🔹 Обучение fold {fold + 1}/{n_folds}")
        print(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

        fold_model_predictions = []
        fold_model_accuracies = []

        # Обучаем каждую модель в ансамбле
        for i, model in enumerate(models):
            print(f"   🧠 Модель {i+1}/{len(models)}...", end=" ")

            # Компиляция модели
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy", Precision(), Recall()],
            )

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
                ),
            ]

            # Обучение
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=16,
                callbacks=callbacks,
                verbose=0,
            )

            # Оценка модели
            val_loss, val_acc, val_precision, val_recall = model.evaluate(
                X_val, y_val, verbose=0
            )
            fold_model_accuracies.append(val_acc)

            # Прогнозирование
            val_pred = model.predict(X_val, verbose=0)
            fold_model_predictions.append(val_pred)

            print(f"Accuracy: {val_acc:.3f}")

        # Усреднение предсказаний моделей
        avg_pred = np.mean(fold_model_predictions, axis=0)
        ensemble_predictions[val_idx] = avg_pred

        # Метрики фолда
        fold_accuracy = np.mean(fold_model_accuracies)
        fold_metrics.append(
            {
                "fold": fold + 1,
                "accuracy": fold_accuracy,
                "model_accuracies": fold_model_accuracies,
            }
        )

    # 6. Анализ результатов
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 80)

    # Средняя точность по фолдам
    avg_accuracy = np.mean([fm["accuracy"] for fm in fold_metrics])
    print(f"\n📈 Средняя точность ансамбля: {avg_accuracy:.3f}")

    # Точность по моделям
    print("\n🧠 Точность отдельных моделей:")
    for i, model_metrics in enumerate(
        zip(*[fm["model_accuracies"] for fm in fold_metrics])
    ):
        avg_model_acc = np.mean(model_metrics)
        print(f"  Модель {i+1}: {avg_model_acc:.3f}")

    # Классификационный отчет
    y_true = np.argmax(y_categorical, axis=1)
    y_pred = np.argmax(ensemble_predictions, axis=1)

    print("\n📊 Классификационный отчет:")
    try:
        print(
            classification_report(
                y_true, y_pred, target_names=actual_category_names, digits=3
            )
        )
    except:
        print("Не удалось создать classification report")

    # 7. Добавление результатов в DataFrame
    df["NN_Категория_риска"] = np.nan
    df["NN_Уверенность"] = np.nan
    df["NN_Категория_текст"] = ""

    # Обратное маппирование категорий
    reverse_class_mapping = {new: old for old, new in class_mapping.items()}
    category_map_text = {
        0: "A: Низкий риск",
        1: "B: Средний риск",
        2: "C: Высокий риск",
        3: "D: Очень высокий риск",
    }

    # Заполняем результаты для валидных индексов
    for i, idx in enumerate(valid_indices):
        if i < len(y_pred):  # На всякий случай проверяем границы
            original_category = reverse_class_mapping.get(y_pred[i], y_pred[i])
            df.at[idx, "NN_Категория_риска"] = original_category
            df.at[idx, "NN_Уверенность"] = np.max(ensemble_predictions[i])

            # Получаем текстовое представление категории
            if original_category in category_map_text:
                df.at[idx, "NN_Категория_текст"] = category_map_text[original_category]
            else:
                df.at[idx, "NN_Категория_текст"] = f"Категория {original_category}"

    # 8. Анализ распределения рисков
    print("\n" + "=" * 80)
    print("РАСПРЕДЕЛЕНИЕ РИСКОВ ПО АКЦИЯМ")
    print("=" * 80)

    # Считаем распределение по предсказанным категориям
    predicted_categories = df["NN_Категория_текст"].value_counts()

    # Полный список возможных категорий для отчета
    all_categories = [
        "A: Низкий риск",
        "B: Средний риск",
        "C: Высокий риск",
        "D: Очень высокий риск",
    ]

    total_valid = predicted_categories.sum() if not predicted_categories.empty else 0

    for category in all_categories:
        count = predicted_categories.get(category, 0)
        percentage = count / total_valid * 100 if total_valid > 0 else 0
        print(f"{category:<30}: {count:>3} акций ({percentage:.1f}%)")

    # 9. Статистика по уверенности предсказаний
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ПО УВЕРЕННОСТИ ПРЕДСКАЗАНИЙ")
    print("=" * 80)

    confidence_scores = df["NN_Уверенность"].dropna()
    if len(confidence_scores) > 0:
        print(f"Средняя уверенность: {confidence_scores.mean():.3f}")
        print(f"Медианная уверенность: {confidence_scores.median():.3f}")
        print(f"Минимальная уверенность: {confidence_scores.min():.3f}")
        print(f"Максимальная уверенность: {confidence_scores.max():.3f}")

    # 10. Рекомендации по результатам
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)

    # Акции с низким риском (если такие есть)
    low_risk_stocks = df[df["NN_Категория_текст"] == "A: Низкий риск"]
    if not low_risk_stocks.empty:
        print(f"\n🏆 ТОП-5 АКЦИЙ С НИЗКИМ РИСКОМ:")
        print("-" * 70)
        for _, row in low_risk_stocks.head(5).iterrows():
            ticker = row.get("Тикер", "N/A")
            name = str(row.get("Название", "N/A"))[:25]
            p_e = row.get("P_E", "N/A")
            div_yield = row.get("dividend_yield", row.get("Дивидендная доходность", 0))
            beta = row.get("Бета", "N/A")
            confidence = row.get("NN_Уверенность", 0)

            # Форматируем вывод
            p_e_str = f"{float(p_e):.1f}" if isinstance(p_e, (int, float)) else str(p_e)
            div_yield_pct = (
                float(div_yield) * 100 if isinstance(div_yield, (int, float)) else 0
            )
            beta_str = (
                f"{float(beta):.2f}" if isinstance(beta, (int, float)) else str(beta)
            )

            print(
                f"{ticker:<8} {name:<25} "
                f"P/E: {p_e_str:<6} "
                f"ДД: {div_yield_pct:.1f}% "
                f"Beta: {beta_str:<6} "
                f"Уверенность: {confidence:.2f}"
            )

    # Акции с высоким риском
    high_risk_stocks = df[df["NN_Категория_текст"] == "D: Очень высокий риск"]
    if not high_risk_stocks.empty:
        print(f"\n⚠️  АКЦИИ С ОЧЕНЬ ВЫСОКИМ РИСКОМ (осторожно!):")
        print("-" * 70)
        for _, row in high_risk_stocks.head(3).iterrows():
            ticker = row.get("Тикер", "N/A")
            name = str(row.get("Название", "N/A"))[:25]
            p_e = row.get("P_E", "N/A")
            debt_cap = row.get("debt_capital", "N/A")
            confidence = row.get("NN_Уверенность", 0)

            print(
                f"{ticker:<8} {name:<25} "
                f"P/E: {p_e if isinstance(p_e, str) else f'{p_e:.1f}':<6} "
                f"Долг/Капитал: {debt_cap if isinstance(debt_cap, str) else f'{debt_cap:.2f}':<5} "
                f"Уверенность: {confidence:.2f}"
            )

    return df, models, scaler


def get_risk_recommendations(df):
    """Получение персональных рекомендаций на основе оценки риска"""

    recommendations = []

    for _, row in df.iterrows():
        ticker = row.get("Тикер", "Unknown")
        risk_category = row.get("NN_Категория_текст", "")
        confidence = row.get("NN_Уверенность", 0)

        if pd.isna(risk_category) or risk_category == "":
            continue

        # Базовые рекомендации по категориям
        if risk_category == "A: Низкий риск":
            recommendation = {
                "ticker": ticker,
                "risk_level": "Низкий",
                "action": "Рассмотреть для консервативного портфеля",
                "allocation": "5-15%",
                "monitoring": "Ежеквартально",
                "confidence": confidence,
            }
        elif risk_category == "B: Средний риск":
            recommendation = {
                "ticker": ticker,
                "risk_level": "Средний",
                "action": "Подходит для сбалансированного портфеля",
                "allocation": "3-8%",
                "monitoring": "Ежемесячно",
                "confidence": confidence,
            }
        elif risk_category == "C: Высокий риск":
            recommendation = {
                "ticker": ticker,
                "risk_level": "Высокий",
                "action": "Только для агрессивных инвесторов",
                "allocation": "1-3%",
                "monitoring": "Еженедельно",
                "confidence": confidence,
            }
        elif risk_category == "D: Очень высокий риск":
            recommendation = {
                "ticker": ticker,
                "risk_level": "Очень высокий",
                "action": "Спекулятивная позиция, высокая осторожность",
                "allocation": "0-1%",
                "monitoring": "Ежедневно",
                "confidence": confidence,
            }
        else:
            continue

        # Добавляем дополнительные факторы
        beta = row.get("Бета", 1.0)
        if isinstance(beta, (int, float)):
            if beta > 1.5:
                recommendation["note"] = f"Высокая волатильность (Beta={beta:.2f})"
            elif beta < 0.5:
                recommendation["note"] = f"Низкая корреляция с рынком (Beta={beta:.2f})"

        recommendations.append(recommendation)

    return recommendations


# Пример использования
if __name__ == "__main__":
    try:
        # Проверяем версию TensorFlow
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")

        # Загрузка данных
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = f"{parent_dir}/../data/fundamentals_shares.xlsx"
        output_file = f"{parent_dir}/../data/risk_assessment_results.xlsx"
        print("\nЗагрузка данных...")
        df = pd.read_excel(input_file)
        print(f"Загружено {len(df)} записей")

        # Обучение нейросети
        df_with_risk, models, scaler = train_risk_assessment_ensemble(df, n_folds=3)

        # Получение рекомендаций
        recommendations = get_risk_recommendations(df_with_risk)

        # Сохранение результатов
        df_with_risk.to_excel(output_file, index=False)

        print(f"\n✅ Анализ рисков завершен!")
        print(f"✅ Результаты сохранены в risk_assessment_results.xlsx")
        print(f"✅ Проанализировано {len(recommendations)} акций")

    except Exception as e:
        print(f"\n❌ Ошибка при выполнении анализа: {str(e)}")
        import traceback

        traceback.print_exc()
