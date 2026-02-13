import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

# Импорт констант
from nn_risk_constants import (
    NN_ARCH,
    NN_FEATURE,
    NN_THRESHOLD,
    RISK_CAT,
    RISK_SCORE,
    NN_PORTFOLIO,
    NN_FEATURE_ALIASES,
    NN_FILES,
    NN_FORMAT,
    NN_REC,
)

warnings.filterwarnings("ignore")


# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


class NNRiskPathConfig:
    """Конфигурация путей к файлам для нейросетевого анализа рисков"""

    @staticmethod
    def setup_directories():
        """Создание необходимых директорий"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nn_risk_analyzer_dir = f"{parent_dir}/../data/nn_risk_analyzer"
        os.makedirs(nn_risk_analyzer_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "nn_risk_analyzer_dir": nn_risk_analyzer_dir,
            "input_file": f"{parent_dir}/../data/fundamentals_shares.xlsx",
            "nn_risk_portfolio_base": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_BASE}",
            "nn_risk_efficient_frontier": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_EFFICIENT_FRONTIER}",
            "nn_risk_portfolio_comparison": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_COMPARISON}",
            "nn_risk_portfolio_results": f"{nn_risk_analyzer_dir}/{NN_FILES.NN_RISK_PORTFOLIO_RESULTS}",
        }


NN_RISK_PATHS = NNRiskPathConfig.setup_directories()


# ==================== КЛАССЫ ДАННЫХ ====================


@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""

    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float
    var_95: float
    cvar_95: float


@dataclass
class RiskProfile:
    """Профиль риска компании"""

    ticker: str
    risk_category: str
    risk_score: float
    confidence: float
    expected_return: float
    volatility: float
    sharpe_candidate: float


# ==================== ФУНКЦИИ ДЛЯ НЕЙРОСЕТЕВОГО АНАЛИЗА ====================


def create_advanced_risk_assessment_nn(input_shape, num_classes=4):
    """Создание усовершенствованного ансамбля нейросетей для оценки риска"""
    models = []

    # 1. Основная сеть с регуляризацией
    model1 = keras.Sequential(
        [
            layers.Dense(
                NN_ARCH.DENSE_LAYER_1,
                activation="relu",
                input_shape=(input_shape,),
                kernel_regularizer=regularizers.l2(NN_ARCH.L2_REGULARIZER),
            ),
            layers.BatchNormalization(),
            layers.Dropout(NN_ARCH.DROPOUT_MEDIUM),
            layers.Dense(
                NN_ARCH.DENSE_LAYER_2,
                activation="relu",
                kernel_regularizer=regularizers.l2(NN_ARCH.L2_REGULARIZER),
            ),
            layers.BatchNormalization(),
            layers.Dropout(NN_ARCH.DROPOUT_LOW),
            layers.Dense(NN_ARCH.DENSE_LAYER_3, activation="relu"),
            layers.Dense(NN_ARCH.DENSE_LAYER_4, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    models.append(model1)

    # 2. Сеть с residual connections
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(NN_ARCH.RESIDUAL_LAYER_1, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(NN_ARCH.DROPOUT_HIGH)(x)
    residual = layers.Dense(NN_ARCH.RESIDUAL_LAYER_2, activation="relu")(x)
    x = layers.Dense(NN_ARCH.RESIDUAL_LAYER_2, activation="relu")(residual)
    x = layers.Add()([residual, x])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(NN_ARCH.DENSE_LAYER_2, activation="relu")(x)
    x = layers.Dropout(NN_ARCH.DROPOUT_MEDIUM)(x)
    x = layers.Dense(NN_ARCH.DENSE_LAYER_3, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model2 = keras.Model(inputs=inputs, outputs=outputs)
    models.append(model2)

    # 3. Широкая сеть
    model3 = keras.Sequential(
        [
            layers.Dense(
                NN_ARCH.WIDE_LAYER_1, activation="relu", input_shape=(input_shape,)
            ),
            layers.BatchNormalization(),
            layers.Dropout(NN_ARCH.DROPOUT_HIGH),
            layers.Dense(NN_ARCH.WIDE_LAYER_2, activation="relu"),
            layers.Dropout(NN_ARCH.DROPOUT_MEDIUM),
            layers.Dense(NN_ARCH.WIDE_LAYER_3, activation="relu"),
            layers.Dense(NN_ARCH.WIDE_LAYER_4, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    models.append(model3)

    return models


def prepare_features_for_nn(df):
    """Подготовка признаков для нейросети с использованием статистических методов"""

    all_potential_features = [
        "P/E",
        "P/B",
        "P/S",
        "P/FCF",
        "EV/EBITDA",
        "EV/S",
        "ROE",
        "ROA",
        "ROIC",
        "NPM",
        "EBITDA margin",
        "Debt/Capital",
        "Debt/EBITDA",
        "Net Debt/EBITDA",
        "dividend_yield",
        "EPS",
        "Beta",
        "Бета",
    ]

    available_features = []
    feature_aliases = NN_FEATURE_ALIASES.FEATURE_ALIASES

    for standard_name in all_potential_features:
        aliases = feature_aliases.get(standard_name, [standard_name])
        for alias in aliases:
            if alias in df.columns:
                available_features.append(standard_name)
                break

    available_features = list(dict.fromkeys(available_features))

    print(f"Используется {len(available_features)} признаков для нейросети:")
    for i, feat in enumerate(available_features):
        print(f"  {i+1}. {feat}")

    X = []
    tickers = []
    valid_indices = []
    feature_stats = {}

    for feature in available_features:
        values = []
        aliases = feature_aliases.get(feature, [feature])

        for alias in aliases:
            if alias in df.columns:
                vals = df[alias].dropna()
                for v in vals:
                    try:
                        values.append(float(v))
                    except:
                        pass

        if values:
            values = np.array(values)
            q1 = np.percentile(values, NN_FEATURE.Q1_PERCENTILE)
            q3 = np.percentile(values, NN_FEATURE.Q3_PERCENTILE)
            iqr = q3 - q1
            lower_bound = q1 - NN_FEATURE.IQR_MULTIPLIER * iqr
            upper_bound = q3 + NN_FEATURE.IQR_MULTIPLIER * iqr
            clean_values = values[(values >= lower_bound) & (values <= upper_bound)]

            if len(clean_values) > 0:
                feature_stats[feature] = {
                    "mean": np.mean(clean_values),
                    "median": np.median(clean_values),
                    "std": np.std(clean_values),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }
            else:
                feature_stats[feature] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }

    for idx, row in df.iterrows():
        feature_vector = []
        valid = True

        for feature in available_features:
            value = None
            aliases = feature_aliases.get(feature, [feature])

            for alias in aliases:
                if alias in row and pd.notna(row[alias]):
                    try:
                        value = float(row[alias])
                        break
                    except:
                        pass

            if value is None:
                value = feature_stats.get(feature, {}).get("median", 0)

            value = NNRiskUtils.clip_extreme_values(feature, value)
            feature_vector.append(value)

        if len(feature_vector) == len(available_features):
            X.append(feature_vector)
            tickers.append(row.get("Тикер", f"Row_{idx}"))
            valid_indices.append(idx)

    if not X:
        print("❌ Недостаточно данных для обучения нейросети")
        return None, None, None, None, None

    X = np.array(X)
    print(f"✅ Подготовлено {len(X)} акций с {X.shape[1]} признаками")

    return X, tickers, valid_indices, available_features, feature_stats


class NNRiskUtils:
    """Утилиты для нейросетевого анализа рисков"""

    @staticmethod
    def clip_extreme_values(feature: str, value: float) -> float:
        """Нормализация экстремальных значений"""
        if feature in ["P/E", "P/B", "P/S", "EV/EBITDA"]:
            if value > NN_FEATURE.PE_MAX:
                return NN_FEATURE.PE_MAX
            elif value < NN_FEATURE.PE_MIN:
                return NN_FEATURE.PE_MIN
        elif feature in ["ROE", "ROA", "ROIC", "NPM"]:
            if value > NN_FEATURE.ROE_MAX:
                return NN_FEATURE.ROE_MAX
            elif value < NN_FEATURE.ROE_MIN:
                return NN_FEATURE.ROE_MIN
        elif feature in ["Debt/Capital", "Debt/EBITDA"]:
            if value > NN_FEATURE.DEBT_CAPITAL_MAX:
                return NN_FEATURE.DEBT_CAPITAL_MAX
            elif value < NN_FEATURE.DEBT_CAPITAL_MIN:
                return NN_FEATURE.DEBT_CAPITAL_MIN
        return value

    @staticmethod
    def get_risk_level_for_feature(feature: str, value: float, stats: Dict) -> int:
        """Определение уровня риска для конкретного признака"""
        median = stats.get("median", 0)
        q1 = stats.get("q1", 0)
        q3 = stats.get("q3", 0)
        iqr = stats.get("iqr", 1)

        if feature in [
            "P/E",
            "P/B",
            "P/S",
            "P/FCF",
            "EV/EBITDA",
            "EV/S",
            "Debt/Capital",
            "Debt/EBITDA",
            "Net Debt/EBITDA",
        ]:
            if value <= NN_THRESHOLD.PE_RISK_FREE:
                return RISK_SCORE.RISK_HIGH
            elif value <= median * NN_THRESHOLD.PE_LOW_RISK:
                return RISK_SCORE.RISK_VERY_LOW
            elif value <= median * NN_THRESHOLD.PE_MEDIUM_RISK:
                return RISK_SCORE.RISK_LOW
            elif value <= median * NN_THRESHOLD.PE_HIGH_RISK:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        elif feature in [
            "ROE",
            "ROA",
            "ROIC",
            "NPM",
            "EBITDA margin",
            "dividend_yield",
            "EPS",
        ]:
            if value <= NN_THRESHOLD.PE_RISK_FREE:
                return RISK_SCORE.RISK_HIGH
            elif value >= median * NN_THRESHOLD.ROE_LOW_RISK:
                return RISK_SCORE.RISK_VERY_LOW
            elif value >= median * NN_THRESHOLD.ROE_MEDIUM_RISK:
                return RISK_SCORE.RISK_LOW
            elif value >= median * NN_THRESHOLD.ROE_HIGH_RISK:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        elif feature in ["Beta", "Бета"]:
            if value < NN_THRESHOLD.BETA_VERY_LOW:
                return RISK_SCORE.RISK_LOW
            elif value < NN_THRESHOLD.BETA_LOW:
                return RISK_SCORE.RISK_VERY_LOW
            elif value < NN_THRESHOLD.BETA_MEDIUM:
                return RISK_SCORE.RISK_LOW
            elif value < NN_THRESHOLD.BETA_HIGH:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

        else:
            deviation = abs(value - median) / (iqr if iqr > 0 else 1)
            if deviation < NN_THRESHOLD.DEVIATION_LOW:
                return RISK_SCORE.RISK_VERY_LOW
            elif deviation < NN_THRESHOLD.DEVIATION_MEDIUM:
                return RISK_SCORE.RISK_LOW
            elif deviation < NN_THRESHOLD.DEVIATION_HIGH:
                return RISK_SCORE.RISK_MEDIUM
            else:
                return RISK_SCORE.RISK_HIGH

    @staticmethod
    def get_feature_weight(feature: str) -> float:
        """Получение веса для признака"""
        if feature in RISK_SCORE.HIGH_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_HIGH
        elif feature in RISK_SCORE.MEDIUM_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_MEDIUM
        elif feature in RISK_SCORE.ABOVE_AVG_IMPORTANCE_FEATURES:
            return RISK_SCORE.WEIGHT_ABOVE_AVG
        else:
            return RISK_SCORE.WEIGHT_BASE


def calculate_risk_categories_statistical(
    df, X, feature_names, feature_stats, use_ae_scores=True
):
    """Рассчет категорий риска на основе статистических методов"""

    y = []
    category_details = []

    for i, idx in enumerate(df.index):
        if i >= len(X):
            break

        feature_vector = X[i]
        risk_factors = []
        risk_details = {}
        weighted_risks = []
        weights = []

        for j, feature in enumerate(feature_names):
            if j < len(feature_vector):
                value = feature_vector[j]
                stats = feature_stats.get(feature, {})

                risk_level = NNRiskUtils.get_risk_level_for_feature(
                    feature, value, stats
                )
                weight = NNRiskUtils.get_feature_weight(feature)

                risk_factors.append(risk_level)
                weights.append(weight)
                weighted_risks.append(risk_level * weight)
                risk_details[f"{feature}_risk"] = risk_level

        if use_ae_scores and "AE_Аномалия" in df.columns:
            try:
                ae_anomaly = df.loc[idx, "AE_Аномалия"] if idx in df.index else False
                if pd.notna(ae_anomaly) and ae_anomaly:
                    risk_factors.append(RISK_SCORE.RISK_HIGH)
                    weights.append(RISK_SCORE.AE_WEIGHT)
                    weighted_risks.append(RISK_SCORE.RISK_HIGH * RISK_SCORE.AE_WEIGHT)
                    risk_details["ae_risk"] = RISK_SCORE.RISK_HIGH
                else:
                    risk_factors.append(RISK_SCORE.RISK_VERY_LOW)
                    weights.append(RISK_SCORE.AE_WEIGHT)
                    weighted_risks.append(
                        RISK_SCORE.RISK_VERY_LOW * RISK_SCORE.AE_WEIGHT
                    )
                    risk_details["ae_risk"] = RISK_SCORE.RISK_VERY_LOW
            except:
                pass

        if weights:
            avg_risk = np.sum(weighted_risks) / np.sum(weights)
        else:
            avg_risk = RISK_SCORE.RISK_MEDIUM

        risk_details["avg_risk"] = avg_risk
        y.append(avg_risk)
        category_details.append(risk_details)

    y = np.array(y)

    p25 = np.percentile(y, NN_FEATURE.Q1_PERCENTILE) if len(y) > 0 else 1.0
    p50 = np.percentile(y, 50) if len(y) > 0 else 1.8
    p75 = np.percentile(y, NN_FEATURE.Q3_PERCENTILE) if len(y) > 0 else 2.5

    print(f"\n📊 Статистические пороги категорий риска:")
    print(f"  25-й процентиль: {p25:.3f}")
    print(f"  50-й процентиль (медиана): {p50:.3f}")
    print(f"  75-й процентиль: {p75:.3f}")

    y_categories = []
    for risk_score in y:
        if risk_score <= p25:
            category = RISK_CAT.RISK_A_CODE
        elif risk_score <= p50:
            category = RISK_CAT.RISK_B_CODE
        elif risk_score <= p75:
            category = RISK_CAT.RISK_C_CODE
        else:
            category = RISK_CAT.RISK_D_CODE
        y_categories.append(category)

    return np.array(y_categories), category_details


def train_risk_assessment_ensemble(df, n_folds=None, use_ae_results=True):
    """Обучение ансамбля нейросетей для оценки риска"""

    n_folds = n_folds or NN_ARCH.N_FOLDS

    print(NN_FORMAT.SEPARATOR)
    print("НАЧАЛО ОБУЧЕНИЯ НЕЙРОСЕТИ ДЛЯ ОЦЕНКИ РИСКА")
    print(NN_FORMAT.SEPARATOR)

    X, tickers, valid_indices, feature_names, feature_stats = prepare_features_for_nn(
        df
    )

    if X is None:
        print("❌ Ошибка подготовки данных!")
        return df, None, None

    y_categories, risk_details = calculate_risk_categories_statistical(
        df, X, feature_names, feature_stats, use_ae_results
    )

    if len(y_categories) > len(valid_indices):
        y_categories = y_categories[: len(valid_indices)]
    y_filtered = y_categories

    unique, counts = np.unique(y_filtered, return_counts=True)
    print("\n📊 Распределение категорий риска (на основе статистики):")

    actual_categories = {}
    for cat, count in zip(unique, counts):
        cat_name = RISK_CAT.CATEGORY_MAP.get(cat, f"Категория {cat}")
        actual_categories[cat] = cat_name
        print(f"  {cat_name}: {count} акций ({count/len(y_filtered)*100:.1f}%)")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    num_classes = len(actual_categories)
    print(f"\n🎯 Количество классов для классификации: {num_classes}")

    if num_classes < 2:
        print("❌ Недостаточно классов для классификации! Нужно минимум 2 класса.")
        return df, None, None

    class_mapping = {
        old: new for new, old in enumerate(sorted(actual_categories.keys()))
    }
    y_mapped = np.array([class_mapping[cat] for cat in y_filtered])
    y_categorical = to_categorical(y_mapped, num_classes=num_classes)

    actual_category_names = [
        actual_categories[old] for old in sorted(actual_categories.keys())
    ]

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=NN_ARCH.RANDOM_STATE
    )
    models = create_advanced_risk_assessment_nn(X.shape[1], num_classes=num_classes)

    ensemble_predictions = np.zeros_like(y_categorical)
    fold_metrics = []

    print("\n" + NN_FORMAT.SEPARATOR)
    print(f"НАЧИНАЕМ КРОСС-ВАЛИДАЦИЮ ({n_folds} фолдов)")
    print(NN_FORMAT.SEPARATOR)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_mapped)):
        print(f"\n🔹 Обучение fold {fold + 1}/{n_folds}")
        print(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

        fold_model_predictions = []
        fold_model_accuracies = []

        for i, model in enumerate(models):
            print(f"   🧠 Модель {i+1}/{len(models)}...", end=" ")

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=NN_ARCH.LEARNING_RATE),
                loss="categorical_crossentropy",
                metrics=["accuracy", Precision(), Recall()],
            )

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=NN_ARCH.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=NN_ARCH.REDUCE_LR_FACTOR,
                    patience=NN_ARCH.REDUCE_LR_PATIENCE,
                    min_lr=NN_ARCH.MIN_LR,
                    verbose=0,
                ),
            ]

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=NN_ARCH.EPOCHS,
                batch_size=NN_ARCH.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0,
            )

            val_loss, val_acc, val_precision, val_recall = model.evaluate(
                X_val, y_val, verbose=0
            )
            fold_model_accuracies.append(val_acc)

            val_pred = model.predict(X_val, verbose=0)
            fold_model_predictions.append(val_pred)

            print(f"Accuracy: {val_acc:.3f}")

        avg_pred = np.mean(fold_model_predictions, axis=0)
        ensemble_predictions[val_idx] = avg_pred

        fold_accuracy = np.mean(fold_model_accuracies)
        fold_metrics.append(
            {
                "fold": fold + 1,
                "accuracy": fold_accuracy,
                "model_accuracies": fold_model_accuracies,
            }
        )

    print("\n" + NN_FORMAT.SEPARATOR)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print(NN_FORMAT.SEPARATOR)

    avg_accuracy = np.mean([fm["accuracy"] for fm in fold_metrics])
    print(f"\n📈 Средняя точность ансамбля: {avg_accuracy:.3f}")

    print("\n🧠 Точность отдельных моделей:")
    for i, model_metrics in enumerate(
        zip(*[fm["model_accuracies"] for fm in fold_metrics])
    ):
        avg_model_acc = np.mean(model_metrics)
        print(f"  Модель {i+1}: {avg_model_acc:.3f}")

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

    df["NN_Категория_риска"] = np.nan
    df["NN_Уверенность"] = np.nan
    df["NN_Категория_текст"] = ""
    df["NN_Статистический_скор"] = np.nan

    reverse_class_mapping = {new: old for old, new in class_mapping.items()}

    for i, idx in enumerate(valid_indices):
        if i < len(y_pred):
            original_category = reverse_class_mapping.get(y_pred[i], y_pred[i])
            df.at[idx, "NN_Категория_риска"] = original_category
            df.at[idx, "NN_Уверенность"] = np.max(ensemble_predictions[i])
            df.at[idx, "NN_Категория_текст"] = RISK_CAT.CATEGORY_MAP.get(
                original_category, f"Категория {original_category}"
            )

            if i < len(y_categories):
                df.at[idx, "NN_Статистический_скор"] = y_categories[i]

    print("\n" + NN_FORMAT.SEPARATOR)
    print("РАСПРЕДЕЛЕНИЕ РИСКОВ ПО АКЦИЯМ")
    print(NN_FORMAT.SEPARATOR)

    predicted_categories = df["NN_Категория_текст"].value_counts()
    all_categories = [
        RISK_CAT.RISK_A_NAME,
        RISK_CAT.RISK_B_NAME,
        RISK_CAT.RISK_C_NAME,
        RISK_CAT.RISK_D_NAME,
    ]
    total_valid = predicted_categories.sum() if not predicted_categories.empty else 0

    for category in all_categories:
        count = predicted_categories.get(category, 0)
        percentage = count / total_valid * 100 if total_valid > 0 else 0
        print(f"{category:<30}: {count:>3} акций ({percentage:.1f}%)")

    print("\n" + NN_FORMAT.SEPARATOR)
    print("СТАТИСТИКА ПО УВЕРЕННОСТИ ПРЕДСКАЗАНИЙ")
    print(NN_FORMAT.SEPARATOR)

    confidence_scores = df["NN_Уверенность"].dropna()
    if len(confidence_scores) > 0:
        print(f"Средняя уверенность: {confidence_scores.mean():.3f}")
        print(f"Медианная уверенность: {confidence_scores.median():.3f}")
        print(f"Минимальная уверенность: {confidence_scores.min():.3f}")
        print(f"Максимальная уверенность: {confidence_scores.max():.3f}")

        print("\n📊 Распределение уверенности по квантилям:")
        quantiles = [0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            val = confidence_scores.quantile(q)
            print(f"  {q*100:.0f}-й процентиль: {val:.3f}")

    return df, models, scaler


def get_risk_recommendations_statistical(df):
    """Получение персональных рекомендаций на основе статистической оценки риска"""

    recommendations = []
    recommendation_map = {
        RISK_CAT.RISK_A_NAME: {
            "risk_level": RISK_CAT.RISK_A_DESC,
            "action": NN_REC.RISK_A_RECOMMENDATION,
            "allocation": NN_REC.RISK_A_ALLOCATION,
            "monitoring": NN_REC.RISK_A_MONITORING,
        },
        RISK_CAT.RISK_B_NAME: {
            "risk_level": RISK_CAT.RISK_B_DESC,
            "action": NN_REC.RISK_B_RECOMMENDATION,
            "allocation": NN_REC.RISK_B_ALLOCATION,
            "monitoring": NN_REC.RISK_B_MONITORING,
        },
        RISK_CAT.RISK_C_NAME: {
            "risk_level": RISK_CAT.RISK_C_DESC,
            "action": NN_REC.RISK_C_RECOMMENDATION,
            "allocation": NN_REC.RISK_C_ALLOCATION,
            "monitoring": NN_REC.RISK_C_MONITORING,
        },
        RISK_CAT.RISK_D_NAME: {
            "risk_level": RISK_CAT.RISK_D_DESC,
            "action": NN_REC.RISK_D_RECOMMENDATION,
            "allocation": NN_REC.RISK_D_ALLOCATION,
            "monitoring": NN_REC.RISK_D_MONITORING,
        },
    }

    for _, row in df.iterrows():
        ticker = row.get("Тикер", "Unknown")
        risk_category = row.get("NN_Категория_текст", "")
        confidence = row.get("NN_Уверенность", 0)
        statistical_score = row.get("NN_Статистический_скор", 2)

        if pd.isna(risk_category) or risk_category == "":
            continue

        rec_data = recommendation_map.get(risk_category)
        if rec_data is None:
            continue

        recommendation = {
            "ticker": ticker,
            "risk_level": rec_data["risk_level"],
            "statistical_score": (
                f"{statistical_score:.2f}"
                if not pd.isna(statistical_score)
                else NN_FORMAT.NA_STRING
            ),
            "action": rec_data["action"],
            "allocation": rec_data["allocation"],
            "monitoring": rec_data["monitoring"],
            "confidence": confidence,
        }

        if not pd.isna(statistical_score):
            if statistical_score > NN_THRESHOLD.STAT_SCORE_HIGH:
                recommendation["note"] = NN_REC.ANOMALY_NOTE

        recommendations.append(recommendation)

    return recommendations


# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================


class NNRiskPortfolioOptimizer:
    """Оптимизация портфеля на основе нейросетевой оценки риска"""

    def __init__(
        self,
        min_weight: float = None,
        max_weight: float = None,
        risk_free_rate: float = None,
    ):
        self.min_weight = min_weight or NN_PORTFOLIO.MIN_WEIGHT
        self.max_weight = max_weight or NN_PORTFOLIO.MAX_WEIGHT
        self.risk_free_rate = risk_free_rate or NN_PORTFOLIO.RISK_FREE_RATE

    def calculate_expected_return(self, row: pd.Series) -> float:
        """Расчет ожидаемой доходности с учетом нейросетевой оценки риска"""
        base_return = NN_PORTFOLIO.BASE_RETURN

        risk_category = row.get("NN_Категория_текст", "")
        risk_premium_map = {
            RISK_CAT.RISK_A_NAME: NN_PORTFOLIO.RISK_A_PREMIUM,
            RISK_CAT.RISK_B_NAME: NN_PORTFOLIO.RISK_B_PREMIUM,
            RISK_CAT.RISK_C_NAME: NN_PORTFOLIO.RISK_C_PREMIUM,
            RISK_CAT.RISK_D_NAME: NN_PORTFOLIO.RISK_D_PREMIUM,
        }
        base_return += risk_premium_map.get(risk_category, 0)

        if pd.notna(row.get("ROE")):
            if row["ROE"] > NN_PORTFOLIO.ROE_HIGH_THRESHOLD:
                base_return += NN_PORTFOLIO.ROE_HIGH_PREMIUM
            elif row["ROE"] > NN_PORTFOLIO.ROE_MEDIUM_THRESHOLD:
                base_return += NN_PORTFOLIO.ROE_MEDIUM_PREMIUM

        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row.get("Дивидендная доходность", 0)
            base_return += div_yield / 100 * NN_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR

        confidence = row.get("NN_Уверенность", 0.5)
        base_return *= (
            NN_PORTFOLIO.CONFIDENCE_SCALE
            + NN_PORTFOLIO.CONFIDENCE_MAX_FACTOR * confidence
        )

        return min(NN_PORTFOLIO.MAX_RETURN, max(NN_PORTFOLIO.MIN_RETURN, base_return))

    def calculate_volatility(self, row: pd.Series) -> float:
        """Расчет волатильности на основе нейросетевой оценки"""
        risk_category = row.get("NN_Категория_текст", "")

        volatility_map = {
            RISK_CAT.RISK_A_NAME: NN_PORTFOLIO.VOLATILITY_A,
            RISK_CAT.RISK_B_NAME: NN_PORTFOLIO.VOLATILITY_B,
            RISK_CAT.RISK_C_NAME: NN_PORTFOLIO.VOLATILITY_C,
            RISK_CAT.RISK_D_NAME: NN_PORTFOLIO.VOLATILITY_D,
        }

        base_vol = volatility_map.get(risk_category, NN_PORTFOLIO.BASE_VOLATILITY)

        if pd.notna(row.get("Бета")):
            beta = row.get("Бета", 1)
            base_vol *= (
                NN_PORTFOLIO.BETA_VOL_FACTOR_MIN
                + NN_PORTFOLIO.BETA_VOL_FACTOR_MAX * beta
            )

        if pd.notna(row.get("Debt/Capital")):
            debt = row.get("Debt/Capital", 0)
            base_vol *= (
                NN_PORTFOLIO.DEBT_VOL_FACTOR_MIN
                + NN_PORTFOLIO.DEBT_VOL_FACTOR_MAX
                * (debt / NN_PORTFOLIO.DEBT_NORMALIZATION)
            )

        return min(
            NN_PORTFOLIO.MAX_VOLATILITY, max(NN_PORTFOLIO.MIN_VOLATILITY, base_vol)
        )

    def create_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Создание матрицы ковариации с учетом корреляций"""
        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["NN_Volatility"].values
        risk_categories = df.get("NN_Категория_текст", ["Unknown"] * n).values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    corr = (
                        NN_PORTFOLIO.INTRA_CATEGORY_CORRELATION
                        if risk_categories[i] == risk_categories[j]
                        else NN_PORTFOLIO.INTER_CATEGORY_CORRELATION
                    )
                    cov_matrix[i, j] = corr * risks[i] * risks[j]

        return cov_matrix

    def calculate_portfolio_metrics(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        port_return = np.sum(expected_returns * weights)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

        var_95 = port_return - NN_PORTFOLIO.VAR_95_COEFF * port_risk
        cvar_95 = port_return - NN_PORTFOLIO.CVAR_95_COEFF * port_risk

        hhi = np.sum(weights**2)
        n = len(weights)
        diversification = 1 - (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 0

        return PortfolioMetrics(
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def optimize_portfolio(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> Dict:
        """Оптимизация портфеля с несколькими целями"""
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return (
                -(port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
            )

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def negative_return(weights):
            return -np.sum(expected_returns * weights)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        init_guess = np.array([1 / n] * n)

        result_sharpe = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_min_risk = minimize(
            portfolio_risk,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        result_max_return = minimize(
            negative_return,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": NN_PORTFOLIO.OPTIMIZER_MAX_ITER},
        )

        combined_weights = (
            NN_PORTFOLIO.SHARPE_PORTFOLIO_WEIGHT * result_sharpe.x
            + NN_PORTFOLIO.MIN_RISK_PORTFOLIO_WEIGHT * result_min_risk.x
            + NN_PORTFOLIO.MAX_RETURN_PORTFOLIO_WEIGHT * result_max_return.x
        )
        combined_weights = combined_weights / combined_weights.sum()

        return {
            "sharpe_weights": result_sharpe.x,
            "min_risk_weights": result_min_risk.x,
            "max_return_weights": result_max_return.x,
            "combined_weights": combined_weights,
            "cov_matrix": cov_matrix,
        }

    def optimize_risk_based_portfolios(self, df: pd.DataFrame) -> Dict:
        """Создание портфелей на основе категорий риска"""
        portfolios = {}

        conservative = df[
            df["NN_Категория_текст"].isin([RISK_CAT.RISK_A_NAME, RISK_CAT.RISK_B_NAME])
        ].copy()
        if len(conservative) > 0:
            weights = self._equal_weight_by_category(conservative)
            portfolios["Консервативный"] = (conservative, weights)

        balanced = df.copy()
        if len(balanced) > 0:
            weights = self._risk_weighted_allocation(balanced)
            portfolios["Сбалансированный"] = (balanced, weights)

        aggressive = df[
            df["NN_Категория_текст"].isin([RISK_CAT.RISK_C_NAME, RISK_CAT.RISK_D_NAME])
        ].copy()
        if len(aggressive) > 0:
            weights = self._return_weighted_allocation(aggressive)
            portfolios["Агрессивный"] = (aggressive, weights)

        dividend = df[
            df["Дивидендная доходность"].fillna(0)
            > NN_PORTFOLIO.DIVIDEND_PREMIUM_FACTOR * 10
        ].copy()
        if len(dividend) > 0:
            weights = self._dividend_weighted_allocation(dividend)
            portfolios["Дивидендный"] = (dividend, weights)

        confidence = df[df["NN_Уверенность"] > NN_PORTFOLIO.CONFIDENCE_SCALE].copy()
        if len(confidence) > 0:
            weights = self._confidence_weighted_allocation(confidence)
            portfolios["Нейросетевой"] = (confidence, weights)

        return portfolios

    def _equal_weight_by_category(self, df: pd.DataFrame) -> np.ndarray:
        """Равномерное распределение по категориям риска"""
        n = len(df)
        weights = np.ones(n) / n
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _risk_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе обратного риска"""
        risks = 1 / df["NN_Volatility"].values
        weights = risks / risks.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _return_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе ожидаемой доходности"""
        returns = df["NN_Expected_Return"].values
        weights = returns / returns.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _dividend_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе дивидендной доходности"""
        div_yield = df["Дивидендная доходность"].fillna(0).values
        weights = div_yield / div_yield.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _confidence_weighted_allocation(self, df: pd.DataFrame) -> np.ndarray:
        """Распределение на основе уверенности нейросети"""
        confidence = df["NN_Уверенность"].values
        weights = confidence / confidence.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()


# ==================== КЛАСС УПРАВЛЕНИЯ ПОРТФЕЛЕМ ====================


class NNRiskPortfolioManager:
    """Управление портфелем на основе нейросетевой оценки риска"""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        weights: np.ndarray,
        optimizer: NNRiskPortfolioOptimizer,
    ):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.optimizer = optimizer

        expected_returns = self.df["NN_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        self.metrics = optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        if "Сектор" in self.df.columns:
            return self.df.groupby("Сектор")["Weight"].sum()
        return pd.Series()

    def get_risk_category_allocation(self) -> pd.Series:
        """Распределение по категориям риска"""
        if "NN_Категория_текст" in self.df.columns:
            return self.df.groupby("NN_Категория_текст")["Weight"].sum()
        return pd.Series()

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or NN_FORMAT.TOP_POSITIONS_SUMMARY
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()

    def get_risk_contribution(self) -> pd.Series:
        """Вклад в риск портфеля"""
        if len(self.df) == 0:
            return pd.Series()

        weights = self.weights
        risks = self.df["NN_Volatility"].values
        risk_contribution = weights * risks / np.sum(weights * risks)

        tickers = self.df.get("Тикер", self.df.index)
        return pd.Series(risk_contribution, index=tickers)


# ==================== КЛАСС ВИЗУАЛИЗАЦИИ ====================


class NNRiskPortfolioVisualizer:
    """Визуализация портфелей на основе нейросетевой оценки риска"""

    @staticmethod
    def plot_portfolio_summary(
        portfolio_manager: NNRiskPortfolioManager,
        filename: str = None,
    ):
        """Сводная визуализация портфеля"""
        if filename is None:
            filename = f"{NN_RISK_PATHS['nn_risk_portfolio_base']}_{portfolio_manager.name}.png"

        fig, axes = plt.subplots(2, 2, figsize=NN_FILES.FIGURE_SIZE_SUMMARY)

        NNRiskPortfolioVisualizer._plot_risk_allocation(portfolio_manager, axes[0, 0])
        NNRiskPortfolioVisualizer._plot_risk_return_scatter(
            portfolio_manager, axes[0, 1]
        )
        NNRiskPortfolioVisualizer._plot_risk_contribution(portfolio_manager, axes[1, 0])
        NNRiskPortfolioVisualizer._plot_portfolio_metrics(portfolio_manager, axes[1, 1])

        plt.suptitle(
            f"Портфель на основе нейросетевой оценки риска: {portfolio_manager.name}",
            fontsize=NN_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_allocation(pm: NNRiskPortfolioManager, ax):
        """Визуализация распределения по категориям риска"""
        risk_allocation = pm.get_risk_category_allocation()

        if len(risk_allocation) > 0:
            colors = {
                RISK_CAT.RISK_A_NAME: NN_FORMAT.COLOR_RISK_A,
                RISK_CAT.RISK_B_NAME: NN_FORMAT.COLOR_RISK_B,
                RISK_CAT.RISK_C_NAME: NN_FORMAT.COLOR_RISK_C,
                RISK_CAT.RISK_D_NAME: NN_FORMAT.COLOR_RISK_D,
            }

            plot_colors = [
                colors.get(cat, NN_FORMAT.COLOR_RISK_DEFAULT)
                for cat in risk_allocation.index
            ]

            ax.pie(
                risk_allocation.values,
                labels=risk_allocation.index,
                autopct=NN_FORMAT.MATPLOTLIB_PERCENT,
                startangle=90,
                colors=plot_colors,
                explode=[NN_FORMAT.PIE_EXPLODE_FACTOR] * len(risk_allocation),
            )
            ax.set_title(
                "Распределение по категориям риска",
                fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_risk_return_scatter(pm: NNRiskPortfolioManager, ax):
        """Визуализация риск-доходность позиций"""
        scatter = ax.scatter(
            pm.df["NN_Volatility"],
            pm.df["NN_Expected_Return"],
            s=pm.weights * 3000,
            c=pm.df["NN_Уверенность"],
            cmap=NN_FORMAT.COLOR_CONFIDENCE_CMAP,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        top_n = min(NN_FORMAT.TOP_POSITIONS_SUMMARY, len(pm.df))
        top_positions = pm.get_top_positions(top_n)

        for _, row in top_positions.iterrows():
            ax.annotate(
                row.get("Тикер", "N/A"),
                (row["NN_Volatility"], row["NN_Expected_Return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=NN_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.scatter(
            pm.metrics.risk,
            pm.metrics.expected_return,
            s=300,
            c=NN_FORMAT.COLOR_OPTIMAL_MARKER,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label=f"Портфель (Шарп: {pm.metrics.sharpe_ratio:.2f})",
        )

        ax.set_xlabel("Волатильность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Ожидаемая доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Risk-Return профиль позиций",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label="Уверенность нейросети")

    @staticmethod
    def _plot_risk_contribution(pm: NNRiskPortfolioManager, ax):
        """Визуализация вклада в риск портфеля"""
        risk_contrib = pm.get_risk_contribution()

        if len(risk_contrib) > 0:
            top_n = min(NN_FORMAT.TOP_RISK_CONTRIBUTION, len(risk_contrib))
            top_risk = risk_contrib.nlargest(top_n)

            colors = plt.cm.get_cmap(NN_FORMAT.COLOR_RISK_CONTRIBUTION_CMAP)(
                np.linspace(0.2, 0.8, len(top_risk))
            )
            bars = ax.barh(
                range(len(top_risk)), top_risk.values, color=colors, edgecolor="black"
            )

            ax.set_yticks(range(len(top_risk)))
            ax.set_yticklabels(top_risk.index)
            ax.set_xlabel("Вклад в риск портфеля", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
            ax.set_title(
                "Распределение риска по акциям",
                fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )
            ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="x")

            for bar, value in zip(bars, top_risk.values):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    ha="left",
                    va="center",
                    fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                )

    @staticmethod
    def _plot_portfolio_metrics(pm: NNRiskPortfolioManager, ax):
        """Визуализация метрик портфеля"""
        ax.axis("off")

        risk_alloc = pm.get_risk_category_allocation()
        risk_alloc_str = ""
        for cat, weight in risk_alloc.items():
            risk_alloc_str += f"{cat}: {weight:.1%}\n"

        metrics_text = f"""
        📊 МЕТРИКИ ПОРТФЕЛЯ: {pm.name}
        
        Ожидаемая доходность: {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.expected_return)}
        Риск (волатильность): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.risk)}
        Коэффициент Шарпа: {NN_FORMAT.FLOAT_FORMAT_2D.format(pm.metrics.sharpe_ratio)}
        
        📈 РИСК-МЕТРИКИ
        VaR (95%): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.var_95)}
        CVaR (95%): {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.cvar_95)}
        
        📊 ДИВЕРСИФИКАЦИЯ
        Индекс диверсификации: {NN_FORMAT.PERCENT_FORMAT.format(pm.metrics.diversification_score)}
        Количество позиций: {len(pm.df)}
        Макс. доля: {NN_FORMAT.PERCENT_FORMAT.format(pm.weights.max())}
        
        🤖 НЕЙРОСЕТЕВАЯ ОЦЕНКА
        Средняя уверенность: {NN_FORMAT.PERCENT_FORMAT.format(pm.df['NN_Уверенность'].mean())}
        {risk_alloc_str}
        """

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=NN_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=NN_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=NN_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_portfolio_comparison(
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Сравнение различных портфелей"""
        if not portfolios:
            return

        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_portfolio_comparison"]

        fig, axes = plt.subplots(2, 2, figsize=NN_FILES.FIGURE_SIZE_COMPARISON)

        names = []
        returns = []
        risks = []
        sharpes = []
        var_95s = []

        for name, pm in portfolios.items():
            names.append(name)
            returns.append(pm.metrics.expected_return)
            risks.append(pm.metrics.risk)
            sharpes.append(pm.metrics.sharpe_ratio)
            var_95s.append(pm.metrics.var_95)

        NNRiskPortfolioVisualizer._plot_risk_return_comparison(
            axes[0, 0], names, returns, risks, sharpes
        )
        NNRiskPortfolioVisualizer._plot_sharpe_comparison(axes[0, 1], names, sharpes)
        NNRiskPortfolioVisualizer._plot_var_comparison(axes[1, 0], names, var_95s)
        NNRiskPortfolioVisualizer._plot_best_portfolio_summary(axes[1, 1], portfolios)

        plt.suptitle(
            "Сравнение портфелей на основе нейросетевой оценки риска",
            fontsize=NN_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_return_comparison(ax, names, returns, risks, sharpes):
        """Визуализация сравнения риск-доходность"""
        scatter = ax.scatter(
            risks,
            returns,
            c=sharpes,
            s=300,
            cmap=NN_FORMAT.COLOR_CONFIDENCE_CMAP,
            edgecolors="black",
            linewidths=1.5,
        )

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=NN_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Риск", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение портфелей: Risk-Return",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Коэффициент Шарпа", fontsize=NN_FORMAT.LABEL_FONT_SIZE)

    @staticmethod
    def _plot_sharpe_comparison(ax, names, sharpes):
        """Визуализация сравнения коэффициентов Шарпа"""
        sharpe_array = np.array(sharpes)
        sharpe_range = sharpe_array.max() - sharpe_array.min() + 0.001

        colors = plt.cm.get_cmap(NN_FORMAT.COLOR_CONFIDENCE_CMAP)(
            (sharpe_array - sharpe_array.min()) / sharpe_range
        )

        bars = ax.bar(names, sharpes, color=colors, edgecolor="black")
        ax.axhline(
            y=1,
            color=NN_FORMAT.COLOR_OPTIMAL_MARKER,
            linestyle="--",
            alpha=0.5,
            label="Целевой Шарп = 1",
        )
        ax.set_xlabel("Портфель", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Коэффициент Шарпа", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение коэффициентов Шарпа",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="y")

        for bar, value in zip(bars, sharpes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_var_comparison(ax, names, var_95s):
        """Визуализация сравнения Value at Risk"""
        bars = ax.bar(names, var_95s, color=NN_FORMAT.COLOR_RISK_B, edgecolor="black")
        ax.set_xlabel("Портфель", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("VaR (95%)", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение Value at Risk",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10, axis="y")

        for bar, value in zip(bars, var_95s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.01,
                f"{value:.1%}",
                ha="center",
                va="top",
                fontsize=NN_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
                color="white",
            )

    @staticmethod
    def _plot_best_portfolio_summary(ax, portfolios):
        """Визуализация информации о лучшем портфеле"""
        ax.axis("off")

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        top_n = min(NN_FORMAT.TOP_POSITIONS_BEST, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        text = f"🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {best_portfolio.name}\n\n"
        text += f"Доходность: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.expected_return)}\n"
        text += (
            f"Риск: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.risk)}\n"
        )
        text += f"Шарп: {NN_FORMAT.FLOAT_FORMAT_2D.format(best_portfolio.metrics.sharpe_ratio)}\n"
        text += (
            f"VaR: {NN_FORMAT.PERCENT_FORMAT.format(best_portfolio.metrics.var_95)}\n\n"
        )
        text += f"📈 ТОП-{top_n} ПОЗИЦИЙ:\n"

        for _, row in top_positions.iterrows():
            ticker = row.get("Тикер", "N/A")
            weight = row.get("Weight", 0)
            text += f"• {ticker}: {NN_FORMAT.PERCENT_FORMAT.format(weight)}\n"
            risk_cat = row.get("NN_Категория_текст", "N/A")
            if risk_cat != "N/A":
                text += f"  {risk_cat}\n"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=NN_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=NN_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=NN_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_efficient_frontier(
        optimizer: NNRiskPortfolioOptimizer,
        df: pd.DataFrame,
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Построение границы эффективности"""
        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_efficient_frontier"]

        expected_returns = df["NN_Expected_Return"].values
        cov_matrix = optimizer.create_covariance_matrix(df)

        n_portfolios = NN_FILES.N_EFFICIENT_PORTFOLIOS
        n_assets = len(df)

        returns = []
        risks = []
        sharpes = []

        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()

            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            port_sharpe = (
                (port_return - optimizer.risk_free_rate) / port_risk
                if port_risk > 0
                else 0
            )

            returns.append(port_return)
            risks.append(port_risk)
            sharpes.append(port_sharpe)

        plt.figure(figsize=NN_FILES.FIGURE_SIZE_FRONTIER)

        scatter = plt.scatter(
            risks,
            returns,
            c=sharpes,
            cmap=NN_FORMAT.COLOR_EFFICIENT_CMAP,
            alpha=0.2,
            s=10,
        )

        plt.colorbar(scatter, label="Коэффициент Шарпа")

        colors = plt.cm.get_cmap(NN_FORMAT.COLOR_PORTFOLIO_CMAP)(
            np.linspace(0, 1, len(portfolios))
        )

        for (name, pm), color in zip(portfolios.items(), colors):
            plt.scatter(
                pm.metrics.risk,
                pm.metrics.expected_return,
                s=300,
                c=[color],
                marker="*",
                edgecolors="black",
                linewidths=2,
                label=f"{name} (Шарп: {pm.metrics.sharpe_ratio:.2f})",
            )

        market_weights = np.array([1 / n_assets] * n_assets)
        market_return = np.sum(expected_returns * market_weights)
        market_risk = np.sqrt(
            np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        )

        plt.scatter(
            market_risk,
            market_return,
            s=200,
            c=NN_FORMAT.COLOR_MARKET_MARKER,
            marker="s",
            edgecolors="black",
            linewidths=2,
            label="Равновзвешенный",
        )

        plt.xlabel("Риск (волатильность)", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        plt.ylabel("Ожидаемая доходность", fontsize=NN_FORMAT.AXIS_FONT_SIZE)
        plt.title(
            "Граница эффективности Марковица",
            fontsize=NN_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=NN_FEATURE.IQR_MULTIPLIER / 10)

        plt.tight_layout()
        plt.savefig(filename, dpi=NN_FILES.DPI, bbox_inches="tight")
        plt.show()


# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================


class NNRiskReportGenerator:
    """Генерация отчетов для нейросетевого анализа рисков"""

    @staticmethod
    def generate_full_report(
        df_with_risk: pd.DataFrame,
        candidates: pd.DataFrame,
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Генерация полного отчета"""
        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_portfolio_results"]

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_with_risk.to_excel(
                writer, sheet_name=NN_FILES.SHEET_STOCKS_WITH_RISK, index=False
            )

            candidates.to_excel(
                writer, sheet_name=NN_FILES.SHEET_CANDIDATES, index=False
            )

            if portfolios:
                NNRiskReportGenerator._write_portfolio_summary(writer, portfolios)

                best_portfolio = max(
                    portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
                )
                best_portfolio.df.to_excel(
                    writer, sheet_name=NN_FILES.SHEET_BEST_PORTFOLIO, index=False
                )

        print(f"   ✅ Результаты сохранены в {filename}")

    @staticmethod
    def _write_portfolio_summary(writer, portfolios):
        """Запись сводки по портфелям"""
        portfolio_summary = []
        for name, pm in portfolios.items():
            portfolio_summary.append(
                {
                    "Портфель": name,
                    "Доходность": f"{pm.metrics.expected_return:.2%}",
                    "Риск": f"{pm.metrics.risk:.2%}",
                    "Шарп": f"{pm.metrics.sharpe_ratio:.2f}",
                    "VaR": f"{pm.metrics.var_95:.2%}",
                    "CVaR": f"{pm.metrics.cvar_95:.2%}",
                    "Диверсификация": f"{pm.metrics.diversification_score:.1%}",
                    "Позиций": len(pm.df),
                }
            )

        pd.DataFrame(portfolio_summary).to_excel(
            writer, sheet_name=NN_FILES.SHEET_PORTFOLIO_SUMMARY, index=False
        )


# ==================== ИНТЕГРИРОВАННЫЙ ПАЙПЛАЙН ====================


def train_and_optimize_portfolio():
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

    print("\n🧠 Обучение нейросетевой модели оценки риска...")
    df_with_risk, models, scaler = train_risk_assessment_ensemble(
        df, n_folds=NN_ARCH.N_FOLDS
    )

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
            portfolios, NN_RISK_PATHS["nn_risk_portfolio_comparison"]
        )

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        NNRiskPortfolioVisualizer.plot_portfolio_summary(best_portfolio)

        NNRiskPortfolioVisualizer.plot_efficient_frontier(
            optimizer,
            candidates,
            portfolios,
            NN_RISK_PATHS["nn_risk_efficient_frontier"],
        )

    print("\n📄 Сохранение результатов...")

    NNRiskReportGenerator.generate_full_report(
        df_with_risk, candidates, portfolios, NN_RISK_PATHS["nn_risk_portfolio_results"]
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

    return df_with_risk, portfolios


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    try:
        df_results, portfolios = train_and_optimize_portfolio()
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении анализа: {str(e)}")
        import traceback

        traceback.print_exc()
