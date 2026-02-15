from typing import Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from ...ai_risk_models.ai_risk_constants.ai_risk_constants import (
    NN_ARCH,
    NN_FEATURE,
    NN_FEATURE_ALIASES,
    NN_FORMAT,
    NN_REC,
    NN_THRESHOLD,
    RISK_CAT,
    RISK_SCORE,
)


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


class NeuralRiskAssessor:
    """Класс для нейросетевой оценки рисков акций"""

    def __init__(self, n_folds=None):
        self.n_folds = n_folds or NN_ARCH.N_FOLDS
        self.models = None
        self.scaler = None
        self.feature_names = None
        self.feature_stats = None
        self.class_mapping = None
        self.actual_category_names = None

    def create_advanced_risk_assessment_nn(self, input_shape, num_classes=4):
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

    def prepare_features_for_nn(self, df):
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

                value = self.clip_extreme_values(feature, value)
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
        self, df, X, feature_names, feature_stats, use_ae_scores=True
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

                    risk_level = self.get_risk_level_for_feature(feature, value, stats)
                    weight = self.get_feature_weight(feature)

                    risk_factors.append(risk_level)
                    weights.append(weight)
                    weighted_risks.append(risk_level * weight)
                    risk_details[f"{feature}_risk"] = risk_level

            if use_ae_scores and "AE_Аномалия" in df.columns:
                try:
                    ae_anomaly = (
                        df.loc[idx, "AE_Аномалия"] if idx in df.index else False
                    )
                    if pd.notna(ae_anomaly) and ae_anomaly:
                        risk_factors.append(RISK_SCORE.RISK_HIGH)
                        weights.append(RISK_SCORE.AE_WEIGHT)
                        weighted_risks.append(
                            RISK_SCORE.RISK_HIGH * RISK_SCORE.AE_WEIGHT
                        )
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

    def train_risk_assessment_ensemble(self, df, use_ae_results=True):
        """Обучение ансамбля нейросетей для оценки риска"""

        print(NN_FORMAT.SEPARATOR)
        print("НАЧАЛО ОБУЧЕНИЯ НЕЙРОСЕТИ ДЛЯ ОЦЕНКИ РИСКА")
        print(NN_FORMAT.SEPARATOR)

        X, tickers, valid_indices, feature_names, feature_stats = (
            self.prepare_features_for_nn(df)
        )

        if X is None:
            print("❌ Ошибка подготовки данных!")
            return df, None, None

        self.feature_names = feature_names
        self.feature_stats = feature_stats

        y_categories, risk_details = self.calculate_risk_categories_statistical(
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
        self.scaler = scaler

        num_classes = len(actual_categories)
        print(f"\n🎯 Количество классов для классификации: {num_classes}")

        if num_classes < 2:
            print("❌ Недостаточно классов для классификации! Нужно минимум 2 класса.")
            return df, None, None

        self.class_mapping = {
            old: new for new, old in enumerate(sorted(actual_categories.keys()))
        }
        y_mapped = np.array([self.class_mapping[cat] for cat in y_filtered])
        y_categorical = to_categorical(y_mapped, num_classes=num_classes)

        self.actual_category_names = [
            actual_categories[old] for old in sorted(actual_categories.keys())
        ]

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=NN_ARCH.RANDOM_STATE
        )
        models = self.create_advanced_risk_assessment_nn(
            X.shape[1], num_classes=num_classes
        )
        self.models = models

        ensemble_predictions = np.zeros_like(y_categorical)
        fold_metrics = []

        print("\n" + NN_FORMAT.SEPARATOR)
        print(f"НАЧИНАЕМ КРОСС-ВАЛИДАЦИЮ ({self.n_folds} фолдов)")
        print(NN_FORMAT.SEPARATOR)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_mapped)):
            print(f"\n🔹 Обучение fold {fold + 1}/{self.n_folds}")
            print(f"   Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

            fold_model_predictions = []
            fold_model_accuracies = []

            for i, model in enumerate(models):
                print(f"   🧠 Модель {i+1}/{len(models)}...", end=" ")

                model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=NN_ARCH.LEARNING_RATE
                    ),
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
                    y_true, y_pred, target_names=self.actual_category_names, digits=3
                )
            )
        except:
            print("Не удалось создать classification report")

        df["NN_Категория_риска"] = np.nan
        df["NN_Уверенность"] = np.nan
        df["NN_Категория_текст"] = ""
        df["NN_Статистический_скор"] = np.nan

        reverse_class_mapping = {new: old for old, new in self.class_mapping.items()}

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
        total_valid = (
            predicted_categories.sum() if not predicted_categories.empty else 0
        )

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

        return df, self.models, self.scaler

    def get_risk_recommendations_statistical(self, df):
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

    def predict_risk(self, df, use_ae_results=True):
        """Предсказание риска для новых данных с использованием обученной модели"""

        if self.models is None or self.scaler is None:
            print(
                "❌ Модель не обучена! Сначала вызовите train_risk_assessment_ensemble()"
            )
            return df, None

        X, tickers, valid_indices, feature_names, feature_stats = (
            self.prepare_features_for_nn(df)
        )

        if X is None:
            print("❌ Ошибка подготовки данных!")
            return df, None

        X_scaled = self.scaler.transform(X)

        # Получение предсказаний от всех моделей ансамбля
        all_predictions = []
        for model in self.models:
            pred = model.predict(X_scaled, verbose=0)
            all_predictions.append(pred)

        ensemble_predictions = np.mean(all_predictions, axis=0)
        y_pred = np.argmax(ensemble_predictions, axis=1)

        # Добавление результатов в DataFrame
        df["NN_Категория_риска"] = np.nan
        df["NN_Уверенность"] = np.nan
        df["NN_Категория_текст"] = ""

        reverse_class_mapping = {new: old for old, new in self.class_mapping.items()}

        for i, idx in enumerate(valid_indices):
            if i < len(y_pred):
                original_category = reverse_class_mapping.get(y_pred[i], y_pred[i])
                df.at[idx, "NN_Категория_риска"] = original_category
                df.at[idx, "NN_Уверенность"] = np.max(ensemble_predictions[i])
                df.at[idx, "NN_Категория_текст"] = RISK_CAT.CATEGORY_MAP.get(
                    original_category, f"Категория {original_category}"
                )

        return df, ensemble_predictions
