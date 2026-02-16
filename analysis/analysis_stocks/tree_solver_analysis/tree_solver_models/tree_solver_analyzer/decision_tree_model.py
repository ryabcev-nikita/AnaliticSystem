# ==================== КЛАСС МОДЕЛИ ДЕРЕВА РЕШЕНИЙ ====================
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from ...tree_solver_models.tree_solver_constants.tree_solver_constants import (
    FILE_CONSTANTS,
    FORMATTING,
    MODEL_CONSTANTS,
    TARGET_MAPPING,
)
from ...tree_solver_models.tree_solver_market.market_analyzer import MarketAnalyzer


class DecisionTreeModel:
    """
    Модель дерева решений для оценки акций на основе фундаментальных показателей:
    - P/E (Цена/Прибыль)
    - P/BV (Цена/Балансовая стоимость)
    - ROE (Рентабельность собственного капитала)
    - g (Темп роста, рассчитывается в DataLoader)
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ["P/E", "P/BV", "ROE", "g"]  # Темп роста из DataLoader

        # Для обратной совместимости с разными названиями колонок
        self.column_mapping = {
            "P/B": "P/BV",  # Если в данных P/B, используем как P/BV
        }

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Маппинг названий колонок для совместимости"""
        df = df.copy()

        # P/BV может называться по-разному
        if "P/B" in df.columns and "P/BV" not in df.columns:
            df["P/BV"] = df["P/B"]
        elif "P/BV" in df.columns and "P/B" not in df.columns:
            pass  # Уже есть нужная колонка
        elif "P/B" not in df.columns and "P/BV" not in df.columns:
            print("   ⚠️ Внимание: Отсутствуют данные P/B или P/BV")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        df = df.copy()

        # Маппинг колонок
        df = self._map_columns(df)

        # Проверяем наличие всех необходимых признаков
        available_features = []
        missing_features = []

        for feat in self.feature_columns:
            if feat in df.columns:
                available_features.append(feat)
            else:
                missing_features.append(feat)

        if missing_features:
            print(f"   ⚠️ Отсутствуют признаки: {', '.join(missing_features)}")
            # Используем только доступные признаки
            self.feature_columns = available_features

        if not available_features:
            raise ValueError("Нет доступных признаков для обучения модели")

        # Кодирование сектора (опционально, для анализа)
        if "Название" in df.columns:
            df["Сектор"] = df["Название"].apply(MarketAnalyzer.assign_sector)
            df["Сектор_encoded"] = self.label_encoder.fit_transform(df["Сектор"])

        # Статистика по данным
        print(f"\n📊 Доступные признаки для модели: {', '.join(self.feature_columns)}")
        for feat in self.feature_columns:
            if feat in df.columns:
                non_na = df[feat].notna().sum()
                print(
                    f"   • {feat}: {non_na} непустых значений ({non_na/len(df)*100:.1f}%)"
                )

        return df

    def _calculate_peg(self, row) -> float:
        """Расчет PEG ratio для дополнительной оценки"""
        try:
            pe = row.get("P/E", np.nan)
            g = row.get("g", np.nan)

            if pd.notna(pe) and pd.notna(g) and pe > 0 and g > 0:
                return pe / g
            return np.nan
        except:
            return np.nan

    def _assign_target(self, row) -> int:
        """
        Определение целевой переменной для обучения на основе P/E, P/BV, ROE и g

        Используется комбинация PEG ratio и абсолютных значений
        """
        # Проверяем наличие всех необходимых данных
        required_cols = ["P/E", "P/BV", "ROE", "g"]
        if not all(
            pd.notna(row.get(col))
            for col in required_cols
            if col in self.feature_columns
        ):
            return np.nan

        pe = row.get("P/E", np.nan)
        pbv = row.get("P/BV", np.nan)
        roe = row.get("ROE", np.nan)
        g = row.get("g", np.nan)

        # Защита от некорректных значений
        if pe <= 0 or pbv <= 0 or roe <= 0:
            return np.nan

        # Расчет PEG ratio
        peg = pe / g if g > 0 else float("inf")

        # Комбинированная оценка на основе нескольких факторов

        # STRONG BUY: Компании с высоким потенциалом
        if (
            (peg < 0.5 and roe > 20 and pbv < 1.5)
            or (pe < 8 and roe > 25 and g > 15)
            or (pbv < 1 and roe > 15 and g > 10)
        ):
            return TARGET_MAPPING.STRONG_UNDERVALUED

        # BUY: Хорошие компании по разумной цене
        elif (
            (peg < 1 and roe > 15)
            or (pe < 12 and roe > 20)
            or (pbv < 1.5 and roe > 18 and g > 8)
        ):
            return TARGET_MAPPING.UNDERVALUED

        # OVERVALUED: Дорогие компании или с низким потенциалом
        elif (
            (peg > 2 and roe < 15)
            or (pe > 25 and g < 10)
            or (pbv > 3 and roe < 12)
            or (g < 5 and pe > 20)
        ):
            return TARGET_MAPPING.OVERVALUED

        # FAIR VALUE: Все остальные случаи
        else:
            return TARGET_MAPPING.FAIR_VALUE

    def _calculate_fundamental_score(self, row) -> float:
        """
        Расчет фундаментального скора (0-100) на основе всех показателей
        Используется для дополнительного анализа
        """
        score = 0
        weights = {"pe": 0.25, "pbv": 0.20, "roe": 0.30, "g": 0.25}

        # Оценка по P/E (чем ниже, тем лучше)
        if pd.notna(row.get("P/E")):
            pe = row["P/E"]
            if pe < 5:
                score += weights["pe"] * 100
            elif pe < 10:
                score += weights["pe"] * 80
            elif pe < 15:
                score += weights["pe"] * 60
            elif pe < 20:
                score += weights["pe"] * 40
            elif pe < 25:
                score += weights["pe"] * 20

        # Оценка по P/BV (чем ниже, тем лучше)
        if pd.notna(row.get("P/BV")):
            pbv = row["P/BV"]
            if pbv < 0.5:
                score += weights["pbv"] * 100
            elif pbv < 1:
                score += weights["pbv"] * 80
            elif pbv < 1.5:
                score += weights["pbv"] * 60
            elif pbv < 2:
                score += weights["pbv"] * 40
            elif pbv < 3:
                score += weights["pbv"] * 20

        # Оценка по ROE (чем выше, тем лучше)
        if pd.notna(row.get("ROE")):
            roe = row["ROE"]
            if roe > 30:
                score += weights["roe"] * 100
            elif roe > 20:
                score += weights["roe"] * 80
            elif roe > 15:
                score += weights["roe"] * 60
            elif roe > 10:
                score += weights["roe"] * 40
            elif roe > 5:
                score += weights["roe"] * 20

        # Оценка по темпу роста g (чем выше, тем лучше)
        if pd.notna(row.get("g")):
            g = row["g"]
            if g > 25:
                score += weights["g"] * 100
            elif g > 20:
                score += weights["g"] * 80
            elif g > 15:
                score += weights["g"] * 60
            elif g > 10:
                score += weights["g"] * 40
            elif g > 5:
                score += weights["g"] * 20

        return score

    def train(
        self, df: pd.DataFrame, use_stratification: bool = True, verbose: bool = True
    ):
        """
        Обучение дерева решений

        Parameters:
        -----------
        df : pd.DataFrame
            Данные для обучения (должны содержать колонки P/E, P/BV, ROE, g)
        use_stratification : bool
            Использовать стратификацию по секторам
        verbose : bool
            Детальный вывод информации
        """
        if verbose:
            print("🌳 Обучение модели дерева решений...")

        # Подготовка признаков
        df = self.prepare_features(df)

        # Проверка наличия g
        if "g" not in df.columns:
            print(
                "   ⚠️ Внимание: Отсутствует колонка 'g'. Модель будет обучена без темпа роста."
            )
            self.feature_columns = [f for f in self.feature_columns if f != "g"]

        # Определение целевой переменной
        df["Оценка"] = df.apply(self._assign_target, axis=1)

        # Фильтруем строки с определенной оценкой
        df_model = df[df["Оценка"].notna()].copy()

        if len(df_model) == 0:
            raise ValueError("Нет данных для обучения модели после фильтрации")

        if verbose:
            print(f"\n📊 Данные для обучения: {len(df_model)} компаний")
            print("   Распределение целевой переменной:")
            target_dist = df_model["Оценка"].map(TARGET_MAPPING.LABELS).value_counts()
            for label, count in target_dist.items():
                print(f"     • {label}: {count} ({count/len(df_model)*100:.1f}%)")

        # Подготовка матрицы признаков
        X = df_model[self.feature_columns].copy()
        y = df_model["Оценка"]

        # Заполнение пропусков
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                if verbose:
                    print(
                        f"   • Заполнены пропуски в {col}: медиана = {median_val:.2f}"
                    )

        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)

        # Стратифицированное разбиение
        stratify = (
            df_model["Сектор_encoded"]
            if use_stratification and "Сектор_encoded" in df_model.columns
            else y
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=MODEL_CONSTANTS.TEST_SIZE,
            random_state=MODEL_CONSTANTS.RANDOM_STATE,
            stratify=stratify,
        )

        # Обучение модели
        self.model = DecisionTreeClassifier(
            max_depth=MODEL_CONSTANTS.MAX_DEPTH,
            min_samples_split=MODEL_CONSTANTS.MIN_SAMPLES_SPLIT,
            min_samples_leaf=MODEL_CONSTANTS.MIN_SAMPLES_LEAF,
            random_state=MODEL_CONSTANTS.RANDOM_STATE,
            class_weight="balanced",  # Учитываем дисбаланс классов
            criterion="gini",  # Можно использовать 'entropy' для информационного выигрыша
        )

        self.model.fit(X_train, y_train)

        # Оценка качества
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        results = {
            "train_accuracy": self.model.score(X_train, y_train),
            "test_accuracy": self.model.score(X_test, y_test),
            "feature_importance": dict(
                zip(self.feature_columns, self.model.feature_importances_)
            ),
            "class_distribution": df_model["Оценка"].value_counts().to_dict(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(self.feature_columns),
            "n_classes": len(self.model.classes_),
            "tree_depth": self.model.get_depth(),
            "tree_leaves": self.model.get_n_leaves(),
        }

        # Расчет матриц ошибок
        from sklearn.metrics import confusion_matrix, classification_report

        results["train_confusion_matrix"] = confusion_matrix(y_train, train_pred)
        results["test_confusion_matrix"] = confusion_matrix(y_test, test_pred)
        results["classification_report"] = classification_report(
            y_test,
            test_pred,
            target_names=[
                TARGET_MAPPING.LABELS[i] for i in sorted(self.model.classes_)
            ],
            output_dict=True,
        )

        if verbose:
            print(f"\n✅ Модель обучена успешно:")
            print(f"   • Глубина дерева: {results['tree_depth']}")
            print(f"   • Количество листьев: {results['tree_leaves']}")
            print(f"   • Точность на обучении: {results['train_accuracy']:.2%}")
            print(f"   • Точность на тесте: {results['test_accuracy']:.2%}")

            print(f"\n   📈 Важность признаков:")
            for feat, imp in sorted(
                results["feature_importance"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"     • {feat}: {imp:.2%}")

        return results

    def predict(
        self, df: pd.DataFrame, add_fundamental_score: bool = True
    ) -> pd.DataFrame:
        """
        Предсказание для всех акций

        Parameters:
        -----------
        df : pd.DataFrame
            Данные для предсказания
        add_fundamental_score : bool
            Добавить дополнительный фундаментальный скор
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train().")

        df = df.copy()

        # Подготовка данных
        df = self._map_columns(df)

        # Проверка наличия признаков
        available_features = [f for f in self.feature_columns if f in df.columns]
        if not available_features:
            raise ValueError("Нет доступных признаков для предсказания")

        # Добавляем сектор для анализа
        if "Название" in df.columns:
            df["Сектор"] = df["Название"].apply(MarketAnalyzer.assign_sector)

        # Подготовка матрицы признаков
        X = df[available_features].copy()

        # Заполнение пропусков
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Предсказания
        df["Predicted_Оценка"] = self.model.predict(X_scaled)
        df["Predicted_Уверенность"] = np.max(self.model.predict_proba(X_scaled), axis=1)
        df["Predicted_Оценка_текст"] = df["Predicted_Оценка"].map(TARGET_MAPPING.LABELS)

        # Добавляем дополнительные метрики для анализа
        if "g" in df.columns and "P/E" in df.columns:
            df["PEG"] = np.where(
                (df["g"] > 0) & (df["P/E"] > 0), df["P/E"] / df["g"], np.nan
            )

        if add_fundamental_score:
            df["Fundamental_Score"] = df.apply(
                self._calculate_fundamental_score, axis=1
            )

            # Категоризация по фундаментальному скору
            def categorize_score(score):
                if pd.isna(score):
                    return "Не определен"
                elif score >= 80:
                    return "Отлично"
                elif score >= 60:
                    return "Хорошо"
                elif score >= 40:
                    return "Средне"
                elif score >= 20:
                    return "Ниже среднего"
                else:
                    return "Слабо"

            df["Fundamental_Category"] = df["Fundamental_Score"].apply(categorize_score)

        return df

    def plot_tree(
        self, filename: str = None, figsize: tuple = None, max_depth: int = 3
    ):
        """
        Визуализация дерева решений

        Parameters:
        -----------
        filename : str
            Путь для сохранения изображения
        figsize : tuple
            Размер фигуры
        max_depth : int
            Максимальная глубина отображаемого дерева
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train().")

        if filename is None:
            filename = PATHS["decision_tree"]

        if figsize is None:
            figsize = FILE_CONSTANTS.FIGURE_SIZE_TREE

        plt.figure(figsize=figsize)

        # Человеко-понятные названия признаков
        feature_names_display = []
        for feat in self.feature_columns:
            if feat == "P/E":
                feature_names_display.append("P/E")
            elif feat == "P/BV":
                feature_names_display.append("P/BV")
            elif feat == "ROE":
                feature_names_display.append("ROE (%)")
            elif feat == "g":
                feature_names_display.append("Темп роста g (%)")
            else:
                feature_names_display.append(feat)

        class_names = [TARGET_MAPPING.LABELS[i] for i in sorted(self.model.classes_)]

        plot_tree(
            self.model,
            max_depth=max_depth,  # Ограничиваем глубину для читаемости
            feature_names=feature_names_display,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=FORMATTING.TREE_FONT_SIZE,
            proportion=True,  # Показываем пропорции классов
            impurity=False,  # Не показываем impurity
            precision=2,  # Точность отображения чисел
        )

        plt.title(
            "Дерево решений для оценки акций\n(на основе P/E, P/BV, ROE и темпа роста g)",
            fontsize=FORMATTING.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()

        print(f"📊 Дерево решений сохранено в: {filename}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Получение важности признаков в виде DataFrame"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала выполните train().")

        importance_df = pd.DataFrame(
            {
                "Признак": self.feature_columns,
                "Важность": self.model.feature_importances_,
            }
        ).sort_values("Важность", ascending=False)

        return importance_df

    def get_decision_rules(self, max_depth: int = None) -> list:
        """
        Получение правил принятия решений из дерева в человеко-читаемом формате
        """
        if self.model is None:
            return []

        from sklearn.tree import _tree

        tree = self.model.tree_
        feature_names = self.feature_columns
        class_names = [TARGET_MAPPING.LABELS[i] for i in sorted(self.model.classes_)]

        rules = []

        def recurse(node, depth, condition):
            if tree.feature[node] != _tree.TREE_UNDEFINED:  # Не лист
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Левая ветка (<= threshold)
                left_condition = f"{feature} ≤ {threshold:.2f}"
                recurse(
                    tree.children_left[node], depth + 1, condition + [left_condition]
                )

                # Правая ветка (> threshold)
                right_condition = f"{feature} > {threshold:.2f}"
                recurse(
                    tree.children_right[node], depth + 1, condition + [right_condition]
                )
            else:  # Лист
                if max_depth is None or depth <= max_depth:
                    samples = tree.n_node_samples[node]
                    if samples > 0:
                        class_dist = tree.value[node][0]
                        pred_class = np.argmax(class_dist)
                        confidence = class_dist[pred_class] / samples

                        if confidence > 0.5:  # Только значимые правила
                            rule = {
                                "conditions": " и ".join(condition),
                                "prediction": class_names[pred_class],
                                "samples": samples,
                                "confidence": f"{confidence:.1%}",
                                "class_distribution": {
                                    class_names[i]: int(class_dist[i])
                                    for i in range(len(class_dist))
                                },
                            }
                            rules.append(rule)

        recurse(0, 0, [])
        return sorted(rules, key=lambda x: x["samples"], reverse=True)
