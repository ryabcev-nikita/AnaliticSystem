# ==================== КЛАСС МОДЕЛИ ДЕРЕВА РЕШЕНИЙ ====================
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
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
    """Модель дерева решений для оценки акций"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            "P/E",
            "P/B",
            "P/S",
            "P/FCF",
            "ROE",
            "ROA",
            "Averange_dividend_yield",
            "Бета",
            "Debt/Capital",
            "NPM",
            "Сектор_encoded",
        ]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        df = df.copy()
        df["Сектор"] = df["Название"].apply(MarketAnalyzer.assign_sector)
        df["Сектор_encoded"] = self.label_encoder.fit_transform(df["Сектор"])
        return df

    def train(self, df: pd.DataFrame):
        """Обучение дерева решений"""
        df = self.prepare_features(df)
        df["Оценка"] = df.apply(self._assign_target, axis=1)
        df_model = df[df["Оценка"].notna()].copy()

        X = df_model[self.feature_columns].copy()
        y = df_model["Оценка"]

        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=MODEL_CONSTANTS.TEST_SIZE,
            random_state=MODEL_CONSTANTS.RANDOM_STATE,
            stratify=y,
        )

        self.model = DecisionTreeClassifier(
            max_depth=MODEL_CONSTANTS.MAX_DEPTH,
            min_samples_split=MODEL_CONSTANTS.MIN_SAMPLES_SPLIT,
            min_samples_leaf=MODEL_CONSTANTS.MIN_SAMPLES_LEAF,
            random_state=MODEL_CONSTANTS.RANDOM_STATE,
        )

        self.model.fit(X_train, y_train)

        return {
            "train_accuracy": self.model.score(X_train, y_train),
            "test_accuracy": self.model.score(X_test, y_test),
            "feature_importance": dict(
                zip(self.feature_columns, self.model.feature_importances_)
            ),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предсказание для всех акций"""
        df = df.copy()
        df["Сектор"] = df["Название"].apply(MarketAnalyzer.assign_sector)
        df["Сектор_encoded"] = self.label_encoder.transform(df["Сектор"])

        X = df[self.feature_columns].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        X_scaled = self.scaler.transform(X)

        df["Predicted_Оценка"] = self.model.predict(X_scaled)
        df["Predicted_Уверенность"] = np.max(self.model.predict_proba(X_scaled), axis=1)
        df["Predicted_Оценка_текст"] = df["Predicted_Оценка"].map(TARGET_MAPPING.LABELS)

        return df

    @staticmethod
    def _assign_target(row):
        """Определение целевой переменной для обучения"""
        if (
            pd.notna(row.get("P/E", np.nan))
            and pd.notna(row.get("P/B", np.nan))
            and pd.notna(row.get("ROE", np.nan))
        ):

            pe = row["P/E"]
            pb = row["P/B"]
            roe = row["ROE"]

            if (
                pe < MODEL_CONSTANTS.PE_STRONG_BUY_THRESHOLD
                and pb < MODEL_CONSTANTS.PB_STRONG_BUY_THRESHOLD
                and pe > 0
                and roe > 0
            ):
                return TARGET_MAPPING.STRONG_UNDERVALUED
            elif (
                pe < MODEL_CONSTANTS.PE_BUY_THRESHOLD
                and pb < MODEL_CONSTANTS.PB_BUY_THRESHOLD
                and roe > 0
            ):
                return TARGET_MAPPING.UNDERVALUED
            elif (
                pe > MODEL_CONSTANTS.PE_SELL_THRESHOLD
                or pb > MODEL_CONSTANTS.PB_SELL_THRESHOLD
                and roe > 0
            ):
                return TARGET_MAPPING.OVERVALUED
            else:
                return TARGET_MAPPING.FAIR_VALUE
        return np.nan

    def plot_tree(self, filename: str = None):
        """Визуализация дерева решений"""
        if filename is None:
            filename = PATHS["decision_tree"]

        plt.figure(figsize=FILE_CONSTANTS.FIGURE_SIZE_TREE)
        plot_tree(
            self.model,
            feature_names=self.feature_columns,
            class_names=list(TARGET_MAPPING.LABELS.values()),
            filled=True,
            rounded=True,
            fontsize=FORMATTING.TREE_FONT_SIZE,
        )
        plt.title(
            "Дерево решений для оценки акций", fontsize=FORMATTING.TITLE_FONT_SIZE
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()
