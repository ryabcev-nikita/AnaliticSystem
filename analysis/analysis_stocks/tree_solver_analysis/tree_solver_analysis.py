import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings

# Импорт констант
from tree_solver_constants import (
    FINANCIAL,
    VALUATION_SCORES,
    RETURN_PREMIUMS,
    RISK_PREMIUMS,
    MODEL_CONSTANTS,
    TARGET_MAPPING,
    PORTFOLIO_CONSTANTS,
    FILE_CONSTANTS,
    SECTOR_KEYWORDS,
    SECTOR_NAMES,
    FORMATTING,
    CONVERSION,
    REPORT,
)

warnings.filterwarnings("ignore")

# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


class PathConfig:
    """Конфигурация путей к файлам"""

    @staticmethod
    def setup_directories():
        """Создание необходимых директорий"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree_solver_dir = f"{parent_dir}/../data/tree_solver"
        os.makedirs(tree_solver_dir, exist_ok=True)

        return {
            "parent_dir": parent_dir,
            "tree_solver_dir": tree_solver_dir,
            "file_path": f"{parent_dir}/../data/fundamentals_shares.xlsx",
            "decision_tree": f"{tree_solver_dir}/{FILE_CONSTANTS.DECISION_TREE_FILE}",
            "efficient_frontier": f"{tree_solver_dir}/{FILE_CONSTANTS.EFFICIENT_FRONTIER_FILE}",
            "portfolio_report": f"{tree_solver_dir}/{FILE_CONSTANTS.INVEST_PORTFOLIO_REPORT}",
            "optimal_portfolio": f"{tree_solver_dir}/{FILE_CONSTANTS.OPTIMAL_PORTFOLIO_FILE}",
        }


PATHS = PathConfig.setup_directories()

# ==================== КЛАССЫ ДАННЫХ ====================


@dataclass
class MarketBenchmarks:
    """Рыночные бенчмарки на основе медианных значений"""

    pe_median: float
    pb_median: float
    ps_median: float
    roe_median: float
    div_yield_median: float
    debt_capital_median: float
    beta_median: float


@dataclass
class PortfolioMetrics:
    """Метрики портфеля"""

    expected_return: float
    risk: float
    sharpe_ratio: float
    diversification_score: float


# ==================== КЛАСС ЗАГРУЗЧИКА ДАННЫХ ====================


class DataLoader:
    """Загрузка и первичная обработка данных"""

    @staticmethod
    def convert_to_float(value):
        """Конвертация строк с числами в float"""
        if pd.isna(value) or value == "" or value == 0:
            return np.nan
        if isinstance(value, (int, float)):
            return value

        value = str(value).strip()
        value = value.replace(CONVERSION.THOUSAND_SEPARATOR, "").replace(
            CONVERSION.DECIMAL_SEPARATOR, "."
        )

        if CONVERSION.BILLION_PATTERN in value:
            return float(re.sub(r"[^\d.]", "", value)) * CONVERSION.BILLION
        elif CONVERSION.MILLION_PATTERN in value:
            return float(re.sub(r"[^\d.]", "", value)) * CONVERSION.MILLION
        else:
            try:
                return float(re.sub(r"[^\d.-]", "", value))
            except:
                return np.nan

    @staticmethod
    def load_and_clean_data(filepath: str) -> pd.DataFrame:
        """Загрузка и очистка данных"""
        df = pd.read_excel(filepath, sheet_name="Sheet1")

        numeric_columns = [
            "Рыночная капитализация",
            "EV",
            "Выручка",
            "Чистая прибыль",
            "EBITDA",
            "P/E",
            "P/B",
            "P/S",
            "P/FCF",
            "ROE",
            "ROA",
            "ROIC",
            "EV/EBITDA",
            "EV/S",
            "Payot Ratio",
            "NPM",
            "Debt",
            "Debt/Capital",
            "Net_Debt/EBITDA",
            "Debt/EBITDA",
            "EPS",
            "Averange_dividend_yield",
            "Свободный денежный поток",
            "Бета",
            "Дивиденд на акцию",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader.convert_to_float)

        return df


# ==================== КЛАСС АНАЛИЗАТОРА РЫНКА ====================


class MarketAnalyzer:
    """Анализ рыночных мультипликаторов и бенчмарков"""

    @staticmethod
    def calculate_benchmarks(df: pd.DataFrame) -> MarketBenchmarks:
        """Расчет медианных значений мультипликаторов"""
        return MarketBenchmarks(
            pe_median=df["P/E"].median(),
            pb_median=df["P/B"].median(),
            ps_median=df["P/S"].median(),
            roe_median=df["ROE"].median(),
            div_yield_median=df["Averange_dividend_yield"].median(),
            debt_capital_median=df["Debt/Capital"].median(),
            beta_median=df["Бета"].median(),
        )

    @staticmethod
    def assign_sector(name: str) -> str:
        """Определение сектора компании"""
        name = str(name).lower()

        sector_mappings = [
            (SECTOR_KEYWORDS.BANKS, SECTOR_NAMES.BANKS),
            (SECTOR_KEYWORDS.OIL_GAS, SECTOR_NAMES.OIL_GAS),
            (SECTOR_KEYWORDS.METALS, SECTOR_NAMES.METALS),
            (SECTOR_KEYWORDS.ENERGY, SECTOR_NAMES.ENERGY),
            (SECTOR_KEYWORDS.TELECOM, SECTOR_NAMES.TELECOM),
            (SECTOR_KEYWORDS.RETAIL, SECTOR_NAMES.RETAIL),
            (SECTOR_KEYWORDS.CHEMICAL, SECTOR_NAMES.CHEMICAL),
            (SECTOR_KEYWORDS.IT, SECTOR_NAMES.IT),
        ]

        for keywords, sector_name in sector_mappings:
            if any(word in name for word in keywords):
                return sector_name

        return SECTOR_NAMES.OTHER

    @staticmethod
    def calculate_relative_valuation(
        row: pd.Series, benchmarks: MarketBenchmarks
    ) -> Dict:
        """Оценка относительной стоимости на основе медиан"""
        scores = {}

        # P/E оценка
        if pd.notna(row.get("P/E")):
            pe_ratio = row["P/E"]
            if (
                pe_ratio
                < benchmarks.pe_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and pe_ratio > 0
            ):
                scores["pe_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                pe_ratio < benchmarks.pe_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and pe_ratio > 0
            ):
                scores["pe_score"] = VALUATION_SCORES.BUY
            elif (
                pe_ratio > benchmarks.pe_median * FINANCIAL.OVERVALUED_THRESHOLD
                or pe_ratio < 0
            ):
                scores["pe_score"] = VALUATION_SCORES.SELL
            else:
                scores["pe_score"] = VALUATION_SCORES.HOLD
        else:
            scores["pe_score"] = VALUATION_SCORES.HOLD

        # P/S оценка
        if pd.notna(row.get("P/S")):
            ps_ratio = row["P/S"]
            if (
                ps_ratio
                < benchmarks.ps_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
            ):
                scores["ps_score"] = VALUATION_SCORES.STRONG_BUY
            elif ps_ratio < benchmarks.ps_median * FINANCIAL.UNDERVALUED_THRESHOLD:
                scores["ps_score"] = VALUATION_SCORES.BUY
            elif ps_ratio > benchmarks.ps_median * FINANCIAL.OVERVALUED_THRESHOLD:
                scores["ps_score"] = VALUATION_SCORES.SELL
            else:
                scores["ps_score"] = VALUATION_SCORES.HOLD
        else:
            scores["ps_score"] = VALUATION_SCORES.HOLD

        # P/B оценка
        if pd.notna(row.get("P/B")) and pd.notna(row.get("ROE")):
            pb_ratio = row["P/B"]
            if (
                pb_ratio < benchmarks.pb_median * FINANCIAL.PB_STRONG_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                pb_ratio < benchmarks.pb_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.BUY
            elif (
                pb_ratio > benchmarks.pb_median * FINANCIAL.PB_OVERVAULED_THRESHOLD
                and row["ROE"] > 0
            ):
                scores["pb_score"] = VALUATION_SCORES.SELL
            else:
                scores["pb_score"] = VALUATION_SCORES.HOLD
        else:
            scores["pb_score"] = VALUATION_SCORES.HOLD

        # ROE оценка
        if pd.notna(row.get("ROE")):
            roe = row["ROE"]
            if roe > benchmarks.roe_median * FINANCIAL.ROE_STRONG_THRESHOLD and roe > 0:
                scores["roe_score"] = VALUATION_SCORES.STRONG_BUY
            elif roe > benchmarks.roe_median * FINANCIAL.ROE_GOOD_THRESHOLD and roe > 0:
                scores["roe_score"] = VALUATION_SCORES.BUY
            else:
                scores["roe_score"] = VALUATION_SCORES.HOLD
        else:
            scores["roe_score"] = VALUATION_SCORES.HOLD

        # Дивидендная оценка
        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row["Дивидендная доходность"]
            if (
                div_yield
                > benchmarks.div_yield_median * FINANCIAL.DIVIDEND_STRONG_THRESHOLD
            ):
                scores["div_score"] = VALUATION_SCORES.STRONG_BUY
            elif (
                div_yield
                > benchmarks.div_yield_median * FINANCIAL.DIVIDEND_GOOD_THRESHOLD
            ):
                scores["div_score"] = VALUATION_SCORES.BUY
            else:
                scores["div_score"] = VALUATION_SCORES.HOLD
        else:
            scores["div_score"] = VALUATION_SCORES.HOLD

        # Итоговая оценка
        total_score = sum(scores.values())
        scores["total_score"] = total_score

        if total_score >= VALUATION_SCORES.STRONG_BUY_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[
                TARGET_MAPPING.STRONG_UNDERVALUED
            ]
            scores["valuation_code"] = TARGET_MAPPING.STRONG_UNDERVALUED
        elif total_score >= VALUATION_SCORES.BUY_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.UNDERVALUED]
            scores["valuation_code"] = TARGET_MAPPING.UNDERVALUED
        elif total_score <= VALUATION_SCORES.SELL_THRESHOLD:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.OVERVALUED]
            scores["valuation_code"] = TARGET_MAPPING.OVERVALUED
        else:
            scores["valuation"] = TARGET_MAPPING.LABELS[TARGET_MAPPING.FAIR_VALUE]
            scores["valuation_code"] = TARGET_MAPPING.FAIR_VALUE

        return scores


# ==================== КЛАСС МОДЕЛИ ДЕРЕВА РЕШЕНИЙ ====================


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


# ==================== КЛАСС ФУНДАМЕНТАЛЬНОГО АНАЛИЗА ====================


class FundamentalAnalyzer:
    """Фундаментальный анализ и расчет доходности/риска"""

    def __init__(self, benchmarks: MarketBenchmarks):
        self.benchmarks = benchmarks

    def calculate_expected_return(self, row: pd.Series) -> float:
        """Расчет ожидаемой доходности на основе фундаментальных показателей"""
        base_return = FINANCIAL.BASE_RETURN
        score = 0.0

        # P/E премия
        if pd.notna(row.get("P/E")):
            pe = row["P/E"]
            if (
                pe
                < self.benchmarks.pe_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and pe > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PE_PREMIUM
            elif (
                pe < self.benchmarks.pe_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and pe > 0
            ):
                score += RETURN_PREMIUMS.PE_PREMIUM

        # P/S премия
        if pd.notna(row.get("P/S")):
            ps = row["P/S"]
            if (
                ps
                < self.benchmarks.ps_median * FINANCIAL.STRONGLY_UNDERVALUED_THRESHOLD
                and ps > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PS_PREMIUM
            elif (
                ps < self.benchmarks.ps_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and ps > 0
            ):
                score += RETURN_PREMIUMS.PS_PREMIUM

        # P/B премия
        if pd.notna(row.get("P/B")) and pd.notna(row.get("ROE")):
            pb = row["P/B"]
            if (
                pb < self.benchmarks.pb_median * FINANCIAL.PB_STRONG_THRESHOLD
                and row["ROE"] > 0
            ):
                score += RETURN_PREMIUMS.STRONG_PB_PREMIUM
            elif (
                pb < self.benchmarks.pb_median * FINANCIAL.UNDERVALUED_THRESHOLD
                and row["ROE"] > 0
            ):
                score += RETURN_PREMIUMS.PB_PREMIUM

        # ROE премия
        if pd.notna(row.get("ROE")):
            roe = row["ROE"]
            if (
                roe > self.benchmarks.roe_median * FINANCIAL.ROE_STRONG_THRESHOLD
                and roe > 0
            ):
                score += RETURN_PREMIUMS.STRONG_ROE_PREMIUM
            elif (
                roe > self.benchmarks.roe_median * FINANCIAL.ROE_GOOD_THRESHOLD
                and roe > 0
            ):
                score += RETURN_PREMIUMS.ROE_PREMIUM

        # Дивидендная премия
        if pd.notna(row.get("Дивидендная доходность")):
            div_yield = row["Дивидендная доходность"]
            if (
                div_yield
                > self.benchmarks.div_yield_median * FINANCIAL.DIVIDEND_STRONG_THRESHOLD
            ):
                score += RETURN_PREMIUMS.STRONG_DIVIDEND_PREMIUM
            elif (
                div_yield
                > self.benchmarks.div_yield_median * FINANCIAL.DIVIDEND_GOOD_THRESHOLD
            ):
                score += RETURN_PREMIUMS.DIVIDEND_PREMIUM

        # Бонус за оценку модели
        if pd.notna(row.get("Predicted_Оценка")):
            if row["Predicted_Оценка"] == TARGET_MAPPING.STRONG_UNDERVALUED:
                score += RETURN_PREMIUMS.MODEL_STRONG_PREMIUM
            elif row["Predicted_Оценка"] == TARGET_MAPPING.UNDERVALUED:
                score += RETURN_PREMIUMS.MODEL_PREMIUM

        return base_return + score

    def calculate_risk(self, row: pd.Series) -> float:
        """Расчет риска на основе беты, долга и волатильности"""
        base_risk = RISK_PREMIUMS.BASE_RISK

        # Бета риск
        if pd.notna(row.get("Бета")):
            beta = row["Бета"]
            if beta > self.benchmarks.beta_median * FINANCIAL.BETA_HIGH_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_HIGH_PENALTY
            elif beta > self.benchmarks.beta_median * FINANCIAL.UNDERVALUED_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_MEDIUM_PENALTY
            elif beta < self.benchmarks.beta_median * FINANCIAL.BETA_LOW_THRESHOLD:
                base_risk += RISK_PREMIUMS.BETA_LOW_BONUS

        # Долговой риск
        if pd.notna(row.get("Debt/Capital")):
            debt = row["Debt/Capital"]
            if (
                debt
                > self.benchmarks.debt_capital_median * FINANCIAL.DEBT_HIGH_THRESHOLD
            ):
                base_risk += RISK_PREMIUMS.DEBT_HIGH_PENALTY
            elif (
                debt
                > self.benchmarks.debt_capital_median * FINANCIAL.UNDERVALUED_THRESHOLD
            ):
                base_risk += RISK_PREMIUMS.DEBT_MEDIUM_PENALTY

        # Штраф за переоцененность
        if pd.notna(row.get("Predicted_Оценка")):
            if row["Predicted_Оценка"] == TARGET_MAPPING.OVERVALUED:
                base_risk += RISK_PREMIUMS.OVERVALUED_PENALTY

        return max(RISK_PREMIUMS.MIN_RISK, min(RISK_PREMIUMS.MAX_RISK, base_risk))


# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================


class PortfolioOptimizer:
    """Оптимизация портфеля по Марковицу"""

    def __init__(self, min_weight: float = None, max_weight: float = None):
        self.min_weight = min_weight or PORTFOLIO_CONSTANTS.MIN_WEIGHT
        self.max_weight = max_weight or PORTFOLIO_CONSTANTS.MAX_WEIGHT

    def create_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Создание матрицы ковариации"""
        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["Риск"].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    correlation = (
                        PORTFOLIO_CONSTANTS.INTRASECTOR_CORRELATION
                        if df.iloc[i]["Сектор"] == df.iloc[j]["Сектор"]
                        else PORTFOLIO_CONSTANTS.INTERSECTOR_CORRELATION
                    )
                    cov_matrix[i, j] = correlation * risks[i] * risks[j]

        return cov_matrix

    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> Dict:
        """Оптимизация портфеля"""
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_risk if port_risk > 0 else -np.inf

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        init_guess = [1 / n] * n

        # Максимизация Шарпа
        result_sharpe = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Минимизация риска
        result_min_risk = minimize(
            portfolio_risk,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Комбинированный портфель
        combined_weights = (
            PORTFOLIO_CONSTANTS.SHARPE_PORTFOLIO_WEIGHT * result_sharpe.x
            + PORTFOLIO_CONSTANTS.MIN_RISK_PORTFOLIO_WEIGHT * result_min_risk.x
        )
        combined_weights = combined_weights / combined_weights.sum()

        return {
            "sharpe_weights": result_sharpe.x,
            "min_risk_weights": result_min_risk.x,
            "combined_weights": combined_weights,
            "cov_matrix": cov_matrix,
        }


# ==================== КЛАСС ПОРТФЕЛЬНОГО МЕНЕДЖЕРА ====================


class PortfolioManager:
    """Управление портфелем и расчет метрик"""

    def __init__(self, df: pd.DataFrame, weights: np.ndarray):
        self.df = df.copy()
        self.weights = weights
        self.df["weights"] = weights
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        exp_return = np.sum(self.df["Ожидаемая_доходность"] * self.weights)

        optimizer = PortfolioOptimizer()
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        sharpe = exp_return / risk if risk > 0 else 0

        # Индекс диверсификации Херфиндаля-Хиршмана
        hhi = np.sum(self.weights**2)
        n = len(self.weights)
        if n > 1:
            diversification = 1 - (hhi - 1 / n) / (1 - 1 / n)
        else:
            diversification = 0

        return PortfolioMetrics(
            expected_return=exp_return,
            risk=risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        return self.df.groupby("Сектор")["weights"].sum()

    def get_top_positions(
        self, n: int = PORTFOLIO_CONSTANTS.TOP_POSITIONS_N
    ) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()


# ==================== КЛАСС ВИЗУАЛИЗАТОРА ====================


class PortfolioVisualizer:
    """Визуализация портфеля и результатов анализа"""

    @staticmethod
    def plot_portfolio_summary(
        portfolio_df: pd.DataFrame,
        weights: np.ndarray,
        metrics: PortfolioMetrics,
        benchmarks: MarketBenchmarks,
        filename: str = None,
    ):
        """Сводная визуализация портфеля"""
        if filename is None:
            filename = PATHS["optimal_portfolio"]

        fig, axes = plt.subplots(2, 2, figsize=FILE_CONSTANTS.FIGURE_SIZE_SUMMARY)

        plot_df = portfolio_df.copy()
        plot_df["weights"] = weights
        plot_df = plot_df.reset_index(drop=True)

        n_positions = len(plot_df)

        # 1. Pie chart - распределение весов - ИСПРАВЛЕНО
        top_n = min(PORTFOLIO_CONSTANTS.TOP_PIE_N, len(plot_df))
        if top_n > 0:
            top_indices = np.argsort(weights)[::-1][:top_n]
            top_weights = weights[top_indices]
            top_tickers = plot_df.iloc[top_indices]["Тикер"].values

            other_weight = max(0, 1 - top_weights.sum())
            if other_weight > PORTFOLIO_CONSTANTS.MIN_WEIGHT and len(top_weights) < len(
                weights
            ):
                plot_weights = np.append(top_weights, other_weight)
                plot_labels = np.append(top_tickers, ["Другие"])
            else:
                plot_weights = top_weights
                plot_labels = top_tickers
                if abs(1 - plot_weights.sum()) > 0.01:
                    plot_weights = plot_weights / plot_weights.sum()

            # ИСПРАВЛЕНО: используем MATPLOTLIB_PERCENT для autopct
            axes[0, 0].pie(
                plot_weights,
                labels=plot_labels,
                autopct=FORMATTING.MATPLOTLIB_PERCENT,  # '%1.1f%%'
                startangle=90,
                colors=plt.cm.get_cmap(FORMATTING.COLOR_PIE_CMAP)(
                    range(len(plot_weights))
                ),
            )
            axes[0, 0].set_title(
                f"Топ-{top_n} позиций в портфеле",
                fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )

        # 2. Risk-Return scatter
        axes[0, 1].scatter(
            plot_df["Риск"],
            plot_df["Ожидаемая_доходность"],
            s=weights * 3000,
            alpha=0.6,
            c=FORMATTING.COLOR_PORTFOLIO_MARKER,
            edgecolors="black",
            linewidths=0.5,
        )

        for idx, row in plot_df.iterrows():
            if (
                idx < len(weights)
                and weights[idx] > PORTFOLIO_CONSTANTS.ANNOTATION_WEIGHT_THRESHOLD
            ):
                axes[0, 1].annotate(
                    row["Тикер"],
                    (row["Риск"], row["Ожидаемая_доходность"]),
                    fontsize=FORMATTING.ANNOTATION_FONT_SIZE,
                    alpha=0.8,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        axes[0, 1].axhline(
            y=metrics.expected_return,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Портфель: {metrics.expected_return:.1%}",
        )
        axes[0, 1].axvline(x=metrics.risk, color="r", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel(
            "Риск (волатильность)", fontsize=FORMATTING.AXIS_FONT_SIZE
        )
        axes[0, 1].set_ylabel(
            "Ожидаемая доходность", fontsize=FORMATTING.AXIS_FONT_SIZE
        )
        axes[0, 1].set_title(
            "Risk-Return профиль",
            fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # 3. Секторальное распределение - ИСПРАВЛЕНО
        if len(plot_df) > 0 and "weights" in plot_df.columns:
            sector_weights = plot_df.groupby("Сектор")["weights"].sum()
            if len(sector_weights) > 0:
                sector_weights = sector_weights.sort_values(ascending=False)
                colors = plt.cm.get_cmap(FORMATTING.COLOR_SECTOR_CMAP)(
                    np.linspace(0, 1, len(sector_weights))
                )
                # ИСПРАВЛЕНО: используем MATPLOTLIB_PERCENT для autopct
                axes[1, 0].pie(
                    sector_weights.values,
                    labels=sector_weights.index,
                    autopct=FORMATTING.MATPLOTLIB_PERCENT,  # '%1.1f%%'
                    startangle=90,
                    colors=colors,
                )
                axes[1, 0].set_title(
                    "Распределение по секторам",
                    fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
                    fontweight="bold",
                )

        # 4. Метрики портфеля
        axes[1, 1].axis("off")

        min_weight_val = (
            weights[weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT].min()
            if any(weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT)
            else 0
        )

        metrics_text = PortfolioVisualizer._format_metrics_text(
            metrics, benchmarks, n_positions, weights, min_weight_val
        )

        axes[1, 1].text(
            0.05,
            0.5,
            metrics_text,
            transform=axes[1, 1].transAxes,
            fontsize=FORMATTING.LABEL_FONT_SIZE,
            verticalalignment="center",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=FORMATTING.COLOR_PORTFOLIO_BG,
                edgecolor=FORMATTING.COLOR_PORTFOLIO_MARKER,
                alpha=0.9,
            ),
        )
        axes[1, 1].set_title(
            "Сводка портфеля", fontsize=FORMATTING.SUBTITLE_FONT_SIZE, fontweight="bold"
        )

        plt.suptitle(
            "Оптимальный инвестиционный портфель",
            fontsize=FORMATTING.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _format_metrics_text(
        metrics: PortfolioMetrics,
        benchmarks: MarketBenchmarks,
        n_positions: int,
        weights: np.ndarray,
        min_weight: float,
    ) -> str:
        """Форматирование текста с метриками портфеля"""

        div_yield_str = (
            FORMATTING.PERCENT_FORMAT.format(benchmarks.div_yield_median / 100)
            if pd.notna(benchmarks.div_yield_median) and benchmarks.div_yield_median > 0
            else FORMATTING.NA_STRING
        )
        roe_str = (
            FORMATTING.PERCENT_FORMAT.format(benchmarks.roe_median / 100)
            if pd.notna(benchmarks.roe_median) and benchmarks.roe_median > 0
            else FORMATTING.NA_STRING
        )
        pe_str = (
            FORMATTING.FLOAT_FORMAT_1D.format(benchmarks.pe_median)
            if pd.notna(benchmarks.pe_median) and benchmarks.pe_median > 0
            else FORMATTING.NA_STRING
        )
        pb_str = (
            FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.pb_median)
            if pd.notna(benchmarks.pb_median) and benchmarks.pb_median > 0
            else FORMATTING.NA_STRING
        )

        return (
            "\n        📊 МЕТРИКИ ПОРТФЕЛЯ\n        \n"
            f"        Ожидаемая доходность: {FORMATTING.PERCENT_FORMAT.format(metrics.expected_return)}\n"
            f"        Риск (волатильность): {FORMATTING.PERCENT_FORMAT.format(metrics.risk)}\n"
            f"        Коэффициент Шарпа: {FORMATTING.FLOAT_FORMAT_2D.format(metrics.sharpe_ratio)}\n"
            f"        Индекс диверсификации: {FORMATTING.PERCENT_FORMAT.format(metrics.diversification_score)}\n"
            "        \n"
            "        📈 СОСТАВ\n"
            f"        Количество позиций: {n_positions}\n"
            f"        Максимальная доля: {FORMATTING.PERCENT_FORMAT.format(weights.max())}\n"
            f"        Минимальная доля: {FORMATTING.PERCENT_FORMAT.format(min_weight)}\n"
            "        \n"
            "        📉 РЫНОЧНЫЕ БЕНЧМАРКИ\n"
            f"        Медианный P/E: {pe_str}\n"
            f"        Медианный P/B: {pb_str}\n"
            f"        Медианный ROE: {roe_str}\n"
            f"        Мед. див. доходность: {div_yield_str}\n"
        )

    @staticmethod
    def plot_efficient_frontier(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        optimal_weights: np.ndarray,
        optimal_return: float,
        optimal_risk: float,
        filename: str = None,
    ):
        """Построение границы эффективности"""
        if filename is None:
            filename = PATHS["efficient_frontier"]

        n_assets = len(expected_returns)

        if n_assets < 2:
            print("⚠️ Недостаточно активов для построения границы эффективности")
            return

        n_portfolios = PORTFOLIO_CONSTANTS.N_EFFICIENT_PORTFOLIOS
        returns = []
        risks = []
        sharpe_ratios = []

        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()

            port_return = np.sum(expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            returns.append(port_return)
            risks.append(port_risk)
            sharpe_ratios.append(port_return / port_risk if port_risk > 0 else 0)

        plt.figure(figsize=FILE_CONSTANTS.FIGURE_SIZE_FRONTIER)

        scatter = plt.scatter(
            risks,
            returns,
            c=sharpe_ratios,
            cmap=FORMATTING.COLOR_RISK_RETURN_CMAP,
            alpha=0.3,
            s=15,
        )
        plt.colorbar(scatter, label="Коэффициент Шарпа")

        plt.scatter(
            optimal_risk,
            optimal_return,
            c=FORMATTING.COLOR_OPTIMAL_MARKER,
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Оптимальный портфель",
        )

        plt.xlabel("Риск (волатильность)", fontsize=FORMATTING.AXIS_FONT_SIZE)
        plt.ylabel("Ожидаемая доходность", fontsize=FORMATTING.AXIS_FONT_SIZE)
        plt.title(
            "Граница эффективности Марковица",
            fontsize=FORMATTING.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=FILE_CONSTANTS.DPI, bbox_inches="tight")
        plt.show()


# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================


class ReportGenerator:
    """Генерация отчетов и рекомендаций"""

    @staticmethod
    def generate_portfolio_report(
        portfolio_manager: PortfolioManager,
        benchmarks: MarketBenchmarks,
        filename: str = None,
    ):
        """Генерация Excel отчета"""
        if filename is None:
            filename = PATHS["portfolio_report"]

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Состав портфеля
                portfolio_df = portfolio_manager.df.copy()
                portfolio_df = portfolio_df.sort_values("weights", ascending=False)

                portfolio_display = portfolio_df[
                    [
                        "Тикер",
                        "Название",
                        "Сектор",
                        "weights",
                        "Ожидаемая_доходность",
                        "Риск",
                        "P/E",
                        "P/B",
                        "ROE",
                        "Дивидендная доходность",
                        "Predicted_Оценка_текст",
                        "Predicted_Уверенность",
                    ]
                ].copy()

                portfolio_display.columns = [
                    REPORT.COLUMN_TICKER,
                    REPORT.COLUMN_NAME,
                    REPORT.COLUMN_SECTOR,
                    REPORT.COLUMN_WEIGHT,
                    REPORT.COLUMN_EXPECTED_RETURN,
                    REPORT.COLUMN_RISK,
                    REPORT.COLUMN_PE,
                    REPORT.COLUMN_PB,
                    REPORT.COLUMN_ROE,
                    REPORT.COLUMN_DIV_YIELD,
                    REPORT.COLUMN_RATING,
                    REPORT.COLUMN_CONFIDENCE,
                ]

                portfolio_display.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_PORTFOLIO, index=False
                )

                # Секторальное распределение
                sector_weights = portfolio_manager.get_sector_allocation()
                if len(sector_weights) > 0:
                    sector_df = pd.DataFrame(
                        {"Сектор": sector_weights.index, "Доля": sector_weights.values}
                    )
                    sector_df.to_excel(
                        writer, sheet_name=FILE_CONSTANTS.SHEET_SECTORS, index=False
                    )

                # Метрики портфеля
                min_weight_val = (
                    portfolio_manager.weights[
                        portfolio_manager.weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT
                    ].min()
                    if any(portfolio_manager.weights > PORTFOLIO_CONSTANTS.MIN_WEIGHT)
                    else 0
                )

                metrics_df = pd.DataFrame(
                    {
                        "Метрика": [
                            REPORT.METRIC_EXPECTED_RETURN,
                            REPORT.METRIC_RISK,
                            REPORT.METRIC_SHARPE,
                            REPORT.METRIC_DIVERSIFICATION,
                            REPORT.METRIC_N_POSITIONS,
                            REPORT.METRIC_MAX_WEIGHT,
                            REPORT.METRIC_MIN_WEIGHT,
                        ],
                        "Значение": [
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.expected_return
                            ),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.risk
                            ),
                            FORMATTING.FLOAT_FORMAT_2D.format(
                                portfolio_manager.metrics.sharpe_ratio
                            ),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.metrics.diversification_score
                            ),
                            len(portfolio_manager.df),
                            FORMATTING.PERCENT_FORMAT.format(
                                portfolio_manager.weights.max()
                            ),
                            FORMATTING.PERCENT_FORMAT.format(min_weight_val),
                        ],
                    }
                )
                metrics_df.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_METRICS, index=False
                )

                # Рыночные бенчмарки
                benchmarks_df = pd.DataFrame(
                    {
                        "Мультипликатор": [
                            REPORT.BENCHMARK_PE,
                            REPORT.BENCHMARK_PB,
                            REPORT.BENCHMARK_PS,
                            REPORT.BENCHMARK_ROE,
                            REPORT.BENCHMARK_DIV_YIELD,
                            REPORT.BENCHMARK_DEBT,
                            REPORT.BENCHMARK_BETA,
                        ],
                        "Значение": ReportGenerator._format_benchmark_values(
                            benchmarks
                        ),
                    }
                )
                benchmarks_df.to_excel(
                    writer, sheet_name=FILE_CONSTANTS.SHEET_BENCHMARKS, index=False
                )

            print(f"   ✅ Отчет сохранен: {filename}")

        except Exception as e:
            print(f"   ❌ Ошибка при сохранении отчета: {e}")

    @staticmethod
    def _format_benchmark_values(benchmarks: MarketBenchmarks) -> List[str]:
        """Форматирование значений бенчмарков"""
        return [
            (
                FORMATTING.FLOAT_FORMAT_1D.format(benchmarks.pe_median)
                if pd.notna(benchmarks.pe_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.pb_median)
                if pd.notna(benchmarks.pb_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.ps_median)
                if pd.notna(benchmarks.ps_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.roe_median / 100)
                if pd.notna(benchmarks.roe_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.div_yield_median / 100)
                if pd.notna(benchmarks.div_yield_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.PERCENT_FORMAT.format(benchmarks.debt_capital_median / 100)
                if pd.notna(benchmarks.debt_capital_median)
                else FORMATTING.NA_STRING
            ),
            (
                FORMATTING.FLOAT_FORMAT_2D.format(benchmarks.beta_median)
                if pd.notna(benchmarks.beta_median)
                else FORMATTING.NA_STRING
            ),
        ]

    @staticmethod
    def print_recommendations(portfolio_manager: PortfolioManager):
        """Вывод рекомендаций в консоль"""
        print(FORMATTING.SEPARATOR)
        print("🎯 КЛЮЧЕВЫЕ ИНВЕСТИЦИОННЫЕ РЕКОМЕНДАЦИИ")
        print(FORMATTING.SEPARATOR)

        # Топ-3 рекомендации
        top_n = min(
            PORTFOLIO_CONSTANTS.TOP_RECOMMENDATIONS_N, len(portfolio_manager.df)
        )
        top_positions = portfolio_manager.get_top_positions(top_n)

        print(f"\n🔹 ТОП-{top_n} АКЦИИ ДЛЯ ПОКУПКИ:")
        for _, row in top_positions.iterrows():
            print(f"   • {row['Тикер']} - {row['Название']}")
            print(
                f"     Доля: {FORMATTING.PERCENT_FORMAT.format(row['weights'])} | "
                f"Доходность: {FORMATTING.PERCENT_FORMAT.format(row['Ожидаемая_доходность'])} | "
                f"Риск: {FORMATTING.PERCENT_FORMAT.format(row['Риск'])}"
            )
            print(
                f"     Оценка: {row['Predicted_Оценка_текст']} "
                f"(уверенность: {FORMATTING.PERCENT_FORMAT.format(row['Predicted_Уверенность'])})"
            )

        print("\n🔸 ДИВЕРСИФИКАЦИЯ:")
        print(
            f"   • Индекс диверсификации: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.metrics.diversification_score)}"
        )
        sector_allocation = portfolio_manager.get_sector_allocation()
        print(f"   • Количество секторов: {len(sector_allocation)}")
        if len(sector_allocation) > 0:
            print(
                f"   • Максимальная доля сектора: {FORMATTING.PERCENT_FORMAT.format(sector_allocation.max())}"
            )

        print("\n🔹 РИСК-МЕНЕДЖМЕНТ:")
        print(
            f"   • Коэффициент Шарпа: {FORMATTING.FLOAT_FORMAT_2D.format(portfolio_manager.metrics.sharpe_ratio)} "
            f"(выше 1 - отлично)"
        )
        print(
            f"   • Ожидаемая волатильность: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.metrics.risk)}"
        )
        print(
            f"   • Максимальная доля одной акции: {FORMATTING.PERCENT_FORMAT.format(portfolio_manager.weights.max())} "
            f"(лимит {FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.MAX_WEIGHT)})"
        )

        print("\n🔸 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
        print("   • Ребалансировка портфеля каждые 3-6 месяцев")
        print("   • Мониторинг фундаментальных показателей ежеквартально")
        print(
            f"   • Stop-loss: {FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.STOP_LOSS_THRESHOLD)} от цены покупки для каждой позиции"
        )
        print(
            f"   • Take-profit: +{FORMATTING.PERCENT_FORMAT.format(PORTFOLIO_CONSTANTS.TAKE_PROFIT_THRESHOLD)} для недооцененных акций"
        )

        print(FORMATTING.SEPARATOR)


# ==================== ДОПОЛНИТЕЛЬНЫЙ КЛАСС ДЛЯ АНАЛИЗА МУЛЬТИПЛИКАТОРОВ ====================


class MultiplierAnalyzer:
    """Детальный анализ мультипликаторов"""

    @staticmethod
    def analyze_sector_multipliers(df: pd.DataFrame) -> pd.DataFrame:
        """Анализ мультипликаторов по секторам"""
        sector_stats = []

        for sector in df["Сектор"].unique():
            sector_df = df[df["Сектор"] == sector]

            stats = {
                "Сектор": sector,
                "Количество": len(sector_df),
                "P/E_медиана": sector_df["P/E"].median(),
                "P/B_медиана": sector_df["P/B"].median(),
                "ROE_медиана": sector_df["ROE"].median(),
                "Див.доходность_медиана": sector_df["Дивидендная доходность"].median(),
                "Бета_медиана": sector_df["Бета"].median(),
                "Капитализация_медиана": sector_df["Рыночная капитализация"].median()
                / CONVERSION.BILLION,
            }
            sector_stats.append(stats)

        return pd.DataFrame(sector_stats)

    @staticmethod
    def find_best_values(df: pd.DataFrame) -> Dict:
        """Поиск лучших значений мультипликаторов"""
        best_values = {
            "Минимальный P/E": df[df["P/E"] > 0]["P/E"].min(),
            "Минимальный P/B": df[df["P/B"] > 0]["P/B"].min(),
            "Максимальный ROE": df["ROE"].max(),
            "Максимальная див.доходность": df["Дивидендная доходность"].max(),
            "Минимальная бета": df[df["Бета"] > 0]["Бета"].min(),
        }
        return best_values


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================


def main():
    """Основной пайплайн анализа и оптимизации портфеля"""

    print("🚀 Запуск анализа фундаментальных показателей и оптимизации портфеля...")

    # Шаг 1: Загрузка данных
    print("📥 Загрузка данных...")
    loader = DataLoader()
    df = loader.load_and_clean_data(PATHS["file_path"])

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
    optimizer = PortfolioOptimizer(
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
