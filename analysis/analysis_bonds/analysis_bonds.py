# bonds_portfolio_optimization.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, date
import warnings
import math

warnings.filterwarnings("ignore")

from bonds_constants import (
    ANALYSIS_DATE,
    DATA_DIR,
    RESULTS_DIR,
    RISK_FREE_RATE,
    INFLATION_RATE,
    TAX_RATE,
    CURRENCY_PARAMS,
    RISK_LEVELS,
    SECTORS,
    OPTIMIZATION_PARAMS,
    SCORING_WEIGHTS,
    COLORS,
    PLOT_STYLE,
    OUTPUT_FILES,
)

# Создаем директорию для результатов
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class Bond:
    """Класс для хранения информации об облигации"""

    ticker: str
    name: str
    sector: str
    currency: str
    maturity_date: date
    nominal: float
    risk_level: int
    floating_coupon: bool
    coupon_rate: float

    # Расчетные параметры
    years_to_maturity: float = 0
    current_yield: float = 0
    yield_to_maturity: float = 0
    modified_duration: float = 0
    convexity: float = 0
    credit_spread: float = 0
    liquidity_score: float = 0
    tax_equivalent_yield: float = 0
    real_yield: float = 0

    # Скоринговые параметры
    total_score: float = 0
    yield_score: float = 0
    risk_score: float = 0
    liquidity_score_norm: float = 0
    duration_score: float = 0
    sector_score: float = 0
    currency_score: float = 0

    def calculate_metrics(self, base_rate: float = RISK_FREE_RATE):
        """Расчет метрик облигации"""
        # Дней до погашения
        today = datetime.now().date()
        days_to_maturity = (self.maturity_date - today).days
        self.years_to_maturity = max(days_to_maturity / 365, 0.1)

        # Текущая доходность
        if self.nominal > 0:
            self.current_yield = self.coupon_rate / 100

        # Доходность к погашению (упрощенная)
        if self.years_to_maturity > 0:
            if self.floating_coupon:
                # Для флоатеров: текущая ставка + спред
                self.yield_to_maturity = base_rate + (self.coupon_rate / 100)
            else:
                # Для фиксированных: приближение
                self.yield_to_maturity = self.current_yield

        # Кредитный спред
        self.credit_spread = (
            self.yield_to_maturity - CURRENCY_PARAMS[self.currency]["risk_free_rate"]
        )

        # Дюрация (упрощенная)
        if not self.floating_coupon:
            self.modified_duration = self.years_to_maturity / (
                1 + self.yield_to_maturity
            )
        else:
            self.modified_duration = 0.1  # Флоатеры имеют низкую дюрацию

        # Выпуклость (упрощенная)
        self.convexity = self.years_to_maturity**2 / 100

        # Ликвидность (на основе номинала и риск-уровня)
        self.liquidity_score = min(self.nominal / 1000, 1) * (1 - self.risk_level * 0.2)

        # Налоговый эквивалент
        self.tax_equivalent_yield = self.current_yield / (1 - TAX_RATE)

        # Реальная доходность
        self.real_yield = self.current_yield - INFLATION_RATE


class BondsDataLoader:
    """Класс для загрузки и предобработки данных об облигациях"""

    @staticmethod
    def load_bonds_data(file_path: str) -> pd.DataFrame:
        """Загрузка данных из Excel файла"""
        print(f"   Загрузка данных из: {file_path}")

        df = pd.read_excel(file_path, sheet_name=0, header=0)

        # Переименовываем колонки
        df.columns = [
            "ticker",
            "name",
            "sector",
            "currency",
            "maturity_date",
            "nominal",
            "risk_level",
            "floating_coupon",
            "coupon_rate",
        ]

        # Обработка пропусков
        df["sector"] = df["sector"].fillna("other")
        df["currency"] = df["currency"].fillna("rub")
        df["nominal"] = df["nominal"].fillna(1000)
        df["risk_level"] = df["risk_level"].fillna(2)
        df["floating_coupon"] = df["floating_coupon"].fillna(False)
        df["coupon_rate"] = df["coupon_rate"].fillna(0)

        # Конвертация дат
        df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")

        # Удаляем строки без даты погашения
        df = df.dropna(subset=["maturity_date"])

        # Фильтруем только будущие даты погашения
        today = datetime.now()
        df = df[df["maturity_date"] > today]

        return df


class BondsAnalyzer:
    """Класс для анализа облигаций"""

    def __init__(self, bonds_df: pd.DataFrame):
        self.bonds_df = bonds_df
        self.bonds_list = []
        self.process_bonds()

    def process_bonds(self):
        """Обработка облигаций и расчет метрик"""
        for _, row in self.bonds_df.iterrows():
            try:
                bond = Bond(
                    ticker=row["ticker"],
                    name=row["name"],
                    sector=row["sector"],
                    currency=row["currency"],
                    maturity_date=row["maturity_date"].date(),
                    nominal=row["nominal"],
                    risk_level=(
                        int(row["risk_level"]) if not pd.isna(row["risk_level"]) else 2
                    ),
                    floating_coupon=bool(row["floating_coupon"]),
                    coupon_rate=(
                        float(row["coupon_rate"])
                        if not pd.isna(row["coupon_rate"])
                        else 0
                    ),
                )
                bond.calculate_metrics()
                self.bonds_list.append(bond)
            except Exception as e:
                print(f"   Ошибка обработки {row.get('ticker', 'Unknown')}: {e}")
                continue

    def get_bonds_dataframe(self) -> pd.DataFrame:
        """Получение DataFrame с рассчитанными метриками"""
        data = []
        for bond in self.bonds_list:
            data.append(
                {
                    "ticker": bond.ticker,
                    "name": (
                        bond.name[:30] + "..." if len(bond.name) > 30 else bond.name
                    ),
                    "sector": bond.sector,
                    "currency": bond.currency,
                    "years_to_maturity": round(bond.years_to_maturity, 2),
                    "nominal": bond.nominal,
                    "risk_level": bond.risk_level,
                    "coupon_rate": bond.coupon_rate,
                    "current_yield": round(bond.current_yield * 100, 2),
                    "yield_to_maturity": round(bond.yield_to_maturity * 100, 2),
                    "modified_duration": round(bond.modified_duration, 2),
                    "credit_spread": round(bond.credit_spread * 100, 2),
                    "liquidity_score": round(bond.liquidity_score, 2),
                    "floating_coupon": bond.floating_coupon,
                    "maturity_date": bond.maturity_date,
                }
            )

        return pd.DataFrame(data)

    def get_statistics_by_risk(self) -> pd.DataFrame:
        """Статистика по уровням риска"""
        df = self.get_bonds_dataframe()
        stats = (
            df.groupby("risk_level")
            .agg(
                {
                    "current_yield": ["mean", "min", "max", "std"],
                    "modified_duration": ["mean", "min", "max"],
                    "years_to_maturity": "mean",
                    "ticker": "count",
                }
            )
            .round(2)
        )

        stats.columns = [
            "Avg Yield",
            "Min Yield",
            "Max Yield",
            "Yield Std",
            "Avg Duration",
            "Min Duration",
            "Max Duration",
            "Avg Maturity",
            "Count",
        ]
        return stats

    def get_statistics_by_currency(self) -> pd.DataFrame:
        """Статистика по валютам"""
        df = self.get_bonds_dataframe()
        stats = (
            df.groupby("currency")
            .agg(
                {
                    "current_yield": ["mean", "min", "max"],
                    "modified_duration": "mean",
                    "years_to_maturity": "mean",
                    "ticker": "count",
                    "nominal": "sum",
                }
            )
            .round(2)
        )

        stats.columns = [
            "Avg Yield",
            "Min Yield",
            "Max Yield",
            "Avg Duration",
            "Avg Maturity",
            "Count",
            "Total Nominal",
        ]
        return stats


class BondScorer:
    """Класс для скоринга облигаций"""

    def __init__(self, bonds_list: List[Bond]):
        self.bonds_list = bonds_list
        self.calculate_scores()

    def calculate_scores(self):
        """Расчет скоринговых оценок"""

        # Нормализация доходности
        yields = [b.current_yield for b in self.bonds_list]
        min_yield, max_yield = min(yields), max(yields)
        yield_range = max_yield - min_yield if max_yield > min_yield else 1

        # Нормализация дюрации
        durations = [b.modified_duration for b in self.bonds_list]
        min_dur, max_dur = min(durations), max(durations)
        dur_range = max_dur - min_dur if max_dur > min_dur else 1

        for bond in self.bonds_list:
            # 1. Score по доходности (чем выше, тем лучше)
            bond.yield_score = (bond.current_yield - min_yield) / yield_range

            # 2. Score по риску (чем ниже риск, тем лучше)
            bond.risk_score = 1 - (bond.risk_level / 3)

            # 3. Score по ликвидности
            bond.liquidity_score_norm = bond.liquidity_score

            # 4. Score по дюрации (для target duration)
            target_dur = OPTIMIZATION_PARAMS["target_duration"]
            bond.duration_score = 1 - min(
                abs(bond.modified_duration - target_dur) / target_dur, 1
            )

            # 5. Score по сектору
            sector_weight = SECTORS.get(bond.sector, {}).get("max_weight", 0.1)
            bond.sector_score = min(sector_weight * 5, 1)  # Нормализация

            # 6. Score по валюте
            currency_params = CURRENCY_PARAMS.get(bond.currency, CURRENCY_PARAMS["rub"])
            if bond.current_yield >= currency_params["min_yield"]:
                bond.currency_score = 1
            else:
                bond.currency_score = bond.current_yield / currency_params["min_yield"]

            # Общий score
            bond.total_score = (
                SCORING_WEIGHTS["yield_score"] * bond.yield_score
                + SCORING_WEIGHTS["risk_score"] * bond.risk_score
                + SCORING_WEIGHTS["liquidity_score"] * bond.liquidity_score_norm
                + SCORING_WEIGHTS["duration_score"] * bond.duration_score
                + SCORING_WEIGHTS["sector_score"] * bond.sector_score
                + SCORING_WEIGHTS["currency_score"] * bond.currency_score
            )

    def get_top_bonds(self, n: int = 50) -> List[Bond]:
        """Получение топ-N облигаций по скорингу"""
        sorted_bonds = sorted(
            self.bonds_list, key=lambda x: x.total_score, reverse=True
        )
        return sorted_bonds[:n]

    def get_bonds_by_risk_level(self, risk_level: int) -> List[Bond]:
        """Получение облигаций с определенным уровнем риска"""
        return [b for b in self.bonds_list if b.risk_level == risk_level]


class BondsPortfolioOptimizer:
    """Класс для оптимизации портфеля облигаций"""

    def __init__(self, bonds_list: List[Bond]):
        self.bonds_list = bonds_list
        self.n_bonds = len(bonds_list)
        self.selected_indices = []
        self.optimal_weights = None

    def portfolio_statistics(self, weights: np.ndarray) -> Dict:
        """Расчет статистик портфеля"""
        portfolio_yield = 0
        portfolio_duration = 0
        portfolio_risk = 0
        portfolio_convexity = 0

        for i, weight in enumerate(weights):
            bond = self.bonds_list[i]
            portfolio_yield += weight * bond.current_yield
            portfolio_duration += weight * bond.modified_duration
            portfolio_risk += weight * bond.risk_level
            portfolio_convexity += weight * bond.convexity

        # Оценка диверсификации
        hhi = np.sum(weights**2)  # Индекс Херфиндаля-Хиршмана
        diversification_score = 1 - (hhi - 1 / self.n_bonds) / (1 - 1 / self.n_bonds)

        return {
            "yield": portfolio_yield,
            "duration": portfolio_duration,
            "risk_score": portfolio_risk,
            "convexity": portfolio_convexity,
            "hhi": hhi,
            "diversification": diversification_score,
            "n_bonds": np.sum(weights > 0.001),
        }

    def objective_function(self, weights: np.ndarray) -> float:
        """Целевая функция для оптимизации"""
        stats = self.portfolio_statistics(weights)

        # Максимизация доходности
        yield_score = stats["yield"] * 100

        # Минимизация риска
        risk_penalty = stats["risk_score"] * 0.01

        # Штраф за отклонение от целевой дюрации
        duration_penalty = (
            abs(stats["duration"] - OPTIMIZATION_PARAMS["target_duration"]) * 0.005
        )

        # Штраф за недостаточную диверсификацию
        diversification_penalty = (1 - stats["diversification"]) * 0.02

        # Штраф за слишком большое количество облигаций
        n_bonds_penalty = (
            max(0, stats["n_bonds"] - OPTIMIZATION_PARAMS["max_bonds"]) * 0.005
        )
        n_bonds_penalty += (
            max(0, OPTIMIZATION_PARAMS["min_bonds"] - stats["n_bonds"]) * 0.01
        )

        return -(
            yield_score
            - risk_penalty
            - duration_penalty
            - diversification_penalty
            - n_bonds_penalty
        )

    def check_constraints(self, weights: np.ndarray) -> bool:
        """Проверка ограничений"""
        stats = self.portfolio_statistics(weights)

        # Проверка минимальной доходности
        if stats["yield"] < OPTIMIZATION_PARAMS["min_current_yield"]:
            return False

        # Проверка максимального веса на одну облигацию
        if np.max(weights) > OPTIMIZATION_PARAMS["max_weight_per_bond"]:
            return False

        # Проверка на одного эмитента (упрощенно - по первому слову названия)
        issuer_weights = {}
        for i, weight in enumerate(weights):
            if weight > 0:
                issuer = self.bonds_list[i].name.split()[0]
                issuer_weights[issuer] = issuer_weights.get(issuer, 0) + weight

        if (
            max(issuer_weights.values())
            > OPTIMIZATION_PARAMS["max_weight_single_issuer"]
        ):
            return False

        return True

    def optimize_portfolio(
        self, method: str = "differential_evolution"
    ) -> Tuple[np.ndarray, Dict]:
        """Оптимизация портфеля"""
        n = len(self.bonds_list)

        if method == "differential_evolution":
            # Эволюционный алгоритм для глобальной оптимизации
            bounds = [(0, OPTIMIZATION_PARAMS["max_weight_per_bond"]) for _ in range(n)]

            def objective_with_constraints(x):
                if not self.check_constraints(x):
                    return 1e10
                return self.objective_function(x)

            result = differential_evolution(
                objective_with_constraints,
                bounds,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                seed=42,
            )

            weights = result.x
        else:
            # SLSQP для локальной оптимизации
            init_weights = np.array([1 / n] * n)
            bounds = [(0, OPTIMIZATION_PARAMS["max_weight_per_bond"]) for _ in range(n)]
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            result = minimize(
                self.objective_function,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )
            weights = result.x

        # Нормализация весов
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

        # Оставляем только значимые веса
        weights[weights < 0.005] = 0
        weights = weights / np.sum(weights)

        self.optimal_weights = weights
        self.selected_indices = [i for i, w in enumerate(weights) if w > 0]

        return weights, self.portfolio_statistics(weights)

    def get_portfolio_by_risk_profile(self, risk_level: int) -> Dict:
        """Получение портфеля для заданного уровня риска"""
        # Фильтруем облигации по уровню риска
        risk_bonds = [b for b in self.bonds_list if b.risk_level <= risk_level]

        if not risk_bonds:
            return {}

        # Сортируем по скорингу
        risk_bonds.sort(key=lambda x: x.total_score, reverse=True)

        # Берем топ-30 облигаций
        selected_bonds = risk_bonds[:30]

        # Создаем временный оптимизатор
        temp_optimizer = BondsPortfolioOptimizer(selected_bonds)
        weights, stats = temp_optimizer.optimize_portfolio()

        portfolio = {"bonds": selected_bonds, "weights": weights, "statistics": stats}

        return portfolio


class BondsPortfolioVisualizer:
    """Класс для визуализации результатов"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def plot_yield_curve(self, bonds_df: pd.DataFrame, show_plot: bool = True):
        """Построение кривой доходности"""
        plt.figure(figsize=PLOT_STYLE["figure_size"]["medium"])

        for risk_level in [0, 1, 2, 3]:
            mask = bonds_df["risk_level"] == risk_level
            subset = bonds_df[mask]
            if len(subset) > 0:
                plt.scatter(
                    subset["years_to_maturity"],
                    subset["current_yield"],
                    label=RISK_LEVELS[risk_level]["name"],
                    color=COLORS[f"risk_{risk_level}"],
                    alpha=0.6,
                    s=50,
                )

        plt.xlabel("Years to Maturity", fontsize=PLOT_STYLE["label_fontsize"])
        plt.ylabel("Current Yield (%)", fontsize=PLOT_STYLE["label_fontsize"])
        plt.title("Yield Curve by Risk Level", fontsize=PLOT_STYLE["title_fontsize"])
        plt.legend(fontsize=PLOT_STYLE["legend_fontsize"])
        plt.grid(True, alpha=0.3)

        # Добавляем линию безрисковой ставки
        plt.axhline(
            y=RISK_FREE_RATE * 100,
            color="red",
            linestyle="--",
            label=f"Risk-Free Rate ({RISK_FREE_RATE*100:.1f}%)",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["yield_curve"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_sector_allocation(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение распределения по секторам"""
        sector_weights = {}
        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                sector = bond.sector
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Сортируем по весу
        sector_weights = dict(
            sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        )

        plt.figure(figsize=PLOT_STYLE["figure_size"]["small"])

        colors = [COLORS.get(s, "#999999") for s in sector_weights.keys()]
        labels = [SECTORS.get(s, {}).get("name", s) for s in sector_weights.keys()]

        wedges, texts, autotexts = plt.pie(
            sector_weights.values(),
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )

        plt.title(
            "Portfolio Sector Allocation", fontsize=PLOT_STYLE["title_fontsize"], pad=20
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["sector_allocation"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_currency_allocation(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение распределения по валютам"""
        currency_weights = {}
        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                currency = bond.currency
                currency_weights[currency] = currency_weights.get(currency, 0) + weight

        plt.figure(figsize=PLOT_STYLE["figure_size"]["small"])

        colors = [COLORS.get(c, "#999999") for c in currency_weights.keys()]

        wedges, texts, autotexts = plt.pie(
            currency_weights.values(),
            labels=[c.upper() for c in currency_weights.keys()],
            autopct="%1.1f%%",
            colors=colors,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )

        plt.title(
            "Portfolio Currency Allocation",
            fontsize=PLOT_STYLE["title_fontsize"],
            pad=20,
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.results_dir, OUTPUT_FILES["currency_allocation"] + ".png"
            ),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_maturity_profile(
        self, portfolio_bonds: List[Bond], weights: np.ndarray, show_plot: bool = True
    ):
        """Построение профиля погашений"""
        maturities = []
        bond_weights = []

        for bond, weight in zip(portfolio_bonds, weights):
            if weight > 0:
                maturities.append(bond.years_to_maturity)
                bond_weights.append(weight * 100)

        plt.figure(figsize=PLOT_STYLE["figure_size"]["medium"])

        plt.bar(range(len(maturities)), bond_weights, color="steelblue", alpha=0.7)
        plt.xlabel("Bonds", fontsize=PLOT_STYLE["label_fontsize"])
        plt.ylabel("Weight (%)", fontsize=PLOT_STYLE["label_fontsize"])
        plt.title("Portfolio Maturity Profile", fontsize=PLOT_STYLE["title_fontsize"])
        plt.xticks(
            range(len(maturities)),
            [f"{m:.1f}y" for m in maturities],
            rotation=45,
            ha="right",
        )
        plt.grid(True, alpha=0.3, axis="y")

        # Добавляем среднюю дюрацию
        avg_duration = np.average(maturities, weights=bond_weights)
        plt.axhline(
            y=avg_duration,
            color="red",
            linestyle="--",
            label=f"Average Duration: {avg_duration:.2f}y",
        )
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["maturity_profile"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_risk_analysis(self, portfolio_stats: Dict, show_plot: bool = True):
        """Построение анализа рисков"""
        fig, axes = plt.subplots(1, 2, figsize=PLOT_STYLE["figure_size"]["medium"])

        # Радарная диаграмма (упрощенно - столбцы)
        metrics = ["Yield", "Duration", "Diversification", "Risk Score", "Convexity"]
        values = [
            portfolio_stats["yield"] * 100,
            portfolio_stats["duration"],
            portfolio_stats["diversification"] * 100,
            10 - portfolio_stats["risk_score"],  # Инвертируем для наглядности
            portfolio_stats["convexity"] * 10,
        ]

        ax1 = axes[0]
        x_pos = np.arange(len(metrics))
        ax1.bar(
            x_pos, values, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics, rotation=45, ha="right")
        ax1.set_ylabel("Score")
        ax1.set_title("Portfolio Risk Metrics")
        ax1.grid(True, alpha=0.3, axis="y")

        # Круговая диаграмма распределения риска
        ax2 = axes[1]
        risk_labels = ["Yield", "Duration", "Concentration", "Credit"]
        risk_values = [
            portfolio_stats["yield"] * 30,
            portfolio_stats["duration"] * 5,
            (1 - portfolio_stats["diversification"]) * 50,
            portfolio_stats["risk_score"] * 5,
        ]

        ax2.pie(
            risk_values,
            labels=risk_labels,
            autopct="%1.1f%%",
            colors=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
        )
        ax2.set_title("Risk Distribution")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, OUTPUT_FILES["risk_analysis"] + ".png"),
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )

        if show_plot:
            plt.show()
        else:
            plt.close()


class BondsExcelReportGenerator:
    """Класс для генерации Excel отчетов"""

    @staticmethod
    def save_to_excel(
        results_dir: str,
        bonds_df: pd.DataFrame,
        portfolio_bonds: List[Bond],
        portfolio_weights: np.ndarray,
        portfolio_stats: Dict,
        stats_by_risk: pd.DataFrame,
        stats_by_currency: pd.DataFrame,
        risk_portfolios: Dict = None,
    ):
        """Сохранение результатов в Excel файл"""

        file_path = os.path.join(results_dir, f"{OUTPUT_FILES['full_report']}.xlsx")

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

            # 1. Общая информация о портфеле
            portfolio_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Дата анализа",
                        "Количество облигаций",
                        "Средняя доходность",
                        "Средняя дюрация",
                        "Средний уровень риска",
                        "Индекс диверсификации",
                        "Концентрация (HHI)",
                        "Выпуклость",
                    ],
                    "Value": [
                        ANALYSIS_DATE,
                        portfolio_stats["n_bonds"],
                        f"{portfolio_stats['yield']*100:.2f}%",
                        f"{portfolio_stats['duration']:.2f} лет",
                        f"{portfolio_stats['risk_score']:.2f}",
                        f"{portfolio_stats['diversification']*100:.1f}%",
                        f"{portfolio_stats['hhi']:.4f}",
                        f"{portfolio_stats['convexity']:.4f}",
                    ],
                }
            )
            portfolio_summary.to_excel(
                writer, sheet_name="Portfolio Summary", index=False, startrow=1
            )
            writer.sheets["Portfolio Summary"].cell(
                row=1, column=1, value="Сводная информация по портфелю"
            )

            # 2. Состав портфеля
            portfolio_data = []
            for bond, weight in zip(portfolio_bonds, portfolio_weights):
                if weight > 0:
                    portfolio_data.append(
                        {
                            "Ticker": bond.ticker,
                            "Name": bond.name,
                            "Sector": bond.sector,
                            "Currency": bond.currency,
                            "Risk Level": bond.risk_level,
                            "Weight (%)": f"{weight*100:.2f}%",
                            "Coupon (%)": bond.coupon_rate,
                            "Yield (%)": f"{bond.current_yield*100:.2f}%",
                            "Duration": f"{bond.modified_duration:.2f}",
                            "Maturity": bond.maturity_date.strftime("%Y-%m-%d"),
                            "Years to Maturity": f"{bond.years_to_maturity:.2f}",
                            "Score": f"{bond.total_score:.4f}",
                        }
                    )

            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df.to_excel(
                writer, sheet_name="Portfolio Holdings", index=False, startrow=1
            )
            writer.sheets["Portfolio Holdings"].cell(
                row=1, column=1, value="Состав оптимального портфеля"
            )

            # 3. Статистика по уровням риска
            stats_by_risk.to_excel(writer, sheet_name="Risk Statistics", startrow=1)
            writer.sheets["Risk Statistics"].cell(
                row=1, column=1, value="Статистика по уровням риска"
            )

            # 4. Статистика по валютам
            stats_by_currency.to_excel(
                writer, sheet_name="Currency Statistics", startrow=1
            )
            writer.sheets["Currency Statistics"].cell(
                row=1, column=1, value="Статистика по валютам"
            )

            # 5. Полный список облигаций
            bonds_df_sorted = bonds_df.sort_values("total_score", ascending=False)
            bonds_df_sorted.to_excel(
                writer, sheet_name="All Bonds", index=False, startrow=1
            )
            writer.sheets["All Bonds"].cell(
                row=1, column=1, value="Полный список облигаций с рейтингом"
            )

            # 6. Портфели по уровням риска
            if risk_portfolios:
                risk_data = []
                for level, portfolio in risk_portfolios.items():
                    if portfolio:
                        risk_data.append(
                            {
                                "Risk Level": level,
                                "Risk Name": RISK_LEVELS[level]["name"],
                                "Number of Bonds": portfolio["statistics"]["n_bonds"],
                                "Yield (%)": f"{portfolio['statistics']['yield']*100:.2f}%",
                                "Duration": f"{portfolio['statistics']['duration']:.2f}",
                                "Risk Score": f"{portfolio['statistics']['risk_score']:.2f}",
                                "Diversification": f"{portfolio['statistics']['diversification']*100:.1f}%",
                            }
                        )

                risk_df = pd.DataFrame(risk_data)
                risk_df.to_excel(
                    writer, sheet_name="Risk Portfolios", index=False, startrow=1
                )
                writer.sheets["Risk Portfolios"].cell(
                    row=1, column=1, value="Портфели по уровням риска"
                )

            # 7. Параметры оптимизации
            params_df = pd.DataFrame(
                {
                    "Parameter": list(OPTIMIZATION_PARAMS.keys()),
                    "Value": list(OPTIMIZATION_PARAMS.values()),
                }
            )
            params_df.to_excel(
                writer, sheet_name="Optimization Params", index=False, startrow=1
            )
            writer.sheets["Optimization Params"].cell(
                row=1, column=1, value="Параметры оптимизации"
            )

        print(f"\n   Excel отчет сохранен: {file_path}")


class BondsPortfolioAnalysisPipeline:
    """Основной класс для запуска анализа"""

    def __init__(self, show_plots: bool = True):
        self.show_plots = show_plots
        self.results_dir = RESULTS_DIR

        self.bonds_df = None
        self.bonds_list = None
        self.analyzer = None
        self.scorer = None
        self.optimizer = None
        self.visualizer = BondsPortfolioVisualizer(self.results_dir)

    def run(self, file_path: str) -> Dict[str, Any]:
        """Запуск полного цикла анализа"""

        print("\n" + "=" * 80)
        print("АНАЛИЗ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ ОБЛИГАЦИЙ")
        print("=" * 80)

        # 1. Загрузка данных
        print("\n1. Загрузка данных об облигациях...")
        loader = BondsDataLoader()
        raw_df = loader.load_bonds_data(file_path)
        print(f"   Загружено {len(raw_df)} облигаций")

        # 2. Анализ облигаций
        print("\n2. Анализ облигаций и расчет метрик...")
        self.analyzer = BondsAnalyzer(raw_df)
        self.bonds_df = self.analyzer.get_bonds_dataframe()
        self.bonds_list = self.analyzer.bonds_list
        print(f"   Обработано {len(self.bonds_list)} облигаций")

        # 3. Статистика
        stats_by_risk = self.analyzer.get_statistics_by_risk()
        stats_by_currency = self.analyzer.get_statistics_by_currency()

        print("\n   Статистика по уровням риска:")
        print(stats_by_risk)
        print("\n   Статистика по валютам:")
        print(stats_by_currency)

        # 4. Скоринг облигаций
        print("\n3. Скоринг облигаций...")
        self.scorer = BondScorer(self.bonds_list)
        top_bonds = self.scorer.get_top_bonds(50)
        print(f"   Отобрано топ-50 облигаций для оптимизации")

        # 5. Визуализация кривой доходности
        print("\n4. Построение кривой доходности...")
        self.visualizer.plot_yield_curve(self.bonds_df, self.show_plots)

        # 6. Оптимизация портфеля
        print("\n5. Оптимизация портфеля...")
        self.optimizer = BondsPortfolioOptimizer(top_bonds)
        optimal_weights, portfolio_stats = self.optimizer.optimize_portfolio()

        print(f"\n   Результаты оптимизации:")
        print(f"   Количество облигаций: {portfolio_stats['n_bonds']}")
        print(f"   Средняя доходность: {portfolio_stats['yield']*100:.2f}%")
        print(f"   Средняя дюрация: {portfolio_stats['duration']:.2f} лет")
        print(f"   Средний риск: {portfolio_stats['risk_score']:.2f}")
        print(f"   Диверсификация: {portfolio_stats['diversification']*100:.1f}%")

        # 7. Получение портфелей для разных уровней риска
        print("\n6. Формирование портфелей по уровням риска...")
        risk_portfolios = {}
        for level in [0, 1, 2, 3]:
            portfolio = self.optimizer.get_portfolio_by_risk_profile(level)
            if portfolio:
                risk_portfolios[level] = portfolio
                print(
                    f"   Риск-уровень {level} ({RISK_LEVELS[level]['name']}): "
                    f"{portfolio['statistics']['n_bonds']} облигаций, "
                    f"доходность {portfolio['statistics']['yield']*100:.2f}%, "
                    f"дюрация {portfolio['statistics']['duration']:.2f} лет"
                )

        # 8. Визуализация портфеля
        print("\n7. Визуализация портфеля...")
        selected_bonds = [top_bonds[i] for i in self.optimizer.selected_indices]
        selected_weights = optimal_weights[optimal_weights > 0]

        self.visualizer.plot_sector_allocation(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_currency_allocation(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_maturity_profile(
            selected_bonds, selected_weights, self.show_plots
        )
        self.visualizer.plot_risk_analysis(portfolio_stats, self.show_plots)

        # 9. Сохранение результатов в Excel
        print("\n8. Сохранение результатов...")

        # Добавляем скоринговые оценки в DataFrame
        score_data = []
        for bond in self.bonds_list:
            score_data.append(
                {
                    "ticker": bond.ticker,
                    "total_score": bond.total_score,
                    "yield_score": bond.yield_score,
                    "risk_score": bond.risk_score,
                    "liquidity_score": bond.liquidity_score_norm,
                    "duration_score": bond.duration_score,
                    "sector_score": bond.sector_score,
                    "currency_score": bond.currency_score,
                }
            )

        scores_df = pd.DataFrame(score_data)
        self.bonds_df = self.bonds_df.merge(scores_df, on="ticker", how="left")
        self.bonds_df = self.bonds_df.sort_values("total_score", ascending=False)

        self.excel_reporter = BondsExcelReportGenerator()
        self.excel_reporter.save_to_excel(
            self.results_dir,
            self.bonds_df,
            selected_bonds,
            selected_weights,
            portfolio_stats,
            stats_by_risk,
            stats_by_currency,
            risk_portfolios,
        )

        print(f"\n   Все результаты сохранены в: {self.results_dir}")

        return {
            "bonds_df": self.bonds_df,
            "bonds_list": self.bonds_list,
            "optimal_portfolio": {
                "bonds": selected_bonds,
                "weights": selected_weights,
                "statistics": portfolio_stats,
            },
            "risk_portfolios": risk_portfolios,
            "statistics": {"by_risk": stats_by_risk, "by_currency": stats_by_currency},
        }


def main():
    file_path = os.path.join(DATA_DIR, "bonds_data.xlsx")
    pipeline = BondsPortfolioAnalysisPipeline()
    results = pipeline.run(file_path)

    return results


if __name__ == "__main__":
    results = main()
