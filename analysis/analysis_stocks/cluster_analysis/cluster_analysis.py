import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings

# Импорт констант
from cluster_constants import (
    CLUSTER,
    CLUSTER_THRESHOLDS,
    CLUSTER_SCORES,
    PORTFOLIO_CLUSTER,
    RETURN_PREMIUMS_CLUSTER,
    RISK_PREMIUMS_CLUSTER,
    SECTOR_KEYWORDS_CLUSTER,
    SECTOR_NAMES_CLUSTER,
    SCORING,
    CLUSTER_FILES,
    CLUSTER_FORMAT,
    CLUSTER_REPORT,
)

warnings.filterwarnings("ignore")

# ==================== КОНФИГУРАЦИЯ ПУТЕЙ ====================


class ClusterPathConfig:
    """Конфигурация путей к файлам для кластерного анализа"""

    @staticmethod
    def setup_directories():
        """Создание необходимых директорий"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cluster_dir = f"{parent_dir}/../data/cluster_analysis"
        os.makedirs(cluster_dir, exist_ok=True)

        return {
            "cluster_dir": cluster_dir,
            "data_path": f"{parent_dir}/../data/fundamentals_shares.xlsx",
            "cluster_analysis": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_ANALYSIS_FILE}",
            "cluster_optimization": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_OPTIMIZATION_FILE}",
            "portfolio_comparison": f"{cluster_dir}/{CLUSTER_FILES.PORTFOLIO_COMPARISON_FILE}",
            "cluster_allocation": f"{cluster_dir}/{CLUSTER_FILES.CLUSTER_ALLOCATION_FILE}",
            "clustered_companies": f"{cluster_dir}/{CLUSTER_FILES.CLUSTERED_COMPANIES_FILE}",
            "investment_cluster_report": f"{cluster_dir}/{CLUSTER_FILES.INVESTMENT_CLUSTER_REPORT}",
        }


CLUSTER_PATHS = ClusterPathConfig.setup_directories()

# ==================== КЛАССЫ ДАННЫХ ====================


@dataclass
class ClusterCharacteristics:
    """Характеристики кластера"""

    cluster_id: int
    size: int
    avg_pe: float
    avg_pb: float
    avg_roe: float
    avg_div_yield: float
    avg_risk: float
    description: str
    recommendation: str


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
        value = value.replace(" ", "").replace(",", ".")

        if "млрд" in value:
            return float(re.sub(r"[^\d.]", "", value)) * 1e9
        elif "млн" in value:
            return float(re.sub(r"[^\d.]", "", value)) * 1e6
        else:
            try:
                return float(re.sub(r"[^\d.-]", "", value))
            except:
                return np.nan

    @staticmethod
    def load_and_clean_data(filepath: str) -> pd.DataFrame:
        """Загрузка и очистка данных"""
        df = pd.read_excel(filepath, sheet_name="Sheet1")

        column_mapping = {
            "Тикер": "Ticker",
            "Название": "Company",
            "Рыночная капитализация": "Market_Cap",
            "P/E": "PE",
            "P/B": "PB",
            "P/S": "PS",
            "P/FCF": "PFCF",
            "ROE": "ROE",
            "ROA": "ROA",
            "ROIC": "ROIC",
            "EV/EBITDA": "EV_EBITDA",
            "Averange_dividend_yield": "Div_Yield",
            "Бета": "Beta",
            "Debt/Capital": "Debt_Capital",
            "Свободный денежный поток": "FCF",
            "Чистая прибыль": "Net_Income",
            "Выручка": "Revenue",
        }

        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)

        numeric_columns = [
            "Market_Cap",
            "PE",
            "PB",
            "PS",
            "PFCF",
            "ROE",
            "ROA",
            "ROIC",
            "EV_EBITDA",
            "Div_Yield",
            "Beta",
            "Debt_Capital",
            "FCF",
            "Net_Income",
            "Revenue",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader.convert_to_float)

        df["Sector"] = df["Company"].apply(DataLoader.assign_sector)

        return df

    @staticmethod
    def assign_sector(name: str) -> str:
        """Определение сектора компании"""
        if pd.isna(name):
            return SECTOR_NAMES_CLUSTER.OTHER

        name = str(name).lower()

        sector_mappings = [
            (SECTOR_KEYWORDS_CLUSTER.BANKS, SECTOR_NAMES_CLUSTER.BANKS),
            (SECTOR_KEYWORDS_CLUSTER.OIL_GAS, SECTOR_NAMES_CLUSTER.OIL_GAS),
            (SECTOR_KEYWORDS_CLUSTER.METALS, SECTOR_NAMES_CLUSTER.METALS),
            (SECTOR_KEYWORDS_CLUSTER.ENERGY, SECTOR_NAMES_CLUSTER.ENERGY),
            (SECTOR_KEYWORDS_CLUSTER.TELECOM, SECTOR_NAMES_CLUSTER.TELECOM),
            (SECTOR_KEYWORDS_CLUSTER.RETAIL, SECTOR_NAMES_CLUSTER.RETAIL),
            (SECTOR_KEYWORDS_CLUSTER.CHEMICAL, SECTOR_NAMES_CLUSTER.CHEMICAL),
            (SECTOR_KEYWORDS_CLUSTER.IT, SECTOR_NAMES_CLUSTER.IT),
        ]

        for keywords, sector_name in sector_mappings:
            if any(word in name for word in keywords):
                return sector_name

        return SECTOR_NAMES_CLUSTER.OTHER


# ==================== КЛАСС КЛАСТЕРНОГО АНАЛИЗА ====================


class ClusterAnalyzer:
    """Кластерный анализ компаний на основе мультипликаторов"""

    def __init__(self, n_clusters: int = None):
        self.n_clusters = n_clusters or CLUSTER.DEFAULT_N_CLUSTERS
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=CLUSTER.PCA_COMPONENTS)
        self.cluster_profiles = None
        self.feature_names = []

    def find_optimal_clusters(
        self, df: pd.DataFrame, features: List[str], max_clusters: int = None
    ) -> Tuple[int, pd.DataFrame]:
        """Поиск оптимального количества кластеров"""
        max_clusters = max_clusters or CLUSTER.MAX_CLUSTERS

        df_clean = df[features].dropna()

        if len(df_clean) < CLUSTER.MIN_DATA_FOR_CLUSTERING:
            print(
                f"   ⚠️ Недостаточно данных для кластеризации. "
                f"Доступно: {len(df_clean)} компаний"
            )
            return min(CLUSTER.MIN_CLUSTERS, len(df_clean)), pd.DataFrame()

        scaled_data = self.scaler.fit_transform(df_clean)

        inertias = []
        silhouette_scores = []
        max_k = min(max_clusters + 1, len(df_clean))

        for k in range(CLUSTER.MIN_CLUSTERS, max_k):
            kmeans = KMeans(
                n_clusters=k, random_state=CLUSTER.RANDOM_STATE, n_init=CLUSTER.N_INIT
            )
            labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)

            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(scaled_data, labels))
            else:
                silhouette_scores.append(0)

        self._plot_optimization(inertias, silhouette_scores)

        optimal_k = (
            np.argmax(silhouette_scores) + CLUSTER.MIN_CLUSTERS
            if silhouette_scores
            else CLUSTER.MIN_CLUSTERS
        )

        print(f"   Оптимальное количество кластеров: {optimal_k}")

        return optimal_k, pd.DataFrame(
            {
                "clusters": range(CLUSTER.MIN_CLUSTERS, max_k),
                "inertia": inertias,
                "silhouette": silhouette_scores
                + [0] * (len(inertias) - len(silhouette_scores)),
            }
        )

    def _plot_optimization(self, inertias: List[float], silhouette_scores: List[float]):
        """Визуализация оптимизации количества кластеров"""
        fig, axes = plt.subplots(1, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_OPTIMIZATION)

        k_range = range(CLUSTER.MIN_CLUSTERS, CLUSTER.MIN_CLUSTERS + len(inertias))

        axes[0].plot(k_range, inertias, marker="o", linewidth=2)
        axes[0].set_xlabel(
            "Количество кластеров", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE
        )
        axes[0].set_ylabel("Inertia", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        axes[0].set_title(
            "Метод локтя", fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE, fontweight="bold"
        )
        axes[0].grid(True, alpha=CLUSTER.GRID_ALPHA)

        axes[1].plot(
            k_range[: len(silhouette_scores)],
            silhouette_scores,
            marker="o",
            linewidth=2,
            color="green",
        )
        axes[1].set_xlabel(
            "Количество кластеров", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE
        )
        axes[1].set_ylabel("Silhouette Score", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        axes[1].set_title(
            "Анализ силуэтов",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        axes[1].grid(True, alpha=CLUSTER.GRID_ALPHA)

        plt.suptitle(
            "Определение оптимального количества кластеров",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            CLUSTER_PATHS["cluster_optimization"],
            dpi=CLUSTER_FILES.DPI,
            bbox_inches="tight",
        )
        plt.show()

    def fit_predict(
        self, df: pd.DataFrame, features: List[str], n_clusters: int = None
    ) -> pd.DataFrame:
        """Обучение модели и предсказание кластеров"""
        if n_clusters:
            self.n_clusters = n_clusters

        df_clean = df[df[features].notna().all(axis=1)].copy()
        self.feature_names = features

        if len(df_clean) < self.n_clusters:
            print(
                f"   ⚠️ Недостаточно данных для кластеризации. "
                f"Уменьшаем количество кластеров до {len(df_clean)}"
            )
            self.n_clusters = max(CLUSTER.MIN_CLUSTERS, len(df_clean))

        scaled_data = self.scaler.fit_transform(df_clean[features])

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=CLUSTER.RANDOM_STATE,
            n_init=CLUSTER.N_INIT,
        )
        df_clean["Cluster"] = self.kmeans.fit_predict(scaled_data)

        pca_result = self.pca.fit_transform(scaled_data)
        df_clean["PCA1"] = pca_result[:, 0]
        df_clean["PCA2"] = pca_result[:, 1]

        return df_clean

    def analyze_clusters(self, df: pd.DataFrame) -> List[ClusterCharacteristics]:
        """Анализ характеристик кластеров"""
        cluster_profiles = []

        for cluster_id in range(self.n_clusters):
            cluster_data = df[df["Cluster"] == cluster_id]

            if len(cluster_data) == 0:
                continue

            avg_pe = cluster_data["PE"].median() if "PE" in cluster_data else 0
            avg_pb = cluster_data["PB"].median() if "PB" in cluster_data else 0
            avg_roe = cluster_data["ROE"].median() if "ROE" in cluster_data else 0
            avg_div = (
                cluster_data["Div_Yield"].median() if "Div_Yield" in cluster_data else 0
            )

            risk_factors = []
            if "Beta" in cluster_data:
                risk_factors.append(cluster_data["Beta"].mean())
            if "Debt_Capital" in cluster_data:
                risk_factors.append(cluster_data["Debt_Capital"].mean() / 100)

            avg_risk = (
                np.mean(risk_factors) if risk_factors else PORTFOLIO_CLUSTER.BASE_RISK
            )

            description = self._describe_cluster(avg_pe, avg_pb, avg_roe)
            recommendation = self._get_recommendation(avg_pe, avg_pb, avg_roe, avg_div)

            cluster_profiles.append(
                ClusterCharacteristics(
                    cluster_id=cluster_id,
                    size=len(cluster_data),
                    avg_pe=avg_pe,
                    avg_pb=avg_pb,
                    avg_roe=avg_roe,
                    avg_div_yield=avg_div,
                    avg_risk=avg_risk,
                    description=description,
                    recommendation=recommendation,
                )
            )

        self.cluster_profiles = cluster_profiles
        return cluster_profiles

    def _describe_cluster(self, avg_pe: float, avg_pb: float, avg_roe: float) -> str:
        """Формирование описания кластера"""
        if (
            avg_pe < CLUSTER_THRESHOLDS.PE_DEEP_VALUE
            and avg_pb < CLUSTER_THRESHOLDS.PB_DEEP_VALUE
        ):
            return CLUSTER_REPORT.DESC_DEEP_VALUE
        elif (
            avg_pe < CLUSTER_THRESHOLDS.PE_VALUE
            and avg_pb < CLUSTER_THRESHOLDS.PB_VALUE
        ):
            return CLUSTER_REPORT.DESC_VALUE
        elif (
            avg_pe > CLUSTER_THRESHOLDS.PE_GROWTH
            and avg_pb > CLUSTER_THRESHOLDS.PB_GROWTH
        ):
            return CLUSTER_REPORT.DESC_GROWTH_OVER
        elif avg_roe > CLUSTER_THRESHOLDS.ROE_HIGH:
            return CLUSTER_REPORT.DESC_HIGH_PROFIT
        elif avg_roe > CLUSTER_THRESHOLDS.ROE_GOOD:
            return CLUSTER_REPORT.DESC_PROFIT
        else:
            return CLUSTER_REPORT.DESC_FAIR

    def _get_recommendation(
        self, avg_pe: float, avg_pb: float, avg_roe: float, avg_div: float
    ) -> str:
        """Формирование рекомендации для кластера"""
        score = 0

        if avg_pe < CLUSTER_THRESHOLDS.PE_VALUE:
            score += CLUSTER_SCORES.PE_VALUE_SCORE
        if avg_pb < CLUSTER_THRESHOLDS.PB_VALUE:
            score += CLUSTER_SCORES.PB_VALUE_SCORE
        if avg_roe > CLUSTER_THRESHOLDS.ROE_GOOD:
            score += CLUSTER_SCORES.ROE_GOOD_SCORE
        if avg_div > CLUSTER_THRESHOLDS.DIV_GOOD:
            score += CLUSTER_SCORES.DIV_GOOD_SCORE

        if score >= CLUSTER_SCORES.AGGRESSIVE_BUY_THRESHOLD:
            return CLUSTER_REPORT.REC_AGGRESSIVE_BUY
        elif score >= CLUSTER_SCORES.BUY_THRESHOLD:
            return CLUSTER_REPORT.REC_BUY
        elif score >= CLUSTER_SCORES.HOLD_THRESHOLD:
            return CLUSTER_REPORT.REC_HOLD
        else:
            return CLUSTER_REPORT.REC_AVOID

    def plot_clusters(self, df: pd.DataFrame, save_path: str = None):
        """Визуализация кластеров"""
        if save_path is None:
            save_path = CLUSTER_PATHS["cluster_analysis"]

        fig, axes = plt.subplots(2, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_CLUSTERS)

        self._plot_pca_clusters(df, axes[0, 0])
        self._plot_pb_roe_clusters(df, axes[0, 1])
        self._plot_cluster_sizes(df, axes[1, 0])
        self._plot_cluster_profiles(axes[1, 1])

        plt.suptitle(
            "Результаты кластерного анализа компаний",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=CLUSTER_FILES.DPI, bbox_inches="tight")
        plt.show()

    def _plot_pca_clusters(self, df: pd.DataFrame, ax):
        """Визуализация PCA проекции кластеров"""
        if "PCA1" in df.columns and "PCA2" in df.columns:
            scatter = ax.scatter(
                df["PCA1"],
                df["PCA2"],
                c=df["Cluster"],
                cmap=CLUSTER_FORMAT.COLOR_CLUSTER_CMAP,
                s=CLUSTER.SCATTER_POINT_SIZE,
                alpha=CLUSTER.SCATTER_ALPHA,
                edgecolors="black",
                linewidths=0.5,
            )

            if self.kmeans is not None:
                centroids = self.pca.transform(self.kmeans.cluster_centers_)
                ax.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    marker="X",
                    s=CLUSTER.CENTROID_POINT_SIZE,
                    c=CLUSTER_FORMAT.COLOR_CENTROID,
                    edgecolors="black",
                    linewidths=2,
                    label="Центроиды",
                )

            ax.set_xlabel("PCA Component 1", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
            ax.set_ylabel("PCA Component 2", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
            ax.set_title(
                "Кластеризация компаний (PCA проекция)",
                fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
                fontweight="bold",
            )
            ax.grid(True, alpha=CLUSTER.GRID_ALPHA)
            ax.legend()

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Кластер", fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE)

    def _plot_pb_roe_clusters(self, df: pd.DataFrame, ax):
        """Визуализация распределения кластеров по P/B и ROE"""
        for cluster_id in df["Cluster"].unique():
            cluster_data = df[df["Cluster"] == cluster_id]
            if "PB" in cluster_data and "ROE" in cluster_data:
                ax.scatter(
                    cluster_data["PB"],
                    cluster_data["ROE"],
                    label=f"Кластер {cluster_id}",
                    s=CLUSTER.SCATTER_POINT_SIZE,
                    alpha=CLUSTER.SCATTER_ALPHA,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_xlabel("P/B", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("ROE, %", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Распределение кластеров: P/B vs ROE",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA)

    def _plot_cluster_sizes(self, df: pd.DataFrame, ax):
        """Визуализация размеров кластеров"""
        cluster_sizes = df["Cluster"].value_counts().sort_index()
        colors = plt.cm.get_cmap(CLUSTER_FORMAT.COLOR_CLUSTER_CMAP)(
            np.linspace(0, 1, len(cluster_sizes))
        )
        bars = ax.bar(
            cluster_sizes.index.astype(str),
            cluster_sizes.values,
            color=colors,
            edgecolor="black",
        )

        for bar, size in zip(bars, cluster_sizes.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + CLUSTER.BAR_TEXT_OFFSET,
                str(size),
                ha="center",
                va="bottom",
                fontsize=CLUSTER_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Кластер", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Количество компаний", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Размеры кластеров",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA, axis="y")

    def _plot_cluster_profiles(self, ax):
        """Визуализация профилей кластеров"""
        ax.axis("off")

        if self.cluster_profiles:
            profile_text = "ПРОФИЛИ КЛАСТЕРОВ:\n\n"
            for profile in self.cluster_profiles[:5]:
                profile_text += f"Кластер {profile.cluster_id} ({profile.size} шт.):\n"
                profile_text += f"  • {profile.description}\n"
                profile_text += (
                    f"  • P/E: {profile.avg_pe:.1f} | P/B: {profile.avg_pb:.2f}\n"
                )
                profile_text += f"  • ROE: {profile.avg_roe:.1f}% | Див.: {profile.avg_div_yield:.1f}%\n"
                profile_text += f"  • Рекомендация: {profile.recommendation}\n\n"

            ax.text(
                0.05,
                0.95,
                profile_text,
                transform=ax.transAxes,
                fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE,
                verticalalignment="top",
                family="monospace",
                bbox=dict(
                    boxstyle="round",
                    facecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_BG,
                    edgecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_EDGE,
                    alpha=0.9,
                ),
            )


# ==================== КЛАСС ФУНДАМЕНТАЛЬНОГО АНАЛИЗА ====================


class FundamentalAnalyzer:
    """Расчет фундаментальных метрик и скоринга"""

    @staticmethod
    def calculate_value_score(row: pd.Series) -> float:
        """Расчет скора стоимости (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("PE")) and row["PE"] > 0:
            if row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_STRONG:
                score += SCORING.PE_DEEP_VALUE_BONUS
            elif row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_MEDIUM:
                score += SCORING.PE_VALUE_BONUS
            elif row["PE"] < CLUSTER_THRESHOLDS.SCORE_PE_WEAK:
                score += SCORING.PE_FAIR_BONUS
            elif row["PE"] > CLUSTER_THRESHOLDS.SCORE_PE_OVER:
                score += SCORING.PE_OVER_PENALTY

        if pd.notna(row.get("PB")) and row["PB"] > 0:
            if row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_STRONG:
                score += SCORING.PB_DEEP_VALUE_BONUS
            elif row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_MEDIUM:
                score += SCORING.PB_VALUE_BONUS
            elif row["PB"] < CLUSTER_THRESHOLDS.SCORE_PB_WEAK:
                score += SCORING.PB_FAIR_BONUS
            elif row["PB"] > CLUSTER_THRESHOLDS.SCORE_PB_OVER:
                score += SCORING.PB_OVER_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_quality_score(row: pd.Series) -> float:
        """Расчет скора качества (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_HIGH:
                score += SCORING.ROE_HIGH_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_GOOD:
                score += SCORING.ROE_GOOD_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_MEDIUM:
                score += SCORING.ROE_MEDIUM_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_LOW:
                score += SCORING.ROE_LOW_BONUS
            elif row["ROE"] < 0:
                score += SCORING.ROE_NEGATIVE_PENALTY

        if pd.notna(row.get("Debt_Capital")):
            if row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_LOW:
                score += SCORING.DEBT_LOW_BONUS
            elif row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_MEDIUM:
                score += SCORING.DEBT_MEDIUM_BONUS
            elif row["Debt_Capital"] < CLUSTER_THRESHOLDS.SCORE_DEBT_HIGH:
                score += SCORING.DEBT_HIGH_BONUS
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_CRITICAL:
                score += SCORING.DEBT_CRITICAL_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_growth_score(row: pd.Series) -> float:
        """Расчет скора роста (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_GOOD:
                score += SCORING.ROE_GOOD_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_MEDIUM:
                score += SCORING.ROE_MEDIUM_BONUS
            elif row["ROE"] > CLUSTER_THRESHOLDS.SCORE_ROE_LOW:
                score += SCORING.ROE_LOW_BONUS

        if pd.notna(row.get("PS")) and row["PS"] > 0:
            if row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_OVER:
                score += SCORING.PS_HIGH_BONUS
            elif row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_WEAK:
                score += SCORING.PS_GOOD_BONUS
            elif row["PS"] > CLUSTER_THRESHOLDS.SCORE_PB_MEDIUM:
                score += SCORING.PS_MEDIUM_BONUS

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_income_score(row: pd.Series) -> float:
        """Расчет скора дивидендного дохода (0-100)"""
        score = SCORING.BASE_SCORE

        if pd.notna(row.get("Div_Yield")):
            dy = row["Div_Yield"]
            if dy > CLUSTER_THRESHOLDS.SCORE_DIV_HIGH:
                score += SCORING.DIV_HIGH_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_GOOD:
                score += SCORING.DIV_GOOD_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_MEDIUM:
                score += SCORING.DIV_MEDIUM_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_LOW:
                score += SCORING.DIV_LOW_BONUS
            elif dy > CLUSTER_THRESHOLDS.SCORE_DIV_POOR:
                score += SCORING.DIV_POOR_BONUS
            elif dy < CLUSTER_THRESHOLDS.SCORE_DIV_MIN:
                score += SCORING.DIV_MIN_PENALTY

        return max(SCORING.MIN_SCORE, min(SCORING.MAX_SCORE, score))

    @staticmethod
    def calculate_expected_return(row: pd.Series) -> float:
        """Расчет ожидаемой доходности"""
        base_return = PORTFOLIO_CLUSTER.BASE_EXPECTED_RETURN

        if pd.notna(row.get("PE")) and row["PE"] > 0:
            if row["PE"] < CLUSTER_THRESHOLDS.PE_DEEP_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_DEEP_PREMIUM
            elif row["PE"] < CLUSTER_THRESHOLDS.PE_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_VALUE_PREMIUM
            elif row["PE"] < CLUSTER_THRESHOLDS.PE_FAIR:
                base_return += RETURN_PREMIUMS_CLUSTER.PE_FAIR_PREMIUM

        if pd.notna(row.get("PB")) and row["PB"] > 0:
            if row["PB"] < CLUSTER_THRESHOLDS.PB_DEEP_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_DEEP_PREMIUM
            elif row["PB"] < CLUSTER_THRESHOLDS.PB_VALUE:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_VALUE_PREMIUM
            elif row["PB"] < CLUSTER_THRESHOLDS.PB_FAIR:
                base_return += RETURN_PREMIUMS_CLUSTER.PB_FAIR_PREMIUM

        if pd.notna(row.get("ROE")):
            if row["ROE"] > CLUSTER_THRESHOLDS.ROE_HIGH:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_HIGH_PREMIUM
            elif row["ROE"] > CLUSTER_THRESHOLDS.ROE_GOOD:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_GOOD_PREMIUM
            elif row["ROE"] > CLUSTER_THRESHOLDS.ROE_GOOD * 0.75:
                base_return += RETURN_PREMIUMS_CLUSTER.ROE_MEDIUM_PREMIUM

        if pd.notna(row.get("Div_Yield")):
            base_return += (
                row["Div_Yield"] / 100
            ) * RETURN_PREMIUMS_CLUSTER.DIVIDEND_PREMIUM_FACTOR

        return min(RETURN_PREMIUMS_CLUSTER.MAX_RETURN, base_return)

    @staticmethod
    def calculate_risk(row: pd.Series) -> float:
        """Расчет риска"""
        base_risk = PORTFOLIO_CLUSTER.BASE_RISK

        if pd.notna(row.get("Beta")):
            base_risk += (row["Beta"] - 1) * RISK_PREMIUMS_CLUSTER.BETA_RISK_FACTOR

        if pd.notna(row.get("Debt_Capital")):
            if row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_CRITICAL:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_CRITICAL_PENALTY
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_HIGH:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_HIGH_PENALTY
            elif row["Debt_Capital"] > CLUSTER_THRESHOLDS.SCORE_DEBT_MEDIUM:
                base_risk += RISK_PREMIUMS_CLUSTER.DEBT_MEDIUM_PENALTY

        if pd.notna(row.get("PE")):
            if row["PE"] < 0 or row["PE"] > CLUSTER_THRESHOLDS.PE_GROWTH * 3:
                base_risk += RISK_PREMIUMS_CLUSTER.PE_EXTREME_PENALTY
            elif pd.isna(row["PE"]):
                base_risk += RISK_PREMIUMS_CLUSTER.PE_MISSING_PENALTY

        return max(
            PORTFOLIO_CLUSTER.MIN_RISK, min(PORTFOLIO_CLUSTER.MAX_RISK, base_risk)
        )


# ==================== КЛАСС ОПТИМИЗАТОРА ПОРТФЕЛЯ ====================


class PortfolioOptimizer:
    """Оптимизация портфеля по Марковицу с учетом кластеров"""

    def __init__(self, min_weight: float = None, max_weight: float = None):
        self.min_weight = min_weight or PORTFOLIO_CLUSTER.MIN_WEIGHT
        self.max_weight = max_weight or PORTFOLIO_CLUSTER.MAX_WEIGHT

    def create_covariance_matrix(
        self,
        df: pd.DataFrame,
        intra_cluster_corr: float = None,
        inter_cluster_corr: float = None,
    ) -> np.ndarray:
        """Создание матрицы ковариации с учетом кластерной структуры"""
        intra_cluster_corr = (
            intra_cluster_corr or PORTFOLIO_CLUSTER.INTRA_CLUSTER_CORRELATION
        )
        inter_cluster_corr = (
            inter_cluster_corr or PORTFOLIO_CLUSTER.INTER_CLUSTER_CORRELATION
        )

        n = len(df)
        cov_matrix = np.zeros((n, n))
        risks = df["Risk"].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = risks[i] ** 2
                else:
                    correlation = (
                        intra_cluster_corr
                        if "Cluster" in df.columns
                        and df.iloc[i]["Cluster"] == df.iloc[j]["Cluster"]
                        else inter_cluster_corr
                    )
                    cov_matrix[i, j] = correlation * risks[i] * risks[j]

        return cov_matrix

    def optimize_multi_portfolio(
        self, df: pd.DataFrame, strategies: List[str] = None
    ) -> Dict:
        """Оптимизация нескольких портфелей для разных стратегий"""
        if strategies is None:
            strategies = list(PORTFOLIO_CLUSTER.DEFAULT_STRATEGIES)

        portfolios = {}
        strategy_map = {
            "aggressive": self._optimize_for_max_return,
            "conservative": self._optimize_for_min_risk,
            "balanced": self._optimize_balanced,
            "value": self._optimize_value_portfolio,
            "growth": self._optimize_growth_portfolio,
            "dividend": self._optimize_dividend_portfolio,
            "cluster_based": self._optimize_cluster_based,
        }

        name_map = {
            "aggressive": "Агрессивный",
            "conservative": "Консервативный",
            "balanced": "Сбалансированный",
            "value": "Стоимостной",
            "growth": "Роста",
            "dividend": "Дивидендный",
            "cluster_based": "Кластерный",
        }

        for strategy in strategies:
            try:
                if strategy in strategy_map:
                    weights = strategy_map[strategy](df)
                    portfolios[name_map[strategy]] = weights
            except Exception as e:
                print(f"   ⚠️ Ошибка для стратегии {strategy}: {e}")

        return portfolios

    def _optimize_for_max_return(self, df: pd.DataFrame) -> np.ndarray:
        """Оптимизация для максимальной доходности"""
        expected_returns = df["Expected_Return"].values
        weights = expected_returns / expected_returns.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_for_min_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Оптимизация для минимального риска"""
        risks = df["Risk"].values
        inv_risk = 1 / risks
        weights = inv_risk / inv_risk.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_balanced(self, df: pd.DataFrame) -> np.ndarray:
        """Сбалансированная оптимизация"""
        scores = (df["Value_Score"] + df["Quality_Score"]) / 2
        weights = scores / scores.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_value_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Стоимостной портфель"""
        weights = df["Value_Score"].values / df["Value_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_growth_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Портфель роста"""
        weights = df["Growth_Score"].values / df["Growth_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_dividend_portfolio(self, df: pd.DataFrame) -> np.ndarray:
        """Дивидендный портфель"""
        weights = df["Income_Score"].values / df["Income_Score"].values.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()

    def _optimize_cluster_based(self, df: pd.DataFrame) -> np.ndarray:
        """Кластерный портфель - равномерное распределение по кластерам"""
        n = len(df)
        weights = np.zeros(n)
        unique_clusters = df["Cluster"].unique()
        n_clusters = len(unique_clusters)

        for cluster in unique_clusters:
            cluster_indices = df[df["Cluster"] == cluster].index
            cluster_weight = 1 / n_clusters
            per_stock_weight = cluster_weight / len(cluster_indices)
            weights[cluster_indices] = per_stock_weight

        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights / weights.sum()


# ==================== КЛАСС ПОРТФЕЛЬНОГО МЕНЕДЖЕРА ====================


class PortfolioManager:
    """Управление портфелем и расчет метрик"""

    def __init__(self, name: str, df: pd.DataFrame, weights: np.ndarray):
        self.name = name
        self.df = df.copy()
        self.weights = weights
        self.df["Weight"] = weights
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> PortfolioMetrics:
        """Расчет метрик портфеля"""
        exp_return = np.sum(self.df["Expected_Return"] * self.weights)

        optimizer = PortfolioOptimizer()
        cov_matrix = optimizer.create_covariance_matrix(self.df)

        risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        sharpe = exp_return / risk if risk > 0 else 0

        hhi = np.sum(self.weights**2)
        n = len(self.weights)
        diversification = 1 - (hhi - 1 / n) / (1 - 1 / n) if n > 1 else 0

        return PortfolioMetrics(
            expected_return=exp_return,
            risk=risk,
            sharpe_ratio=sharpe,
            diversification_score=diversification,
        )

    def get_sector_allocation(self) -> pd.Series:
        """Распределение по секторам"""
        if "Sector" in self.df.columns:
            return self.df.groupby("Sector")["Weight"].sum()
        return pd.Series()

    def get_cluster_allocation(self) -> pd.Series:
        """Распределение по кластерам"""
        if "Cluster" in self.df.columns:
            return self.df.groupby("Cluster")["Weight"].sum()
        return pd.Series()

    def get_top_positions(self, n: int = None) -> pd.DataFrame:
        """Топ позиций по весу"""
        n = n or PORTFOLIO_CLUSTER.TOP_POSITIONS_N
        n = min(n, len(self.df))
        top_idx = np.argsort(self.weights)[::-1][:n]
        return self.df.iloc[top_idx].copy()


# ==================== КЛАСС ВИЗУАЛИЗАТОРА ====================


class PortfolioVisualizer:
    """Визуализация портфелей и результатов"""

    @staticmethod
    def plot_portfolio_comparison(
        portfolios: Dict[str, PortfolioManager],
        filename: str = None,
    ):
        """Сравнение различных портфелей"""
        if not portfolios:
            print("   ⚠️ Нет портфелей для визуализации")
            return

        if filename is None:
            filename = CLUSTER_PATHS["portfolio_comparison"]

        n_portfolios = len(portfolios)
        fig, axes = plt.subplots(2, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_COMPARISON)

        names = []
        returns = []
        risks = []
        sharpe = []

        for name, pm in portfolios.items():
            names.append(name)
            returns.append(pm.metrics.expected_return)
            risks.append(pm.metrics.risk)
            sharpe.append(pm.metrics.sharpe_ratio)

        PortfolioVisualizer._plot_risk_return_comparison(
            axes[0, 0], names, returns, risks, sharpe
        )
        PortfolioVisualizer._plot_sharpe_comparison(axes[0, 1], names, sharpe)
        PortfolioVisualizer._plot_diversification_comparison(
            axes[1, 0], names, portfolios
        )
        PortfolioVisualizer._plot_best_portfolio_summary(axes[1, 1], portfolios)

        plt.suptitle(
            "Сравнение инвестиционных портфелей",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=CLUSTER_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _plot_risk_return_comparison(ax, names, returns, risks, sharpe):
        """Визуализация сравнения риск-доходность"""
        scatter = ax.scatter(
            risks,
            returns,
            c=sharpe,
            cmap=CLUSTER_FORMAT.COLOR_SHARPE_CMAP,
            s=200,
            edgecolors="black",
            linewidths=1.5,
        )

        for i, name in enumerate(names):
            ax.annotate(
                name,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=CLUSTER_FORMAT.ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )

        ax.set_xlabel("Риск", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Ожидаемая доходность", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение портфелей: Risk-Return",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Коэффициент Шарпа", fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE)

    @staticmethod
    def _plot_sharpe_comparison(ax, names, sharpe):
        """Визуализация сравнения коэффициентов Шарпа"""
        sharpe_array = np.array(sharpe)
        sharpe_min, sharpe_max = sharpe_array.min(), sharpe_array.max()
        sharpe_range = sharpe_max - sharpe_min + 0.001

        colors = plt.cm.get_cmap(CLUSTER_FORMAT.COLOR_SHARPE_CMAP)(
            (sharpe_array - sharpe_min) / sharpe_range
        )

        bars = ax.bar(names, sharpe, color=colors, edgecolor="black")
        ax.axhline(
            y=1,
            color=CLUSTER_FORMAT.COLOR_SHARPE_TARGET,
            linestyle="--",
            alpha=CLUSTER.SCATTER_ALPHA,
            label="Целевой Шарп = 1",
        )
        ax.set_xlabel("Портфель", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Коэффициент Шарпа", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Сравнение коэффициентов Шарпа",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA, axis="y")

        for bar, value in zip(bars, sharpe):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=CLUSTER_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_diversification_comparison(ax, names, portfolios):
        """Визуализация сравнения диверсификации"""
        diversifications = [
            pm.metrics.diversification_score for pm in portfolios.values()
        ]
        bars = ax.bar(
            names,
            diversifications,
            color=CLUSTER_FORMAT.COLOR_DIVERSIFICATION,
            edgecolor="black",
        )
        ax.set_xlabel("Портфель", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Индекс диверсификации", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Уровень диверсификации",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.grid(True, alpha=CLUSTER.GRID_ALPHA, axis="y")

        for bar, value in zip(bars, diversifications):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=CLUSTER_FORMAT.BAR_TEXT_FONT_SIZE,
                fontweight="bold",
            )

    @staticmethod
    def _plot_best_portfolio_summary(ax, portfolios):
        """Визуализация информации о лучшем портфеле"""
        ax.axis("off")

        best_portfolio = max(portfolios.values(), key=lambda p: p.metrics.sharpe_ratio)
        top_n = min(PORTFOLIO_CLUSTER.TOP_POSITIONS_RECOMMEND, len(best_portfolio.df))
        top_positions = best_portfolio.get_top_positions(top_n)

        text = f"🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {best_portfolio.name}\n\n"
        text += f"Доходность: {best_portfolio.metrics.expected_return:.1%}\n"
        text += f"Риск: {best_portfolio.metrics.risk:.1%}\n"
        text += f"Шарп: {best_portfolio.metrics.sharpe_ratio:.2f}\n\n"
        text += f"ТОП-{top_n} ПОЗИЦИЙ:\n"

        for _, row in top_positions.iterrows():
            text += f"• {row['Ticker']}: {row['Weight']:.1%}\n"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

    @staticmethod
    def plot_cluster_portfolio_allocation(
        portfolio_manager: PortfolioManager, filename: str = None
    ):
        """Визуализация распределения портфеля по кластерам"""
        if filename is None:
            filename = CLUSTER_PATHS["cluster_allocation"]

        fig, axes = plt.subplots(1, 2, figsize=CLUSTER_FILES.FIGURE_SIZE_ALLOCATION)

        cluster_weights = portfolio_manager.get_cluster_allocation()

        if len(cluster_weights) == 0:
            print("   ⚠️ Нет данных о кластерах для визуализации")
            plt.close()
            return

        colors = plt.cm.get_cmap(CLUSTER_FORMAT.COLOR_CLUSTER_CMAP)(
            np.linspace(0, 1, len(cluster_weights))
        )

        explode = [CLUSTER_FORMAT.PIE_EXPLODE_FACTOR] * len(cluster_weights)

        axes[0].pie(
            cluster_weights.values,
            labels=[f"Кластер {int(i)}" for i in cluster_weights.index],
            autopct=CLUSTER_FORMAT.MATPLOTLIB_PERCENT,
            startangle=90,
            colors=colors,
            explode=explode,
        )
        axes[0].set_title(
            "Распределение по кластерам",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )

        axes[1].axis("off")

        text = PortfolioVisualizer._format_cluster_allocation_text(
            portfolio_manager, cluster_weights
        )

        axes[1].text(
            0.05,
            0.95,
            text,
            transform=axes[1].transAxes,
            fontsize=CLUSTER_FORMAT.LABEL_FONT_SIZE,
            verticalalignment="top",
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_BG,
                edgecolor=CLUSTER_FORMAT.COLOR_PORTFOLIO_EDGE,
                alpha=0.9,
            ),
        )

        plt.suptitle(
            f"Анализ портфеля: {portfolio_manager.name}",
            fontsize=CLUSTER_FORMAT.TITLE_FONT_SIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=CLUSTER_FILES.DPI, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _format_cluster_allocation_text(
        pm: PortfolioManager, cluster_weights: pd.Series
    ) -> str:
        """Форматирование текста для распределения по кластерам"""
        text = f"📊 СОСТАВ ПОРТФЕЛЯ ПО КЛАСТЕРАМ: {pm.name}\n\n"

        for cluster_id in cluster_weights.index:
            cluster_data = pm.df[pm.df["Cluster"] == cluster_id]
            weight = cluster_weights[cluster_id]

            text += f"Кластер {cluster_id} - {weight:.1%}\n"
            text += f"  Компаний: {len(cluster_data)}\n"
            text += f"  Средний P/E: {cluster_data['PE'].mean():.1f}\n"
            text += f"  Средний ROE: {cluster_data['ROE'].mean():.1f}%\n"

            if len(cluster_data) > 0:
                top_n = min(PORTFOLIO_CLUSTER.TOP_IN_CLUSTER_N, len(cluster_data))
                top_in_cluster = cluster_data.nlargest(top_n, "Weight")
                text += f"  Топ: {top_in_cluster.iloc[0]['Ticker']} ({top_in_cluster.iloc[0]['Weight']:.1%})"
                if len(top_in_cluster) > 1:
                    text += f", {top_in_cluster.iloc[1]['Ticker']} ({top_in_cluster.iloc[1]['Weight']:.1%})"
            text += "\n\n"

        return text


# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================


class ReportGenerator:
    """Генерация отчетов и рекомендаций"""

    @staticmethod
    def generate_full_report(
        portfolios: Dict[str, PortfolioManager],
        cluster_profiles: List[ClusterCharacteristics],
        df_original: pd.DataFrame,
        filename: str = None,
    ):
        """Генерация полного отчета"""
        if filename is None:
            filename = CLUSTER_PATHS["investment_cluster_report"]

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            ReportGenerator._write_portfolio_summary(writer, portfolios)
            ReportGenerator._write_portfolio_details(writer, portfolios)
            ReportGenerator._write_cluster_profiles(writer, cluster_profiles)
            ReportGenerator._write_all_companies(writer, df_original)

        print(f"   ✅ Отчет сохранен: {filename}")

    @staticmethod
    def _write_portfolio_summary(writer, portfolios):
        """Запись сводки по портфелям"""
        if portfolios:
            summary_data = []
            for name, pm in portfolios.items():
                summary_data.append(
                    {
                        CLUSTER_REPORT.COL_PORTFOLIO: name,
                        CLUSTER_REPORT.COL_RETURN: f"{pm.metrics.expected_return:.2%}",
                        CLUSTER_REPORT.COL_RISK: f"{pm.metrics.risk:.2%}",
                        CLUSTER_REPORT.COL_SHARPE: f"{pm.metrics.sharpe_ratio:.2f}",
                        CLUSTER_REPORT.COL_DIVERSIFICATION: f"{pm.metrics.diversification_score:.2%}",
                        CLUSTER_REPORT.COL_N_POSITIONS: len(pm.df),
                    }
                )

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name=CLUSTER_FILES.SHEET_PORTFOLIO_SUMMARY, index=False
            )

    @staticmethod
    def _write_portfolio_details(writer, portfolios):
        """Запись детальной информации по каждому портфелю"""
        for name, pm in portfolios.items():
            portfolio_df = pm.df.sort_values("Weight", ascending=False)
            cols = [
                "Ticker",
                "Company",
                "Sector",
                "Cluster",
                "Weight",
                "Expected_Return",
                "Risk",
                "PE",
                "PB",
                "ROE",
                "Div_Yield",
                "Value_Score",
                "Quality_Score",
            ]
            available_cols = [c for c in cols if c in portfolio_df.columns]

            portfolio_display = portfolio_df[available_cols].copy()
            sheet_name = f"Портфель_{name[:12]}"
            portfolio_display.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def _write_cluster_profiles(writer, cluster_profiles):
        """Запись профилей кластеров"""
        if cluster_profiles:
            cluster_data = []
            for profile in cluster_profiles:
                cluster_data.append(
                    {
                        CLUSTER_REPORT.COL_CLUSTER: profile.cluster_id,
                        CLUSTER_REPORT.COL_CLUSTER_SIZE: profile.size,
                        CLUSTER_REPORT.COL_AVG_PE: f"{profile.avg_pe:.1f}",
                        CLUSTER_REPORT.COL_AVG_PB: f"{profile.avg_pb:.2f}",
                        CLUSTER_REPORT.COL_AVG_ROE: f"{profile.avg_roe:.1f}%",
                        CLUSTER_REPORT.COL_AVG_DIV: f"{profile.avg_div_yield:.1f}%",
                        CLUSTER_REPORT.COL_RISK_CLUSTER: f"{profile.avg_risk:.1%}",
                        CLUSTER_REPORT.COL_DESCRIPTION: profile.description,
                        CLUSTER_REPORT.COL_RECOMMENDATION: profile.recommendation,
                    }
                )

            pd.DataFrame(cluster_data).to_excel(
                writer, sheet_name=CLUSTER_FILES.SHEET_CLUSTERS, index=False
            )

    @staticmethod
    def _write_all_companies(writer, df_original):
        """Запись всех компаний с кластерами"""
        df_original.to_excel(
            writer, sheet_name=CLUSTER_FILES.SHEET_ALL_COMPANIES, index=False
        )


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
