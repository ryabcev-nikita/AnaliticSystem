# ==================== КЛАСС КЛАСТЕРНОГО АНАЛИЗА ====================


from dataclasses import dataclass
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from ...cluster_models.cluster_loader.path_config import CLUSTER_PATHS
from ...cluster_models.cluster_constants.cluster_constants import (
    CLUSTER,
    CLUSTER_FILES,
    CLUSTER_FORMAT,
    CLUSTER_REPORT,
    CLUSTER_SCORES,
    CLUSTER_THRESHOLDS,
    PORTFOLIO_CLUSTER,
)


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
