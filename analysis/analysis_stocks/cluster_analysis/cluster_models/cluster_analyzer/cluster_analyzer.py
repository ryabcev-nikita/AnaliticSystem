# ==================== КЛАСС КЛАСТЕРНОГО АНАЛИЗА ====================

from dataclasses import dataclass
from typing import List, Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    avg_g: float  # Средний темп роста в кластере
    avg_roe: float
    avg_payout: float  # Средний коэффициент выплат
    avg_div_yield: float
    avg_risk: float
    description: str
    recommendation: str
    growth_category: str  # Категория роста (высокий/средний/низкий)
    valuation_category: str  # Категория оценки (дорого/справедливо/дешево)


class ClusterAnalyzer:
    """
    Кластерный анализ компаний на основе мультипликаторов P/E и темпа роста g.

    Параметр g рассчитывается по формуле: g = (1 - Payout_ratio) * ROE
    где Payout_ratio = Div_Yield * PE / 100 (при наличии Div_Yield)
    """

    # Константы для категоризации
    GROWTH_CATEGORIES = {
        "high": {"threshold": 15, "name": "Высокий рост"},
        "medium": {"threshold": 8, "name": "Средний рост"},
        "low": {"threshold": 0, "name": "Низкий рост"},
    }

    VALUATION_CATEGORIES = {
        "overvalued": {"pe_threshold": 20, "name": "Переоценен"},
        "fair": {"pe_threshold": 12, "name": "Справедливая оценка"},
        "undervalued": {"pe_threshold": 0, "name": "Недооценен"},
    }

    def __init__(self, n_clusters: int = None):
        self.n_clusters = n_clusters or CLUSTER.DEFAULT_N_CLUSTERS
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=CLUSTER.PCA_COMPONENTS)
        self.cluster_profiles = None
        self.feature_names = ["PE", "g"]  # Основные признаки для кластеризации

    @staticmethod
    def calculate_growth_rate(df: pd.DataFrame) -> pd.Series:
        """
        Расчет темпа роста g по формуле: (1 - Payout_ratio) * ROE

        Payout_ratio = Div_Yield * PE / 100 (при наличии данных)
        Если нет данных о дивидендах, используется упрощенная оценка
        """
        df = df.copy()

        # Расчет коэффициента выплат (payout ratio)
        if all(col in df.columns for col in ["Div_Yield", "PE"]):
            # Избегаем деления на ноль и некорректных значений
            payout_ratio = np.where(
                (df["PE"] > 0) & (df["Div_Yield"].notna()),
                np.minimum(df["Div_Yield"] * df["PE"] / 100, 1.0),  # Ограничиваем 1
                0.3,  # Значение по умолчанию, если нет данных
            )
        else:
            # Если нет данных о дивидендах, используем средний payout ratio
            payout_ratio = 0.3

        # Расчет темпа роста
        g = np.where(df["ROE"].notna(), (1 - payout_ratio) * df["ROE"], np.nan)

        # Ограничиваем разумные значения (чтобы избежать выбросов)
        g = np.clip(g, -10, 50)  # Рост от -10% до 50%

        return pd.Series(g, index=df.index, name="g")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Подготовка признаков для кластеризации
        """
        df = df.copy()

        # Расчет темпа роста
        if "g" not in df.columns:
            df["g"] = self.calculate_growth_rate(df)

        # Выбираем только нужные признаки
        features = ["PE", "g"]

        # Удаляем строки с пропущенными значениями в ключевых признаках
        df_clean = df[df[features].notna().all(axis=1)].copy()

        # Логарифмическое преобразование для PE (часто имеет скошенное распределение)
        df_clean["PE_log"] = np.log1p(df_clean["PE"].clip(lower=0))

        # Добавляем преобразованные признаки
        features_with_transform = ["PE_log", "g"]

        return df_clean, features_with_transform

    def find_optimal_clusters(
        self, df: pd.DataFrame, max_clusters: int = None
    ) -> Tuple[int, pd.DataFrame]:
        """Поиск оптимального количества кластеров"""
        max_clusters = max_clusters or CLUSTER.MAX_CLUSTERS

        df_clean, features = self.prepare_features(df)

        if len(df_clean) < CLUSTER.MIN_DATA_FOR_CLUSTERING:
            print(
                f"   ⚠️ Недостаточно данных для кластеризации. "
                f"Доступно: {len(df_clean)} компаний"
            )
            return min(CLUSTER.MIN_CLUSTERS, len(df_clean)), pd.DataFrame()

        scaled_data = self.scaler.fit_transform(df_clean[features])

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

    def fit_predict(self, df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
        """
        Обучение модели и предсказание кластеров на основе P/E и g
        """
        if n_clusters:
            self.n_clusters = n_clusters

        df_clean, features = self.prepare_features(df)

        if "g" not in df_clean.columns:
            df_clean["g"] = self.calculate_growth_rate(df_clean)

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

        # Добавляем оригинальные значения для анализа
        df_clean["PE_original"] = df_clean["PE"]
        df_clean["g_original"] = df_clean["g"]

        # PCA для визуализации
        pca_result = self.pca.fit_transform(scaled_data)
        df_clean["PCA1"] = pca_result[:, 0]
        df_clean["PCA2"] = pca_result[:, 1]

        return df_clean

    def _categorize_growth(self, g: float) -> str:
        """Категоризация по темпу роста"""
        if g >= self.GROWTH_CATEGORIES["high"]["threshold"]:
            return self.GROWTH_CATEGORIES["high"]["name"]
        elif g >= self.GROWTH_CATEGORIES["medium"]["threshold"]:
            return self.GROWTH_CATEGORIES["medium"]["name"]
        else:
            return self.GROWTH_CATEGORIES["low"]["name"]

    def _categorize_valuation(self, pe: float) -> str:
        """Категоризация по оценке (P/E)"""
        if pe >= self.VALUATION_CATEGORIES["overvalued"]["pe_threshold"]:
            return self.VALUATION_CATEGORIES["overvalued"]["name"]
        elif pe >= self.VALUATION_CATEGORIES["fair"]["pe_threshold"]:
            return self.VALUATION_CATEGORIES["fair"]["name"]
        else:
            return self.VALUATION_CATEGORIES["undervalued"]["name"]

    def analyze_clusters(self, df: pd.DataFrame) -> List[ClusterCharacteristics]:
        """Анализ характеристик кластеров"""
        cluster_profiles = []

        for cluster_id in range(self.n_clusters):
            cluster_data = df[df["Cluster"] == cluster_id]

            if len(cluster_data) == 0:
                continue

            # Основные метрики
            avg_pe = (
                cluster_data["PE_original"].median()
                if "PE_original" in cluster_data
                else 0
            )
            avg_g = (
                cluster_data["g_original"].median()
                if "g_original" in cluster_data
                else 0
            )
            avg_roe = cluster_data["ROE"].median() if "ROE" in cluster_data else 0

            # Payout ratio
            if all(col in cluster_data.columns for col in ["Div_Yield", "PE"]):
                payout_ratios = np.where(
                    cluster_data["PE"] > 0,
                    np.minimum(
                        cluster_data["Div_Yield"] * cluster_data["PE"] / 100, 1.0
                    ),
                    0.3,
                )
                avg_payout = np.median(payout_ratios)
            else:
                avg_payout = 0.3

            avg_div = (
                cluster_data["Div_Yield"].median() if "Div_Yield" in cluster_data else 0
            )

            # Риск (упрощенная оценка на основе волатильности роста)
            risk_factors = []
            if "Beta" in cluster_data:
                risk_factors.append(cluster_data["Beta"].mean())

            # Добавляем риск на основе вариации роста
            if len(cluster_data) > 1 and "g_original" in cluster_data:
                g_std = cluster_data["g_original"].std()
                risk_factors.append(g_std / 20)  # Нормализуем

            avg_risk = (
                np.mean(risk_factors) if risk_factors else PORTFOLIO_CLUSTER.BASE_RISK
            )

            # Категоризация
            growth_category = self._categorize_growth(avg_g)
            valuation_category = self._categorize_valuation(avg_pe)

            description = self._describe_cluster(avg_pe, avg_g, avg_roe)
            recommendation = self._get_recommendation(avg_pe, avg_g, avg_roe, avg_div)

            cluster_profiles.append(
                ClusterCharacteristics(
                    cluster_id=cluster_id,
                    size=len(cluster_data),
                    avg_pe=avg_pe,
                    avg_g=avg_g,
                    avg_roe=avg_roe,
                    avg_payout=avg_payout,
                    avg_div_yield=avg_div,
                    avg_risk=avg_risk,
                    description=description,
                    recommendation=recommendation,
                    growth_category=growth_category,
                    valuation_category=valuation_category,
                )
            )

        self.cluster_profiles = cluster_profiles
        return cluster_profiles

    def _describe_cluster(self, avg_pe: float, avg_g: float, avg_roe: float) -> str:
        """Формирование описания кластера на основе P/E и g"""

        # PEG ratio (P/E to Growth)
        peg = avg_pe / max(avg_g, 1) if avg_g > 1 else float("inf")

        if peg < 0.5 and avg_g > 15:
            return "🔍 Deep Value Growth (недооцененный рост)"
        elif peg < 1 and avg_g > 10:
            return "📈 Value Growth (растущие недооцененные)"
        elif peg > 2 and avg_g > 15:
            return "⭐ Growth (растущие, но дорогие)"
        elif avg_pe < 10 and avg_g < 5:
            return "🏦 Value (дешевые, низкий рост)"
        elif avg_pe > 25 and avg_g > 20:
            return "🚀 High Growth (высокий рост)"
        elif avg_pe > 20 and avg_g < 5:
            return "⚠️ Overvalued (переоцененные, низкий рост)"
        else:
            return "⚖️ Balanced (сбалансированные)"

    def _get_recommendation(
        self, avg_pe: float, avg_g: float, avg_roe: float, avg_div: float
    ) -> str:
        """Формирование рекомендации на основе P/E и g"""
        score = 0

        # PEG ratio (чем ниже, тем лучше)
        peg = avg_pe / max(avg_g, 1) if avg_g > 1 else float("inf")

        if peg < 0.5:
            score += 4
        elif peg < 1:
            score += 2
        elif peg < 1.5:
            score += 1
        elif peg > 3:
            score -= 1

        # Дополнительные факторы
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
        self._plot_pe_g_clusters(df, axes[0, 1])  # Изменено с P/B vs ROE на P/E vs g
        self._plot_cluster_sizes(df, axes[1, 0])
        self._plot_cluster_profiles(axes[1, 1])

        plt.suptitle(
            "Кластерный анализ компаний по P/E и темпу роста g",
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

    def _plot_pe_g_clusters(self, df: pd.DataFrame, ax):
        """
        Визуализация распределения кластеров по P/E и g
        (заменяет старый метод _plot_pb_roe_clusters)
        """
        # Используем оригинальные значения для наглядности
        x_col = "PE_original" if "PE_original" in df.columns else "PE"
        y_col = "g_original" if "g_original" in df.columns else "g"

        for cluster_id in df["Cluster"].unique():
            cluster_data = df[df["Cluster"] == cluster_id]
            if x_col in cluster_data and y_col in cluster_data:
                ax.scatter(
                    cluster_data[x_col],
                    cluster_data[y_col],
                    label=f"Кластер {cluster_id}",
                    s=CLUSTER.SCATTER_POINT_SIZE,
                    alpha=CLUSTER.SCATTER_ALPHA,
                    edgecolors="black",
                    linewidths=0.5,
                )

        # Добавляем линии PEG ratio для ориентира
        x_range = np.linspace(0, df[x_col].max(), 100)
        peg_05 = x_range / 0.5  # PEG = 0.5 (сильно недооценено)
        peg_1 = x_range / 1.0  # PEG = 1 (справедливо)
        peg_2 = x_range / 2.0  # PEG = 2 (дорого)

        ax.plot(x_range, peg_2, "r--", alpha=0.5, label="PEG=2 (дорого)")
        ax.plot(x_range, peg_1, "g--", alpha=0.5, label="PEG=1 (справедливо)")
        ax.plot(x_range, peg_05, "b--", alpha=0.5, label="PEG=0.5 (недооценено)")

        ax.set_xlabel("P/E", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_ylabel("Темп роста g, %", fontsize=CLUSTER_FORMAT.AXIS_FONT_SIZE)
        ax.set_title(
            "Распределение кластеров: P/E vs g",
            fontsize=CLUSTER_FORMAT.SUBTITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_ylim(0, min(50, df[y_col].max() * 1.1))  # Ограничиваем для наглядности
        ax.legend(loc="upper left", fontsize=8)
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
        """Визуализация профилей кластеров с фокусом на P/E и g"""
        ax.axis("off")

        if self.cluster_profiles:
            profile_text = "ПРОФИЛИ КЛАСТЕРОВ (P/E vs g):\n\n"
            for profile in self.cluster_profiles[:5]:  # Показываем первые 5 кластеров
                profile_text += f"Кластер {profile.cluster_id} ({profile.size} шт.):\n"
                profile_text += f"  • {profile.description}\n"
                profile_text += (
                    f"  • P/E: {profile.avg_pe:.1f} | g: {profile.avg_g:.1f}%\n"
                )
                profile_text += f"  • PEG: {profile.avg_pe/max(profile.avg_g,1):.2f}\n"
                profile_text += f"  • Категория роста: {profile.growth_category}\n"
                profile_text += f"  • Категория оценки: {profile.valuation_category}\n"
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

    def get_pe_g_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Получение расширенного анализа по P/E и g для каждой компании
        """
        result_df = df.copy()

        if "g" not in result_df.columns:
            result_df["g"] = self.calculate_growth_rate(result_df)

        # Расчет PEG ratio
        result_df["PEG"] = np.where(
            result_df["g"] > 0, result_df["PE"] / result_df["g"], np.nan
        )

        # Категоризация
        result_df["Growth_Category"] = result_df["g"].apply(self._categorize_growth)
        result_df["Valuation_Category"] = result_df["PE"].apply(
            self._categorize_valuation
        )

        # Инвестиционная привлекательность (комбинированный score)
        result_df["Investment_Score"] = 0
        # Низкий PEG - хорошо
        result_df.loc[result_df["PEG"] < 0.5, "Investment_Score"] += 3
        result_df.loc[
            (result_df["PEG"] >= 0.5) & (result_df["PEG"] < 1), "Investment_Score"
        ] += 2
        result_df.loc[
            (result_df["PEG"] >= 1) & (result_df["PEG"] < 1.5), "Investment_Score"
        ] += 1
        result_df.loc[result_df["PEG"] > 2, "Investment_Score"] -= 1

        return result_df
