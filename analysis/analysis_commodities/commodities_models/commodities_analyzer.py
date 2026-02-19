from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from commodities_models.commodities_constants import (
    ENERGY,
    INDUSTRIAL_METALS,
    METAL_NAMES,
    PLOT_STYLE,
    PRECIOUS_METALS,
)


class MetalsCorrelationAnalyzer:
    """Класс для корреляционного анализа металлов"""

    def __init__(self):
        self.correlation_matrix = None

    def analyze(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Проведение корреляционного анализа"""
        self.correlation_matrix = returns.corr()
        return self.correlation_matrix

    def get_top_correlations(self, n: int = 5) -> List[Dict]:
        """Получение наиболее коррелированных пар"""
        corr_pairs = []
        matrix = self.correlation_matrix

        for i in range(len(matrix.columns)):
            for j in range(i + 1, len(matrix.columns)):
                corr_pairs.append(
                    {
                        "metal1": matrix.columns[i],
                        "metal2": matrix.columns[j],
                        "correlation": matrix.iloc[i, j],
                    }
                )

        return sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:n]

    def get_correlations_by_group(self) -> pd.DataFrame:
        """Получение корреляций по группам металлов"""
        if self.correlation_matrix is None:
            return pd.DataFrame()

        # Корреляции внутри групп
        precious_corr = (
            self.correlation_matrix.loc[PRECIOUS_METALS, PRECIOUS_METALS].mean().mean()
        )
        industrial_corr = (
            self.correlation_matrix.loc[INDUSTRIAL_METALS, INDUSTRIAL_METALS]
            .mean()
            .mean()
        )

        # Корреляции между группами
        precious_industrial = (
            self.correlation_matrix.loc[PRECIOUS_METALS, INDUSTRIAL_METALS]
            .mean()
            .mean()
        )
        precious_energy = (
            self.correlation_matrix.loc[PRECIOUS_METALS, ENERGY].mean().mean()
        )
        industrial_energy = (
            self.correlation_matrix.loc[INDUSTRIAL_METALS, ENERGY].mean().mean()
        )

        return pd.DataFrame(
            {
                "Group": ["Precious Metals", "Industrial Metals", "Energy"],
                "Intra-group Correlation": [precious_corr, industrial_corr, 1.0],
                "Correlation with Precious": [
                    1.0,
                    precious_industrial,
                    precious_energy,
                ],
                "Correlation with Industrial": [
                    precious_industrial,
                    1.0,
                    industrial_energy,
                ],
                "Correlation with Energy": [precious_energy, industrial_energy, 1.0],
            }
        )

    def plot(
        self, returns: pd.DataFrame, save_path: str, show_plot: bool = True
    ) -> None:
        """Визуализация корреляционной матрицы"""
        if self.correlation_matrix is None:
            self.analyze(returns)

        plt.figure(figsize=PLOT_STYLE["figure_size"]["large"])
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            xticklabels=[METAL_NAMES[x] for x in self.correlation_matrix.columns],
            yticklabels=[METAL_NAMES[x] for x in self.correlation_matrix.columns],
        )

        plt.title(
            "Correlation Matrix of Metals & Commodities",
            fontsize=PLOT_STYLE["title_fontsize"],
            pad=20,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        plt.savefig(save_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()
