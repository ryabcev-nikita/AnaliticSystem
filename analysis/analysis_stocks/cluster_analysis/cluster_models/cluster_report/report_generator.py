# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================
from typing import Dict, List
import pandas as pd

from ...cluster_models.cluster_analyzer.cluster_analyzer import ClusterCharacteristics
from ...cluster_models.cluster_constants.cluster_constants import (
    CLUSTER_FILES,
    CLUSTER_REPORT,
)
from ...cluster_models.cluster_portfolio.portfolio_optimizer import (
    PortfolioManager,
)


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
