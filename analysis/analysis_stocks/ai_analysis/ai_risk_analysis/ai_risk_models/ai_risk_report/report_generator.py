# ==================== КЛАСС ФОРМИРОВАТЕЛЯ ОТЧЕТОВ ====================


from typing import Dict
import pandas as pd
from ai_risk_models.ai_risk_constants.ai_risk_constants import NN_FILES
from ai_risk_models.ai_risk_portfolio.ai_risk_portfolio_manager import (
    NNRiskPortfolioManager,
)


class NNRiskReportGenerator:
    """Генерация отчетов для нейросетевого анализа рисков"""

    @staticmethod
    def generate_full_report(
        df_with_risk: pd.DataFrame,
        candidates: pd.DataFrame,
        portfolios: Dict[str, NNRiskPortfolioManager],
        filename: str = None,
    ):
        """Генерация полного отчета"""
        if filename is None:
            filename = NN_RISK_PATHS["nn_risk_portfolio_results"]

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_with_risk.to_excel(
                writer, sheet_name=NN_FILES.SHEET_STOCKS_WITH_RISK, index=False
            )

            candidates.to_excel(
                writer, sheet_name=NN_FILES.SHEET_CANDIDATES, index=False
            )

            if portfolios:
                NNRiskReportGenerator._write_portfolio_summary(writer, portfolios)

                best_portfolio = max(
                    portfolios.values(), key=lambda p: p.metrics.sharpe_ratio
                )
                best_portfolio.df.to_excel(
                    writer, sheet_name=NN_FILES.SHEET_BEST_PORTFOLIO, index=False
                )

        print(f"   ✅ Результаты сохранены в {filename}")

    @staticmethod
    def _write_portfolio_summary(writer, portfolios):
        """Запись сводки по портфелям"""
        portfolio_summary = []
        for name, pm in portfolios.items():
            portfolio_summary.append(
                {
                    "Портфель": name,
                    "Доходность": f"{pm.metrics.expected_return:.2%}",
                    "Риск": f"{pm.metrics.risk:.2%}",
                    "Шарп": f"{pm.metrics.sharpe_ratio:.2f}",
                    "VaR": f"{pm.metrics.var_95:.2%}",
                    "CVaR": f"{pm.metrics.cvar_95:.2%}",
                    "Диверсификация": f"{pm.metrics.diversification_score:.1%}",
                    "Позиций": len(pm.df),
                }
            )

        pd.DataFrame(portfolio_summary).to_excel(
            writer, sheet_name=NN_FILES.SHEET_PORTFOLIO_SUMMARY, index=False
        )
