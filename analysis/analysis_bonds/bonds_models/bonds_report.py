import os
from typing import Dict, List

import pandas as pd

from bonds_models.bonds_constants import (
    ANALYSIS_DATE,
    OPTIMIZATION_PARAMS,
    OUTPUT_FILES,
    RISK_LEVELS,
)
from bonds_models.bonds_models import Bond


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
