import numpy as np
from ...ai_anomaly_detector_models.ai_anomaly_constants.ai_anomaly_constants import (
    AE_FORMAT,
    AE_PORTFOLIO,
)


class AEDataUtils:
    """Утилиты для работы с данными автоэнкодера"""

    @staticmethod
    def add_ae_results_to_df(
        df,
        valid_indices,
        errors_np,
        is_anomaly,
        is_strong_anomaly,
        undervalued_scores,
        error_median,
    ):
        """Добавление результатов автоэнкодера в DataFrame"""

        df["AE_Ошибка_реконструкции"] = np.nan
        df["AE_Ошибка_нормализованная"] = np.nan
        df["AE_Аномалия"] = False
        df["AE_Сильная_аномалия"] = False
        df["AE_Недооцененность"] = np.nan
        df["AE_Ранг_недооцененности"] = np.nan
        df["AE_Топ_недооцененные"] = False

        for i, idx in enumerate(valid_indices):
            df.at[idx, "AE_Ошибка_реконструкции"] = errors_np[i]
            df.at[idx, "AE_Ошибка_нормализованная"] = errors_np[i] / error_median
            df.at[idx, "AE_Аномалия"] = is_anomaly[i]
            df.at[idx, "AE_Сильная_аномалия"] = is_strong_anomaly[i]
            df.at[idx, "AE_Недооцененность"] = undervalued_scores[i]

        if undervalued_scores:
            scores_array = np.array(undervalued_scores)
            for i, idx in enumerate(valid_indices):
                percentile = (
                    (scores_array <= scores_array[i]).sum() / len(scores_array) * 100
                )
                df.at[idx, "AE_Ранг_недооцененности"] = percentile

        if len(undervalued_scores) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
            top_indices = np.argsort(undervalued_scores)[
                -AE_PORTFOLIO.TOP_UNDERVALUED_N :
            ][::-1]
            filtered_top = []

            for i in top_indices:
                if not is_anomaly[i]:
                    filtered_top.append(i)
                if len(filtered_top) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
                    break

            if len(filtered_top) < AE_PORTFOLIO.TOP_UNDERVALUED_N:
                for i in top_indices:
                    if i not in filtered_top:
                        filtered_top.append(i)
                    if len(filtered_top) >= AE_PORTFOLIO.TOP_UNDERVALUED_N:
                        break

            for i in filtered_top:
                df.at[valid_indices[i], "AE_Топ_недооцененные"] = True

        return df

    @staticmethod
    def print_top_undervalued(df):
        """Вывод топ-недооцененных акций"""
        print("\n" + AE_FORMAT.SEPARATOR)
        print("ТОП-10 НЕДООЦЕНЕННЫХ АКЦИЙ:")
        print(AE_FORMAT.SEPARATOR)

        undervalued_df = df[df["AE_Недооцененность"].notna()].copy()
        if not undervalued_df.empty:
            undervalued_df = undervalued_df.sort_values(
                "AE_Недооцененность", ascending=False
            )

            print(
                f"{'Тикер':<10} {'Название':<25} {'P/E':<6} {'P/B':<6} {'ДД,%':<6} "
                f"{'ROE,%':<7} {'Скор':<8} {'Ранг':<6}"
            )
            print(AE_FORMAT.SUB_SEPARATOR)

            for _, row in undervalued_df.head(15).iterrows():
                is_top = (
                    AE_FORMAT.STAR_SYMBOL
                    if row.get("AE_Топ_недооцененные", False)
                    else ""
                )
                print(
                    f"{row.get('Тикер', ''):<10} "
                    f"{str(row.get('Название', ''))[:23]:<25} "
                    f"{row.get('P_E', 0):<6.1f} "
                    f"{row.get('P_B', 0):<6.2f} "
                    f"{row.get('dividend_yield', 0)*100:<6.1f} "
                    f"{row.get('ROE', 0):<7.1f} "
                    f"{row.get('AE_Недооцененность', 0):<8.3f} "
                    f"{row.get('AE_Ранг_недооцененности', 0):<6.1f} {is_top}"
                )
