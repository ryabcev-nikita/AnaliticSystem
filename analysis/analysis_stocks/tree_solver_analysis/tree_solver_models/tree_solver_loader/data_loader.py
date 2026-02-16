# ==================== КЛАСС ЗАГРУЗЧИКА ДАННЫХ ====================
import re
import numpy as np
import pandas as pd
from ...tree_solver_models.tree_solver_constants.tree_solver_constants import CONVERSION


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
    def calculate_growth_rate(df: pd.DataFrame) -> pd.Series:
        """
        Расчет темпа роста g по формуле: (1 - Payout_ratio) * ROE

        Payout_ratio = Div_Yield * PE / 100 (при наличии данных)
        Если нет данных о дивидендах, используется отраслевой бенчмарк
        """
        df = df.copy()

        # Проверяем наличие необходимых колонок
        has_dividend_data = all(
            col in df.columns for col in ["Averange_dividend_yield", "P/E"]
        )
        has_roe = "ROE" in df.columns

        if not has_roe:
            print(
                "   ⚠️ Внимание: Отсутствуют данные ROE. g будет рассчитан с ограничениями."
            )
            return pd.Series(np.nan, index=df.index, name="g")

        # Расчет коэффициента выплат (payout ratio)
        if has_dividend_data:
            # Избегаем деления на ноль и некорректных значений
            payout_ratio = np.where(
                (df["P/E"] > 0) & (df["Averange_dividend_yield"].notna()),
                np.minimum(
                    df["Averange_dividend_yield"] * df["P/E"] / 100, 1.0
                ),  # Ограничиваем 1
                0.3,  # Значение по умолчанию (средний payout ratio на рынке)
            )
        else:
            # Если нет данных о дивидендах, используем средний payout ratio по рынку
            print(
                "   ℹ️ Нет данных о дивидендах, используется средний payout ratio (30%)"
            )
            payout_ratio = 0.3

        # Расчет темпа роста
        g = np.where(df["ROE"].notna(), (1 - payout_ratio) * df["ROE"], np.nan)

        # Корректировка на основе исторических данных (если доступны)
        # Здесь можно добавить более сложную логику с учетом отраслевых особенностей

        # Ограничиваем разумные значения (чтобы избежать выбросов)
        g = np.clip(g, -10, 50)  # Рост от -10% до 50%

        # Логарифмируем результат для информации
        valid_g_count = np.sum(~np.isnan(g))
        if valid_g_count > 0:
            print(f"   ✅ Рассчитан темп роста g для {valid_g_count} компаний")
            print(f"      Диапазон g: {np.nanmin(g):.1f}% - {np.nanmax(g):.1f}%")
            print(f"      Средний g: {np.nanmean(g):.1f}%")

        return pd.Series(g, index=df.index, name="g")

    @staticmethod
    def calculate_peg_ratio(df: pd.DataFrame) -> pd.Series:
        """
        Расчет PEG ratio (P/E to Growth)
        PEG = P/E / g, где g - темп роста в процентах
        """
        if "g" not in df.columns or "P/E" not in df.columns:
            return pd.Series(np.nan, index=df.index, name="PEG")

        peg = np.where((df["g"] > 0) & (df["P/E"] > 0), df["P/E"] / df["g"], np.nan)

        # Ограничиваем экстремальные значения
        peg = np.clip(peg, 0, 100)

        return pd.Series(peg, index=df.index, name="PEG")

    @staticmethod
    def validate_fundamental_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация фундаментальных данных и их корректировка
        """
        df = df.copy()

        # Проверка P/E (неотрицательное)
        if "P/E" in df.columns:
            df.loc[df["P/E"] < 0, "P/E"] = np.nan
            # Каппинг экстремальных значений
            pe_99 = df["P/E"].quantile(0.99)
            df.loc[df["P/E"] > pe_99, "P/E"] = pe_99

        # Проверка P/B (неотрицательное)
        if "P/B" in df.columns:
            df.loc[df["P/B"] < 0, "P/B"] = np.nan
            pb_99 = df["P/B"].quantile(0.99)
            df.loc[df["P/B"] > pb_99, "P/B"] = pb_99

        # Проверка ROE (в разумных пределах)
        if "ROE" in df.columns:
            df.loc[df["ROE"] > 100, "ROE"] = 100  # ROE не может быть > 100%
            df.loc[df["ROE"] < -50, "ROE"] = -50  # Ограничиваем убытки

        return df

    @staticmethod
    def add_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление метрик роста в DataFrame
        """
        df = df.copy()

        # Расчет темпа роста g
        df["g"] = DataLoader.calculate_growth_rate(df)

        # Расчет PEG ratio
        df["PEG"] = DataLoader.calculate_peg_ratio(df)

        # Категоризация по темпу роста
        def categorize_growth(g):
            if pd.isna(g):
                return "Не определен"
            elif g >= 20:
                return "Очень высокий рост"
            elif g >= 15:
                return "Высокий рост"
            elif g >= 10:
                return "Средний рост"
            elif g >= 5:
                return "Умеренный рост"
            elif g >= 0:
                return "Низкий рост"
            else:
                return "Спад"

        df["Growth_Category"] = df["g"].apply(categorize_growth)

        # Категоризация по PEG
        def categorize_peg(peg):
            if pd.isna(peg):
                return "Не определен"
            elif peg < 0.5:
                return "Сильно недооценен"
            elif peg < 1:
                return "Недооценен"
            elif peg < 1.5:
                return "Справедливая оценка"
            elif peg < 2:
                return "Переоценен"
            else:
                return "Сильно переоценен"

        df["PEG_Category"] = df["PEG"].apply(categorize_peg)

        # Статистика по добавленным метрикам
        print("\n📈 Метрики роста добавлены:")
        print(f"   • Компаний с рассчитанным g: {df['g'].notna().sum()} из {len(df)}")
        print(
            f"   • Компаний с рассчитанным PEG: {df['PEG'].notna().sum()} из {len(df)}"
        )

        print("\n   Распределение по категориям роста:")
        growth_dist = df["Growth_Category"].value_counts()
        for cat, count in growth_dist.items():
            if cat != "Не определен":
                print(f"     • {cat}: {count} компаний ({count/len(df)*100:.1f}%)")

        return df

    @staticmethod
    def load_and_clean_data(
        filepath: str, add_growth_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Загрузка и очистка данных

        Parameters:
        -----------
        filepath : str
            Путь к файлу с данными
        add_growth_metrics : bool
            Добавить ли метрики роста (g и PEG)
        """
        print(f"📂 Загрузка данных из файла: {filepath}")

        # Загрузка данных
        df = pd.read_excel(filepath, sheet_name="Sheet1")
        print(f"   Загружено {len(df)} строк")

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
            "Бета",
            "Дивиденд на акцию",
        ]

        # Конвертация числовых колонок
        converted_count = 0
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader.convert_to_float)
                converted_count += 1

        print(f"   Сконвертировано {converted_count} числовых колонок")

        # Валидация фундаментальных данных
        df = DataLoader.validate_fundamental_data(df)

        # Добавление метрик роста (опционально)
        if add_growth_metrics:
            df = DataLoader.add_growth_metrics(df)

        # Базовая статистика по пропускам
        total_cells = len(df) * len(numeric_columns)
        missing_cells = df[numeric_columns].isna().sum().sum()
        missing_percentage = (
            (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        )

        print(f"\n📊 Статистика по данным:")
        print(f"   • Всего компаний: {len(df)}")
        print(f"   • Числовых колонок: {len(numeric_columns)}")
        print(f"   • Пропусков в данных: {missing_percentage:.1f}%")

        return df

    @staticmethod
    def get_growth_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Получение сводной статистики по метрикам роста
        """
        if "g" not in df.columns:
            print(
                "   ⚠️ Метрики роста не рассчитаны. Сначала выполните load_and_clean_data с add_growth_metrics=True"
            )
            return pd.DataFrame()

        summary = pd.DataFrame(
            {
                "Метрика": ["Темп роста g (%)", "PEG ratio"],
                "Среднее": [df["g"].mean(), df["PEG"].mean()],
                "Медиана": [df["g"].median(), df["PEG"].median()],
                "Стд. отклонение": [df["g"].std(), df["PEG"].std()],
                "Минимум": [df["g"].min(), df["PEG"].min()],
                "Максимум": [df["g"].max(), df["PEG"].max()],
                "Количество": [df["g"].notna().sum(), df["PEG"].notna().sum()],
            }
        )

        # Добавляем информацию по секторам (если доступно)
        if "Сектор" in df.columns:
            sector_growth = (
                df.groupby("Сектор")["g"].agg(["mean", "median", "count"]).round(1)
            )
            sector_growth.columns = [
                "Средний g (%)",
                "Медианный g (%)",
                "Кол-во компаний",
            ]

            print("\n🏭 Средний темп роста по секторам:")
            print(sector_growth.to_string())

        return summary
