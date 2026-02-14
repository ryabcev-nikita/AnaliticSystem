"""
Класс для обработки данных акций
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants.multiplicator_constants import (
    DATA_PROCESSING,
    COLUMN_MAPPING,
    OUTPUT_FILES,
    PATHS,
)


class StockDataProcessor:
    """Класс для обработки и подготовки данных по акциям"""

    def __init__(self, file_path):
        """
        Инициализация процессора данных

        Parameters:
        file_path (str): путь к файлу с данными
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.outlier_stats = {}
        self.numeric_columns = DATA_PROCESSING["numeric_columns"]
        self.clip_bounds = DATA_PROCESSING["clip_lower_bounds"]
        self.outlier_threshold = DATA_PROCESSING["outlier_threshold"]

    def load_data(self):
        """Загрузка данных из Excel файла"""
        self.raw_data = pd.read_excel(self.file_path)
        print(f"Загружено {len(self.raw_data)} строк")
        return self

    def rename_columns(self):
        """Переименование колонок для удобства работы"""
        self.raw_data = self.raw_data.rename(columns=COLUMN_MAPPING)
        return self

    def convert_numeric(self):
        """Преобразование строковых чисел в числовой формат"""
        for col in self.numeric_columns:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_numeric(
                    self.raw_data[col].astype(str).str.replace(",", "."),
                    errors="coerce",
                )
        return self

    def calculate_payout_ratio(self):
        """Расчет коэффициента выплаты дивидендов"""
        self.raw_data["Payout_Ratio"] = np.where(
            (self.raw_data["EPS"] > 0) & (self.raw_data["EPS"].notna()),
            self.raw_data["Dividend_per_share"] / self.raw_data["EPS"],
            np.nan,
        )
        # Ограничиваем payout ratio от 0 до 1
        self.raw_data["Payout_Ratio"] = self.raw_data["Payout_Ratio"].clip(
            DATA_PROCESSING["payout_ratio_clip"][0],
            DATA_PROCESSING["payout_ratio_clip"][1],
        )
        return self

    def calculate_growth_rate(self):
        """Расчет темпа роста g = (1 - payout_ratio) * ROE"""
        self.raw_data["g"] = (
            (1 - self.raw_data["Payout_Ratio"]) * self.raw_data["ROE"] / 100
        )
        return self

    def calculate_peg(self):
        """Расчет PEG = P/E / g"""
        self.raw_data["PEG"] = np.where(
            (self.raw_data["PE"] > 0)
            & (self.raw_data["g"] > 0)
            & (self.raw_data["g"] < 100),
            self.raw_data["PE"] / self.raw_data["g"],
            np.nan,
        )
        return self

    def plot_boxplots_before_after(self, data_before, data_after, columns):
        """Построение boxplot для визуализации удаления выбросов"""
        n_cols = len(columns)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

        for i, col in enumerate(columns):
            # До удаления выбросов
            axes[0, i].boxplot(data_before[col].dropna())
            axes[0, i].set_title(f"{col} (до)")
            axes[0, i].set_ylabel("Значение")

            # После удаления выбросов
            axes[1, i].boxplot(data_after[col].dropna())
            axes[1, i].set_title(f"{col} (после)")
            axes[1, i].set_ylabel("Значение")

        plt.suptitle(
            "Сравнение распределений до и после удаления выбросов", fontsize=16
        )
        plt.tight_layout()

        # Сохранение
        output_dir = PATHS["output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            os.path.join(output_dir, OUTPUT_FILES["boxplots_before_after"]),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def remove_outliers_iqr(self):
        """
        Удаление выбросов методом межквартильного размаха (IQR)
        Сначала удаляем выбросы, потом заполняем пропуски
        """
        columns_to_check = ["PE", "PBV", "PS", "EV_EBITDA", "ROE", "g"]
        data_before = self.raw_data[columns_to_check].copy()

        for col in columns_to_check:
            if col in self.raw_data.columns:
                # Вычисляем квартили и IQR
                Q1 = self.raw_data[col].quantile(0.25)
                Q3 = self.raw_data[col].quantile(0.75)
                IQR = Q3 - Q1

                # Границы для выбросов
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR

                # Сохраняем статистику
                self.outlier_stats[col] = {
                    "Q1": Q1,
                    "Q3": Q3,
                    "IQR": IQR,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "n_outliers_before": (
                        (self.raw_data[col] < lower_bound)
                        | (self.raw_data[col] > upper_bound)
                    ).sum(),
                }

                # Заменяем выбросы на NaN (потом заполним медианой)
                self.raw_data.loc[self.raw_data[col] < lower_bound, col] = np.nan
                self.raw_data.loc[self.raw_data[col] > upper_bound, col] = np.nan

                print(
                    f"{col}: удалено {self.outlier_stats[col]['n_outliers_before']} выбросов "
                    f"(границы: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

        # Визуализация результатов удаления выбросов
        data_after = self.raw_data[columns_to_check].copy()
        self.plot_boxplots_before_after(data_before, data_after, columns_to_check)

        return self

    def remove_outliers_zscore(self):
        """
        Альтернативный метод: удаление выбросов по Z-score
        """
        columns_to_check = ["PE", "PBV", "PS", "EV_EBITDA", "ROE", "g"]
        data_before = self.raw_data[columns_to_check].copy()

        for col in columns_to_check:
            if col in self.raw_data.columns:
                # Вычисляем Z-score
                mean_val = self.raw_data[col].mean()
                std_val = self.raw_data[col].std()
                z_scores = np.abs((self.raw_data[col] - mean_val) / std_val)

                # Находим выбросы
                outlier_mask = z_scores > self.outlier_threshold

                self.outlier_stats[col] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n_outliers_before": outlier_mask.sum(),
                }

                # Заменяем выбросы на NaN
                self.raw_data.loc[outlier_mask, col] = np.nan

                print(
                    f"{col}: удалено {outlier_mask.sum()} выбросов "
                    f"(Z-score > {self.outlier_threshold})"
                )

        # Визуализация результатов
        data_after = self.raw_data[columns_to_check].copy()
        self.plot_boxplots_before_after(data_before, data_after, columns_to_check)

        return self

    def handle_missing_values(self):
        """
        Обработка пропущенных значений - заполнение медианой
        (выполняется ПОСЛЕ удаления выбросов)
        """
        print("\nЗаполнение пропущенных значений медианой...")

        for col in self.numeric_columns + ["g", "PEG", "Payout_Ratio"]:
            if col in self.raw_data.columns:
                n_missing = self.raw_data[col].isna().sum()
                if n_missing > 0:
                    median_val = self.raw_data[col].median()
                    self.raw_data[col] = self.raw_data[col].fillna(median_val)
                    print(
                        f"{col}: заполнено {n_missing} пропусков медианой {median_val:.4f}"
                    )

        return self

    def add_log_transformations(self):
        """Добавление логарифмических преобразований"""
        for col in ["PE", "PBV", "PS", "EV_EBITDA", "PEG", "g"]:
            if col in self.raw_data.columns and col in self.clip_bounds:
                log_col = f"log_{col}"
                # Клиппинг для избежания отрицательных значений при логарифмировании
                clipped_values = self.raw_data[col].clip(lower=self.clip_bounds[col])
                self.raw_data[log_col] = np.log1p(clipped_values)

        # Добавление масштабированных показателей
        self.raw_data["ROE_scaled"] = self.raw_data["ROE"] / 100
        self.raw_data["ROIC_scaled"] = self.raw_data["ROIC"] / 100
        self.raw_data["NPM_scaled"] = self.raw_data["NPM"] / 100

        return self

    def print_outlier_summary(self):
        """Вывод сводки по удаленным выбросам"""
        print("\n" + "=" * 50)
        print("СВОДКА ПО УДАЛЕНИЮ ВЫБРОСОВ")
        print("=" * 50)

        total_removed = 0
        for col, stats in self.outlier_stats.items():
            removed = stats.get("n_outliers_before", 0)
            total_removed += removed
            print(f"{col}: удалено {removed} выбросов")

        print(f"\nВсего удалено выбросов: {total_removed}")
        print(f"Осталось строк после обработки: {len(self.raw_data)}")
        print("=" * 50)

    def process(self, method="iqr"):
        """
        Полный цикл обработки данных

        Parameters:
        method (str): метод удаления выбросов ('iqr' или 'zscore')
        """
        print("\nНАЧАЛО ОБРАБОТКИ ДАННЫХ")
        print("=" * 50)

        self.load_data()
        self.rename_columns()
        self.convert_numeric()
        self.calculate_payout_ratio()
        self.calculate_growth_rate()
        self.calculate_peg()

        # Удаление выбросов (перед заполнением пропусков!)
        if method == "iqr":
            self.remove_outliers_iqr()
        else:
            self.remove_outliers_zscore()

        # Заполнение пропусков медианой (после удаления выбросов)
        self.handle_missing_values()

        # Логарифмические преобразования
        self.add_log_transformations()

        # Вывод статистики
        self.print_outlier_summary()

        print("\nОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА")
        print("=" * 50)

        return self

    def get_processed_data(self):
        """Получение обработанных данных"""
        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()
        return self.processed_data
