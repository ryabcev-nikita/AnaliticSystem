"""
Класс для регрессионного анализа акций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from constants.multiplicator_constants import (
    ANALYSIS_COLUMNS,
    PLOT_CONFIG,
    OUTPUT_FILES,
    ROBUST_REGRESSION,
    DATA_PROCESSING,
)
import os
import traceback


class RegressionAnalyzer:
    """Класс для проведения регрессионного анализа"""

    def __init__(self, data, output_dir="./results/"):
        """
        Инициализация анализатора

        Parameters:
        data (pd.DataFrame): обработанные данные
        output_dir (str): директория для сохранения результатов
        """
        self.data = data
        self.output_dir = output_dir
        self.models = {}
        self.robust_models = {}
        self.create_output_dir()

    def create_output_dir(self):
        """Создание директории для результатов"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_plot(self, filename):
        """Сохранение графика в файл"""
        try:
            plt.savefig(
                os.path.join(self.output_dir, filename),
                dpi=PLOT_CONFIG["dpi"],
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"Ошибка при сохранении графика {filename}: {e}")

    def correlation_analysis(self):
        """Проведение корреляционного анализа"""
        try:
            corr_columns = [
                col
                for col in ANALYSIS_COLUMNS["correlation"]
                if col in self.data.columns
            ]

            if len(corr_columns) < 2:
                print("Недостаточно данных для корреляционного анализа")
                return None

            correlation_matrix = self.data[corr_columns].corr()

            plt.figure(figsize=PLOT_CONFIG["correlation_figure"])
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                annot_kws={"size": 8},
            )
            plt.title("Корреляционная матрица финансовых показателей")
            plt.tight_layout()
            self.save_plot(OUTPUT_FILES["correlation_matrix"])
            plt.show()

            print("\nКорреляция с P/E:")
            if "PE" in correlation_matrix.columns:
                print(correlation_matrix["PE"].sort_values(ascending=False))

            return correlation_matrix
        except Exception as e:
            print(f"Ошибка при корреляционном анализе: {e}")
            traceback.print_exc()
            return None

    def analyze_pe_robust(self):
        """Робастная регрессия P/E от g"""
        try:
            # Подготовка данных
            valid_data = self.data[["log_g", "PE"]].dropna()

            if len(valid_data) < DATA_PROCESSING["min_data_points"]:
                print(
                    f"Недостаточно данных для робастной регрессии P/E: {len(valid_data)} точек"
                )
                return None

            X = valid_data[["log_g"]]
            X = sm.add_constant(X)
            y = valid_data["PE"]

            # Обычная регрессия для сравнения
            ols_model = sm.OLS(y, X).fit()
            self.models["pe_ols"] = ols_model

            # Робастная регрессия с Huber T norm
            try:
                rlm_model = RLM(y, X, M=HuberT()).fit(
                    maxiter=ROBUST_REGRESSION["max_iter"],
                    tol=ROBUST_REGRESSION["tolerance"],
                )
                self.robust_models["pe"] = rlm_model

                print("\nРезультаты робастной регрессии для P/E от log(g):")
                print(rlm_model.summary())

                # Сравнение коэффициентов
                print("\nСравнение коэффициентов:")
                print(
                    f"OLS: intercept={ols_model.params['const']:.4f}, slope={ols_model.params['log_g']:.4f}"
                )
                print(
                    f"RLM: intercept={rlm_model.params['const']:.4f}, slope={rlm_model.params['log_g']:.4f}"
                )

                # Диагностика весов
                weights = rlm_model.weights
                print(f"\nДиагностика весов робастной регрессии:")
                print(f"Минимальный вес: {weights.min():.4f}")
                print(f"Максимальный вес: {weights.max():.4f}")
                print(f"Медианный вес: {np.median(weights):.4f}")
                print(f"Количество точек с малым весом (<0.5): {(weights < 0.5).sum()}")

                # Визуализация
                self.plot_robust_regression_diagnostics(
                    valid_data, ols_model, rlm_model, "pe"
                )

                return rlm_model

            except Exception as e:
                print(f"Ошибка при робастной регрессии P/E: {e}")
                traceback.print_exc()
                return None

        except Exception as e:
            print(f"Ошибка в analyze_pe_robust: {e}")
            traceback.print_exc()
            return None

    def plot_robust_regression_diagnostics(
        self, data, ols_model, rlm_model, model_name
    ):
        """Построение диагностических графиков для робастной регрессии"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=PLOT_CONFIG["diagnostics_figure"])

            X = data["log_g"]
            y = data["PE"]

            # 1. Данные и линии регрессии
            axes[0, 0].scatter(X, y, alpha=0.6, label="Данные")

            # Сортируем X для гладких линий
            X_sorted = np.sort(X)
            X_pred = sm.add_constant(X_sorted)

            y_ols = ols_model.predict(X_pred)
            y_rlm = rlm_model.predict(X_pred)

            axes[0, 0].plot(X_sorted, y_ols, "r-", label="OLS", linewidth=2)
            axes[0, 0].plot(X_sorted, y_rlm, "g-", label="RLM (Huber)", linewidth=2)
            axes[0, 0].set_xlabel("log(g)")
            axes[0, 0].set_ylabel("P/E")
            axes[0, 0].set_title("Сравнение OLS и RLM")
            axes[0, 0].legend()

            # 2. Веса робастной регрессии
            scatter = axes[0, 1].scatter(
                X, rlm_model.weights, c=rlm_model.weights, cmap="viridis", alpha=0.6
            )
            axes[0, 1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
            axes[0, 1].set_xlabel("log(g)")
            axes[0, 1].set_ylabel("Вес")
            axes[0, 1].set_title("Веса наблюдений (RLM)")
            plt.colorbar(scatter, ax=axes[0, 1])

            # 3. Остатки OLS vs RLM
            axes[0, 2].scatter(
                ols_model.fittedvalues,
                ols_model.resid,
                alpha=0.6,
                label="OLS",
                marker="o",
            )
            axes[0, 2].scatter(
                rlm_model.fittedvalues,
                rlm_model.resid,
                alpha=0.6,
                label="RLM",
                marker="s",
            )
            axes[0, 2].axhline(y=0, color="r", linestyle="--")
            axes[0, 2].set_xlabel("Предсказанные значения")
            axes[0, 2].set_ylabel("Остатки")
            axes[0, 2].set_title("Остатки: OLS vs RLM")
            axes[0, 2].legend()

            # 4. Гистограмма остатков
            axes[1, 0].hist(
                ols_model.resid, bins=20, alpha=0.5, label="OLS", edgecolor="black"
            )
            axes[1, 0].hist(
                rlm_model.resid, bins=20, alpha=0.5, label="RLM", edgecolor="black"
            )
            axes[1, 0].set_xlabel("Остатки")
            axes[1, 0].set_ylabel("Частота")
            axes[1, 0].set_title("Распределение остатков")
            axes[1, 0].legend()

            # 5. Q-Q plot остатков RLM
            stats.probplot(rlm_model.resid, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title("Q-Q plot остатков (RLM)")

            # 6. Влияние наблюдений (Cook's distance для OLS)
            try:
                influence = ols_model.get_influence()
                cooks = influence.cooks_distance[0]
                axes[1, 2].stem(range(len(cooks)), cooks, markerfmt=",")
                axes[1, 2].set_xlabel("Наблюдение")
                axes[1, 2].set_ylabel("Cook's distance")
                axes[1, 2].set_title("Влияние наблюдений (OLS)")
            except:
                axes[1, 2].text(
                    0.5,
                    0.5,
                    "Не удалось рассчитать Cook's distance",
                    ha="center",
                    va="center",
                )

            plt.tight_layout()
            self.save_plot(OUTPUT_FILES["pe_robust_diagnostics"])
            plt.show()

        except Exception as e:
            print(f"Ошибка при построении диагностических графиков: {e}")
            traceback.print_exc()

    def analyze_pbv_robust(self):
        """Робастная регрессия P/BV от ROE"""
        try:
            valid_data = self.data[["ROE_scaled", "log_PBV"]].dropna()

            if len(valid_data) < DATA_PROCESSING["min_data_points"]:
                print(f"Недостаточно данных для робастной регрессии P/BV")
                return None

            X = valid_data[["ROE"]]
            X = sm.add_constant(X)
            y = valid_data["log_PBV"]

            # Обычная регрессия
            ols_model = sm.OLS(y, X).fit()
            self.models["pbv_ols"] = ols_model

            # Робастная регрессия
            try:
                rlm_model = RLM(y, X, M=HuberT()).fit(
                    maxiter=ROBUST_REGRESSION["max_iter"],
                    tol=ROBUST_REGRESSION["tolerance"],
                )
                self.robust_models["pbv"] = rlm_model

                print("\nРезультаты робастной регрессии для log(P/BV) от ROE:")
                print(rlm_model.summary())

                return rlm_model
            except Exception as e:
                print(f"Ошибка при робастной регрессии P/BV: {e}")
                return None

        except Exception as e:
            print(f"Ошибка в analyze_pbv_robust: {e}")
            return None

    def analyze_ps_robust(self):
        """Робастная регрессия P/S от NPM"""
        try:
            valid_data = self.data[["NPM_scaled", "log_PS"]].dropna()

            if len(valid_data) < DATA_PROCESSING["min_data_points"]:
                print(f"Недостаточно данных для робастной регрессии P/S")
                return None

            X = valid_data[["NPM_scaled"]]
            X = sm.add_constant(X)
            y = valid_data["log_PS"]

            # Обычная регрессия
            ols_model = sm.OLS(y, X).fit()
            self.models["ps_ols"] = ols_model

            # Робастная регрессия
            try:
                rlm_model = RLM(y, X, M=HuberT()).fit(
                    maxiter=ROBUST_REGRESSION["max_iter"],
                    tol=ROBUST_REGRESSION["tolerance"],
                )
                self.robust_models["ps"] = rlm_model

                print("\nРезультаты робастной регрессии для log(P/S) от NPM:")
                print(rlm_model.summary())

                return rlm_model
            except Exception as e:
                print(f"Ошибка при робастной регрессии P/S: {e}")
                return None

        except Exception as e:
            print(f"Ошибка в analyze_ps_robust: {e}")
            return None

    def analyze_ev_ebitda_robust(self):
        """Робастная множественная регрессия EV/EBITDA от g и ROIC"""
        try:
            valid_data = self.data[["log_g", "ROIC_scaled", "log_EV_EBITDA"]].dropna()

            if len(valid_data) < DATA_PROCESSING["min_data_points"]:
                print(f"Недостаточно данных для робастной регрессии EV/EBITDA")
                return None

            X = valid_data[["log_g", "ROIC_scaled"]]
            X = sm.add_constant(X)
            y = valid_data["log_EV_EBITDA"]

            # Проверка мультиколлинеарности
            try:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                ]
                print("\nVIF для проверки мультиколлинеарности:")
                print(vif_data)
            except:
                print("Не удалось рассчитать VIF")

            # Обычная регрессия
            ols_model = sm.OLS(y, X).fit()
            self.models["ev_ols"] = ols_model

            # Робастная регрессия
            try:
                rlm_model = RLM(y, X, M=HuberT()).fit(
                    maxiter=ROBUST_REGRESSION["max_iter"],
                    tol=ROBUST_REGRESSION["tolerance"],
                )
                self.robust_models["ev"] = rlm_model

                print(
                    "\nРезультаты робастной регрессии для log(EV/EBITDA) от g и ROIC:"
                )
                print(rlm_model.summary())

                return rlm_model
            except Exception as e:
                print(f"Ошибка при робастной регрессии EV/EBITDA: {e}")
                return None

        except Exception as e:
            print(f"Ошибка в analyze_ev_ebitda_robust: {e}")
            return None

    def run_all_analyses(self):
        """Запуск всех регрессионных анализов"""
        print("\n" + "=" * 50)
        print("РЕГРЕССИОННЫЙ АНАЛИЗ")
        print("=" * 50)

        # Корреляционный анализ
        print("\n1. Корреляционный анализ")
        self.correlation_analysis()

        # Регрессионный анализ P/E
        print("\n2. Анализ P/E")
        self.analyze_pe_robust()

        # Регрессионный анализ P/BV
        print("\n3. Анализ P/BV")
        self.analyze_pbv_robust()

        # Регрессионный анализ P/S
        print("\n4. Анализ P/S")
        self.analyze_ps_robust()

        # Регрессионный анализ EV/EBITDA
        print("\n5. Анализ EV/EBITDA")
        self.analyze_ev_ebitda_robust()

        # Сравнение моделей
        self.compare_models()

        return self.robust_models

    def compare_models(self):
        """Сравнение OLS и робастных моделей"""
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ МОДЕЛЕЙ OLS vs RLM")
        print("=" * 50)

        comparison_data = []

        for model_name in ["pe", "pbv", "ps", "ev"]:
            ols_key = f"{model_name}_ols"
            rlm_key = model_name

            if ols_key in self.models and rlm_key in self.robust_models:
                ols_model = self.models[ols_key]
                rlm_model = self.robust_models[rlm_key]

                # Для RLM используем доступные метрики
                try:
                    # Псевдо R-squared для RLM
                    rlm_r2 = (
                        1 - (rlm_model.deviance / rlm_model.null_deviance)
                        if rlm_model.null_deviance != 0
                        else 0
                    )

                    comparison_data.append(
                        {
                            "Модель": model_name.upper(),
                            "OLS R²": f"{ols_model.rsquared:.4f}",
                            "RLM R² (псевдо)": f"{rlm_r2:.4f}",
                            "RLM итерации": rlm_model.iter,
                            "Смещение коэффициентов": (
                                "Да"
                                if np.any(
                                    np.abs(ols_model.params - rlm_model.params) > 0.1
                                )
                                else "Нет"
                            ),
                        }
                    )
                except:
                    # Если не удалось получить deviance, используем другие метрики
                    comparison_data.append(
                        {
                            "Модель": model_name.upper(),
                            "OLS R²": f"{ols_model.rsquared:.4f}",
                            "RLM статус": "Успешно",
                            "RLM итерации": getattr(rlm_model, "iter", "N/A"),
                            "Смещение коэффициентов": (
                                "Да"
                                if np.any(
                                    np.abs(ols_model.params - rlm_model.params) > 0.1
                                )
                                else "Нет"
                            ),
                        }
                    )

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
        else:
            print("Нет данных для сравнения моделей")
