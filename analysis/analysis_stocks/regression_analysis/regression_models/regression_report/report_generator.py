"""
Класс для генерации итогового отчета
"""

import pandas as pd
import numpy as np
from ..regression_constants.multiplicator_constants import (
    OUTPUT_FILES,
)
import os
import traceback


class ReportGenerator:
    """Класс для генерации отчетов по результатам анализа"""

    def __init__(
        self,
        models,
        robust_models,
        selected_stocks,
        optimal_portfolio,
        portfolio_metrics,
        output_dir="./results/",
    ):
        """
        Инициализация генератора отчетов

        Parameters:
        models (dict): словарь с OLS моделями
        robust_models (dict): словарь с робастными моделями
        selected_stocks (pd.DataFrame): отобранные акции
        optimal_portfolio (pd.DataFrame): оптимальный портфель
        portfolio_metrics (dict): метрики портфеля
        output_dir (str): директория для сохранения результатов
        """
        self.models = models if models is not None else {}
        self.robust_models = robust_models if robust_models is not None else {}
        self.selected_stocks = (
            selected_stocks if selected_stocks is not None else pd.DataFrame()
        )
        self.optimal_portfolio = (
            optimal_portfolio if optimal_portfolio is not None else pd.DataFrame()
        )
        self.portfolio_metrics = (
            portfolio_metrics if portfolio_metrics is not None else {}
        )
        self.output_dir = output_dir

    def get_rlm_metrics(self, model):
        """Безопасное получение метрик для RLM модели"""
        metrics = {}

        # Доступные атрибуты для RLM модели
        try:
            metrics["nobs"] = getattr(model, "nobs", "N/A")
        except:
            metrics["nobs"] = "N/A"

        try:
            metrics["df_resid"] = getattr(model, "df_resid", "N/A")
        except:
            metrics["df_resid"] = "N/A"

        try:
            metrics["iter"] = getattr(model, "iter", "N/A")
        except:
            metrics["iter"] = "N/A"

        # Для RLM модели используем другие метрики вместо deviance
        try:
            # Пробуем получить scale (масштаб остатков)
            metrics["scale"] = model.scale if hasattr(model, "scale") else "N/A"
        except:
            metrics["scale"] = "N/A"

        try:
            # Пробуем получить ковариационную матрицу
            if hasattr(model, "cov_params"):
                metrics["cov_params"] = "Доступна"
            else:
                metrics["cov_params"] = "Недоступна"
        except:
            metrics["cov_params"] = "Недоступна"

        return metrics

    def safe_get_model_params(self, model):
        """Безопасное получение параметров модели"""
        try:
            if hasattr(model, "params"):
                params = model.params
                # Проверяем, не пустой ли объект
                if params is not None:
                    # Для pandas Series используем .empty
                    if hasattr(params, "empty"):
                        if not params.empty:
                            return params
                    # Для словаря или другого контейнера
                    elif len(params) > 0:
                        return params
            return {}
        except:
            return {}

    def safe_series_to_dict(self, series):
        """Безопасное преобразование Series в словарь"""
        try:
            if series is None:
                return {}
            if hasattr(series, "to_dict"):
                return series.to_dict()
            if isinstance(series, dict):
                return series
            return {}
        except:
            return {}

    def generate_text_report(self):
        """Генерация текстового отчета"""
        try:
            report_lines = []

            report_lines.append("=" * 80)
            report_lines.append(
                "ИТОГОВЫЙ ОТЧЕТ ПО РЕГРЕССИОННОМУ АНАЛИЗУ И ФОРМИРОВАНИЮ ПОРТФЕЛЯ"
            )
            report_lines.append("=" * 80)

            # 1. Информация об обработке данных
            report_lines.append("\n1. ОБРАБОТКА ДАННЫХ:")
            report_lines.append("-" * 50)
            report_lines.append("✓ Выбросы удалены методом IQR до заполнения пропусков")
            report_lines.append("✓ Пропуски заполнены медианными значениями")
            report_lines.append("✓ Применены логарифмические преобразования")

            # 2. Результаты регрессионного анализа
            report_lines.append("\n2. РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА:")
            report_lines.append("-" * 50)

            # OLS модели
            report_lines.append("\n2.1 OLS МОДЕЛИ:")
            ols_found = False
            for model_name in ["pe_ols", "pbv_ols", "ps_ols", "ev_ols"]:
                if model_name in self.models:
                    ols_found = True
                    model = self.models[model_name]
                    report_lines.append(f"\n{model_name.upper()}:")
                    try:
                        report_lines.append(f"  R-squared: {model.rsquared:.4f}")
                        report_lines.append(
                            f"  Adj. R-squared: {model.rsquared_adj:.4f}"
                        )
                        report_lines.append(
                            f"  Количество наблюдений: {int(model.nobs) if hasattr(model, 'nobs') else 'N/A'}"
                        )

                        # Коэффициенты
                        params = self.safe_get_model_params(model)
                        if params is not None and len(params) > 0:
                            params_dict = self.safe_series_to_dict(params)
                            for param_name, param_value in params_dict.items():
                                report_lines.append(
                                    f"  {param_name}: {param_value:.4f}"
                                )
                    except Exception as e:
                        report_lines.append(f"  Ошибка получения метрик: {str(e)}")

            if not ols_found:
                report_lines.append("\n  OLS модели не были построены")

            # Робастные модели
            report_lines.append("\n2.2 РОБАСТНЫЕ МОДЕЛИ (Huber):")
            rlm_found = False
            for model_name, model in self.robust_models.items():
                rlm_found = True
                report_lines.append(f"\n{model_name.upper()}:")

                # Получаем метрики безопасно
                metrics = self.get_rlm_metrics(model)

                report_lines.append(f"  Количество наблюдений: {metrics['nobs']}")
                report_lines.append(f"  Число итераций: {metrics['iter']}")
                if metrics["scale"] != "N/A":
                    report_lines.append(
                        f"  Масштаб остатков (scale): {metrics['scale']:.4f}"
                    )
                else:
                    report_lines.append(
                        f"  Масштаб остатков (scale): {metrics['scale']}"
                    )

                # Коэффициенты
                params = self.safe_get_model_params(model)
                if params is not None and len(params) > 0:
                    params_dict = self.safe_series_to_dict(params)
                    for param_name, param_value in params_dict.items():
                        report_lines.append(f"  {param_name}: {param_value:.4f}")
                else:
                    report_lines.append("  Коэффициенты не доступны")

                # Дополнительная информация о качестве модели
                try:
                    if hasattr(model, "weights"):
                        weights = model.weights
                        if weights is not None and len(weights) > 0:
                            report_lines.append(
                                f"  Средний вес наблюдений: {np.mean(weights):.4f}"
                            )
                            report_lines.append(
                                f"  Мин вес: {np.min(weights):.4f}, Макс вес: {np.max(weights):.4f}"
                            )
                except:
                    pass

            if not rlm_found:
                report_lines.append("\n  Робастные модели не были построены")

            # 3. Отобранные акции
            report_lines.append("\n3. ТОП-10 НАИБОЛЕЕ НЕДООЦЕНЕННЫХ АКЦИЙ:")
            report_lines.append("-" * 50)
            if self.selected_stocks is not None and not self.selected_stocks.empty:
                try:
                    # Проверяем наличие необходимых колонок
                    display_cols = []
                    for col in [
                        "Тикер",
                        "Название",
                        "PE",
                        "PBV",
                        "ROE",
                        "g",
                        "Composite_Score_Robust",
                    ]:
                        if col in self.selected_stocks.columns:
                            display_cols.append(col)

                    if display_cols:
                        top_10 = self.selected_stocks.head(10)[display_cols].to_string(
                            index=False
                        )
                        report_lines.append(top_10)
                    else:
                        report_lines.append("Нет доступных колонок для отображения")
                except Exception as e:
                    report_lines.append(
                        f"Ошибка при формировании списка акций: {str(e)}"
                    )
            else:
                report_lines.append("Нет данных об отобранных акциях")

            # 4. Оптимальный портфель
            report_lines.append("\n4. ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ:")
            report_lines.append("-" * 50)
            if self.optimal_portfolio is not None and not self.optimal_portfolio.empty:
                try:
                    portfolio_cols = []
                    for col in ["Тикер", "Название", "Вес_в_портфеле"]:
                        if col in self.optimal_portfolio.columns:
                            portfolio_cols.append(col)

                    if portfolio_cols:
                        portfolio_str = self.optimal_portfolio[
                            portfolio_cols
                        ].to_string(index=False)
                        report_lines.append(portfolio_str)
                    else:
                        report_lines.append(
                            "Нет доступных колонок для отображения портфеля"
                        )

                    if self.portfolio_metrics:
                        report_lines.append(f"\nХарактеристики портфеля:")
                        if "return" in self.portfolio_metrics:
                            report_lines.append(
                                f"Ожидаемая доходность: {self.portfolio_metrics['return']:.2%}"
                            )
                        if "risk" in self.portfolio_metrics:
                            report_lines.append(
                                f"Риск (волатильность): {self.portfolio_metrics['risk']:.2%}"
                            )
                        if "sharpe" in self.portfolio_metrics:
                            report_lines.append(
                                f"Коэффициент Шарпа: {self.portfolio_metrics['sharpe']:.2f}"
                            )
                except Exception as e:
                    report_lines.append(f"Ошибка при формировании портфеля: {str(e)}")
            else:
                report_lines.append("Не удалось сформировать оптимальный портфель")

            # 5. Сохраненные файлы
            report_lines.append("\n5. СОХРАНЕННЫЕ ФАЙЛЫ:")
            report_lines.append("-" * 50)
            files_found = False
            for filename in OUTPUT_FILES.values():
                file_path = os.path.join(self.output_dir, filename)
                if os.path.exists(file_path):
                    report_lines.append(f"✓ {filename}")
                    files_found = True
                else:
                    report_lines.append(f"✗ {filename} (файл не найден)")

            if not files_found:
                report_lines.append("  В директории результатов файлы не найдены")

            return "\n".join(report_lines)

        except Exception as e:
            error_msg = f"Критическая ошибка при генерации отчета: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def save_report(self):
        """Сохранение отчета в файл"""
        try:
            report = self.generate_text_report()

            # Создаем директорию, если не существует
            os.makedirs(self.output_dir, exist_ok=True)

            # Сохраняем отчет
            report_path = os.path.join(
                self.output_dir, OUTPUT_FILES["regression_results"]
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            print("\n" + "=" * 80)
            print("ИТОГОВЫЙ ОТЧЕТ:")
            print("=" * 80)
            print(report)

            print(f"\n✅ Отчет сохранен в: {report_path}")

        except Exception as e:
            print(f"❌ Ошибка при сохранении отчета: {str(e)}")
            traceback.print_exc()
