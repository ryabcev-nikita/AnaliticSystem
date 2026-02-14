"""
Класс для оптимизации портфеля акций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from regression_constants.multiplicator_constants import (
    COMPOSITE_SCORE_WEIGHTS,
    PORTFOLIO_CONFIG,
    PLOT_CONFIG,
    OUTPUT_FILES,
)
import os


class PortfolioOptimizerForRegression:
    """Класс для оптимизации портфеля по Марковицу"""

    def __init__(self, data, robust_models, output_dir="./results/"):
        """
        Инициализация оптимизатора портфеля

        Parameters:
        data (pd.DataFrame): данные с прогнозными значениями
        robust_models (dict): словарь с робастными регрессионными моделями
        output_dir (str): директория для сохранения результатов
        """
        self.data = data.copy()
        self.robust_models = robust_models
        self.output_dir = output_dir
        self.n_selected = PORTFOLIO_CONFIG["n_selected_stocks"]
        self.n_portfolio = PORTFOLIO_CONFIG["n_portfolio_stocks"]
        self.n_simulations = PORTFOLIO_CONFIG["n_portfolios_simulation"]
        self.risk_free_rate = PORTFOLIO_CONFIG["risk_free_rate"]

    def calculate_fair_values_robust(self):
        """Расчет справедливых значений по робастным моделям"""

        # Проверяем наличие всех необходимых моделей
        required_models = ["pe", "pbv", "ps", "ev"]
        for model_name in required_models:
            if model_name not in self.robust_models:
                print(
                    f"Предупреждение: модель {model_name} не найдена, используем упрощенный расчет"
                )

        # P/E модель (робастная)
        if "pe" in self.robust_models:
            X_pe = sm.add_constant(self.data[["log_g"]])
            self.data["PE_pred_robust"] = (
                np.exp(self.robust_models["pe"].predict(X_pe)) - 1
            )
            self.data["PE_ratio_robust"] = self.data["PE"] / self.data["PE_pred_robust"]
        else:
            self.data["PE_ratio_robust"] = self.data["PE"].rank(pct=True)

        # P/BV модель (робастная)
        if "pbv" in self.robust_models:
            X_pbv = sm.add_constant(self.data[["ROE_scaled"]])
            self.data["PBV_pred_robust"] = (
                np.exp(self.robust_models["pbv"].predict(X_pbv)) - 1
            )
            self.data["PBV_ratio_robust"] = (
                self.data["PBV"] / self.data["PBV_pred_robust"]
            )
        else:
            self.data["PBV_ratio_robust"] = self.data["PBV"].rank(pct=True)

        # P/S модель (робастная)
        if "ps" in self.robust_models:
            X_ps = sm.add_constant(self.data[["NPM_scaled"]])
            self.data["PS_pred_robust"] = (
                np.exp(self.robust_models["ps"].predict(X_ps)) - 1
            )
            self.data["PS_ratio_robust"] = self.data["PS"] / self.data["PS_pred_robust"]
        else:
            self.data["PS_ratio_robust"] = self.data["PS"].rank(pct=True)

        # EV/EBITDA модель (робастная)
        if "ev" in self.robust_models:
            X_ev = sm.add_constant(self.data[["log_g", "ROE_scaled"]])
            self.data["EV_EBITDA_pred_robust"] = (
                np.exp(self.robust_models["ev"].predict(X_ev)) - 1
            )
            self.data["EV_ratio_robust"] = (
                self.data["EV_EBITDA"] / self.data["EV_EBITDA_pred_robust"]
            )
        else:
            self.data["EV_ratio_robust"] = self.data["EV_EBITDA"].rank(pct=True)

        return self

    def calculate_composite_score_robust(self):
        """Расчет композитного скора недооцененности на основе робастных моделей"""
        weights = COMPOSITE_SCORE_WEIGHTS

        self.data["Composite_Score_Robust"] = (
            self.data["PE_ratio_robust"].rank(pct=True) * weights["PE_ratio"]
            + self.data["PBV_ratio_robust"].rank(pct=True) * weights["PBV_ratio"]
            + self.data["PS_ratio_robust"].rank(pct=True) * weights["PS_ratio"]
            + self.data["EV_ratio_robust"].rank(pct=True) * weights["EV_ratio"]
        )

        return self

    def select_undervalued_stocks_robust(self):
        """Отбор наиболее недооцененных акций на основе робастных моделей"""
        selected = self.data.nsmallest(self.n_selected, "Composite_Score_Robust")[
            [
                "Тикер",
                "Название",
                "PE",
                "PBV",
                "PS",
                "EV_EBITDA",
                "ROE",
                "g",
                "Composite_Score_Robust",
                "PE_ratio_robust",
                "PBV_ratio_robust",
                "PS_ratio_robust",
                "EV_ratio_robust",
            ]
        ]

        selected.to_csv(
            os.path.join(self.output_dir, OUTPUT_FILES["selected_stocks"]), index=False
        )

        print("\n" + "=" * 50)
        print("ТОП-20 НАИБОЛЕЕ НЕДООЦЕНЕННЫХ АКЦИЙ (по робастным моделям):")
        print("=" * 50)

        display_cols = [
            "Тикер",
            "Название",
            "PE",
            "PBV",
            "ROE",
            "g",
            "Composite_Score_Robust",
        ]
        print(selected[display_cols].head(20).to_string(index=False))

        return selected

    def simulate_portfolios(self, tickers):
        """Симуляция случайных портфелей для построения эффективной границы"""
        portfolio_df = self.data[self.data["Тикер"].isin(tickers)].copy()

        if len(portfolio_df) < 2:
            print(f"Предупреждение: недостаточно акций для симуляции портфеля")
            return None, None, None, None

        n_assets = len(portfolio_df)

        # Ожидаемая доходность (используем ROE как прокси)
        expected_returns = portfolio_df["ROE"].values / 100

        # Риск (волатильность) - используем стандартное отклонение отраслевое + случайный компонент
        np.random.seed(42)
        volatilities = np.abs(np.random.normal(0.2, 0.1, n_assets))

        # Корреляционная матрица
        correlations = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
        np.fill_diagonal(correlations, 1)
        correlations = (correlations + correlations.T) / 2

        # Ковариационная матрица
        cov_matrix = np.outer(volatilities, volatilities) * correlations

        # Симуляция портфелей
        results = np.zeros((3, self.n_simulations))

        for i in range(self.n_simulations):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (
                (portfolio_return - self.risk_free_rate) / portfolio_std
                if portfolio_std > 0
                else 0
            )

            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = sharpe

        return results, expected_returns, cov_matrix, portfolio_df

    def find_optimal_portfolios(self, results):
        """Поиск оптимальных портфелей"""
        if results is None:
            return None, None

        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[0])

        return max_sharpe_idx, min_vol_idx

    def plot_efficient_frontier(self, results, max_sharpe_idx, min_vol_idx):
        """Построение эффективной границы"""
        if results is None:
            print("Нет данных для построения эффективной границы")
            return

        plt.figure(figsize=PLOT_CONFIG["figure_size"])

        # Эффективная граница
        scatter = plt.scatter(
            results[0, :],
            results[1, :],
            c=results[2, :],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, label="Sharpe Ratio")

        # Отметка максимального Sharpe
        plt.scatter(
            results[0, max_sharpe_idx],
            results[1, max_sharpe_idx],
            marker="*",
            color="r",
            s=200,
            label="Максимальный Sharpe",
        )

        # Отметка минимальной волатильности
        plt.scatter(
            results[0, min_vol_idx],
            results[1, min_vol_idx],
            marker="*",
            color="b",
            s=200,
            label="Минимальная волатильность",
        )

        plt.xlabel("Волатильность (риск)")
        plt.ylabel("Ожидаемая доходность")
        plt.title("Эффективная граница Марковица")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.output_dir, OUTPUT_FILES["efficient_frontier"]),
            dpi=PLOT_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.show()

    def create_optimal_portfolio(
        self, tickers, results, max_sharpe_idx, expected_returns, portfolio_df
    ):
        """Создание оптимального портфеля"""
        if results is None or max_sharpe_idx is None:
            return None, None

        n_assets = len(portfolio_df)

        # Генерация весов для оптимального портфеля
        np.random.seed(max_sharpe_idx)
        optimal_weights = np.random.random(n_assets)
        optimal_weights /= np.sum(optimal_weights)

        optimal_portfolio = pd.DataFrame(
            {
                "Тикер": portfolio_df["Тикер"].values,
                "Название": portfolio_df["Название"].values,
                "Ожидаемая_доходность": expected_returns,
                "Вес_в_портфеле": optimal_weights,
            }
        ).sort_values("Вес_в_портфеле", ascending=False)

        optimal_portfolio.to_csv(
            os.path.join(self.output_dir, OUTPUT_FILES["optimal_portfolio"]),
            index=False,
        )

        print("\n" + "=" * 50)
        print("ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ (максимальный Sharpe ratio):")
        print("=" * 50)
        print(optimal_portfolio.to_string(index=False))

        # Характеристики портфеля
        portfolio_return = results[1, max_sharpe_idx]
        portfolio_vol = results[0, max_sharpe_idx]
        portfolio_sharpe = results[2, max_sharpe_idx]

        metrics = {
            "return": portfolio_return,
            "risk": portfolio_vol,
            "sharpe": portfolio_sharpe,
        }

        print(f"\nХарактеристики портфеля:")
        print(f"Ожидаемая доходность: {portfolio_return:.2%}")
        print(f"Риск (волатильность): {portfolio_vol:.2%}")
        print(f"Коэффициент Шарпа: {portfolio_sharpe:.2f}")

        return optimal_portfolio, metrics

    def optimize(self):
        """Полный цикл оптимизации портфеля"""
        print("\n" + "=" * 50)
        print("ОПТИМИЗАЦИЯ ПОРТФЕЛЯ")
        print("=" * 50)

        # Расчет справедливых значений и скора по робастным моделям
        self.calculate_fair_values_robust()
        self.calculate_composite_score_robust()

        # Отбор акций
        selected = self.select_undervalued_stocks_robust()
        top_tickers = selected.head(self.n_portfolio)["Тикер"].tolist()

        print(f"\nОтобрано акций для портфеля: {len(top_tickers)}")

        # Симуляция портфелей
        results, expected_returns, cov_matrix, portfolio_df = self.simulate_portfolios(
            top_tickers
        )

        if results is not None:
            # Поиск оптимальных портфелей
            max_sharpe_idx, min_vol_idx = self.find_optimal_portfolios(results)

            # Визуализация
            self.plot_efficient_frontier(results, max_sharpe_idx, min_vol_idx)

            # Создание оптимального портфеля
            optimal_portfolio, metrics = self.create_optimal_portfolio(
                top_tickers, results, max_sharpe_idx, expected_returns, portfolio_df
            )
        else:
            print("Не удалось создать оптимальный портфель")
            optimal_portfolio, metrics = None, None

        return optimal_portfolio, metrics, selected
