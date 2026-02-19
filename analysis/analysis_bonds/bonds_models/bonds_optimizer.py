from typing import Dict, List, Tuple

import numpy as np

from bonds_models.bonds_models import Bond
from bonds_models.bonds_constants import OPTIMIZATION_PARAMS
from scipy.optimize import minimize, differential_evolution


class BondsPortfolioOptimizer:
    """Класс для оптимизации портфеля облигаций"""

    def __init__(self, bonds_list: List[Bond]):
        self.bonds_list = bonds_list
        self.n_bonds = len(bonds_list)
        self.selected_indices = []
        self.optimal_weights = None

    def portfolio_statistics(self, weights: np.ndarray) -> Dict:
        """Расчет статистик портфеля"""
        portfolio_yield = 0
        portfolio_duration = 0
        portfolio_risk = 0
        portfolio_convexity = 0

        for i, weight in enumerate(weights):
            bond = self.bonds_list[i]
            portfolio_yield += weight * bond.current_yield
            portfolio_duration += weight * bond.modified_duration
            portfolio_risk += weight * bond.risk_level
            portfolio_convexity += weight * bond.convexity

        # Оценка диверсификации
        hhi = np.sum(weights**2)  # Индекс Херфиндаля-Хиршмана
        diversification_score = 1 - (hhi - 1 / self.n_bonds) / (1 - 1 / self.n_bonds)

        return {
            "yield": portfolio_yield,
            "duration": portfolio_duration,
            "risk_score": portfolio_risk,
            "convexity": portfolio_convexity,
            "hhi": hhi,
            "diversification": diversification_score,
            "n_bonds": np.sum(weights > 0.001),
        }

    def objective_function(self, weights: np.ndarray) -> float:
        """Целевая функция для оптимизации"""
        stats = self.portfolio_statistics(weights)

        # Максимизация доходности
        yield_score = stats["yield"] * 100

        # Минимизация риска
        risk_penalty = stats["risk_score"] * 0.01

        # Штраф за отклонение от целевой дюрации
        duration_penalty = (
            abs(stats["duration"] - OPTIMIZATION_PARAMS["target_duration"]) * 0.005
        )

        # Штраф за недостаточную диверсификацию
        diversification_penalty = (1 - stats["diversification"]) * 0.02

        # Штраф за слишком большое количество облигаций
        n_bonds_penalty = (
            max(0, stats["n_bonds"] - OPTIMIZATION_PARAMS["max_bonds"]) * 0.005
        )
        n_bonds_penalty += (
            max(0, OPTIMIZATION_PARAMS["min_bonds"] - stats["n_bonds"]) * 0.01
        )

        return -(
            yield_score
            - risk_penalty
            - duration_penalty
            - diversification_penalty
            - n_bonds_penalty
        )

    def check_constraints(self, weights: np.ndarray) -> bool:
        """Проверка ограничений"""
        stats = self.portfolio_statistics(weights)

        # Проверка минимальной доходности
        if stats["yield"] < OPTIMIZATION_PARAMS["min_current_yield"]:
            return False

        # Проверка максимального веса на одну облигацию
        if np.max(weights) > OPTIMIZATION_PARAMS["max_weight_per_bond"]:
            return False

        # Проверка на одного эмитента (упрощенно - по первому слову названия)
        issuer_weights = {}
        for i, weight in enumerate(weights):
            if weight > 0:
                issuer = self.bonds_list[i].name.split()[0]
                issuer_weights[issuer] = issuer_weights.get(issuer, 0) + weight

        if (
            max(issuer_weights.values())
            > OPTIMIZATION_PARAMS["max_weight_single_issuer"]
        ):
            return False

        return True

    def optimize_portfolio(
        self, method: str = "differential_evolution"
    ) -> Tuple[np.ndarray, Dict]:
        """Оптимизация портфеля"""
        n = len(self.bonds_list)

        if method == "differential_evolution":
            # Эволюционный алгоритм для глобальной оптимизации
            bounds = [(0, OPTIMIZATION_PARAMS["max_weight_per_bond"]) for _ in range(n)]

            def objective_with_constraints(x):
                if not self.check_constraints(x):
                    return 1e10
                return self.objective_function(x)

            result = differential_evolution(
                objective_with_constraints,
                bounds,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                seed=42,
            )

            weights = result.x
        else:
            # SLSQP для локальной оптимизации
            init_weights = np.array([1 / n] * n)
            bounds = [(0, OPTIMIZATION_PARAMS["max_weight_per_bond"]) for _ in range(n)]
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            result = minimize(
                self.objective_function,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )
            weights = result.x

        # Нормализация весов
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

        # Оставляем только значимые веса
        weights[weights < 0.005] = 0
        weights = weights / np.sum(weights)

        self.optimal_weights = weights
        self.selected_indices = [i for i, w in enumerate(weights) if w > 0]

        return weights, self.portfolio_statistics(weights)

    def get_portfolio_by_risk_profile(self, risk_level: int) -> Dict:
        """Получение портфеля для заданного уровня риска"""
        # Фильтруем облигации по уровню риска
        risk_bonds = [b for b in self.bonds_list if b.risk_level <= risk_level]

        if not risk_bonds:
            return {}

        # Сортируем по скорингу
        risk_bonds.sort(key=lambda x: x.total_score, reverse=True)

        # Берем топ-30 облигаций
        selected_bonds = risk_bonds[:30]

        # Создаем временный оптимизатор
        temp_optimizer = BondsPortfolioOptimizer(selected_bonds)
        weights, stats = temp_optimizer.optimize_portfolio()

        portfolio = {"bonds": selected_bonds, "weights": weights, "statistics": stats}

        return portfolio
