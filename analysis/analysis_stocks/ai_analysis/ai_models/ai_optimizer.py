# file: portfolio_optimizer.py
"""
Модуль для оптимизации портфеля по методу Марковица
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ai_models.ai_config import AnalysisConfig


@dataclass
class PortfolioResult:
    """Результат оптимизации портфеля"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    num_assets: int
    max_weight: float
    min_weight: float


class MarkowitzOptimizer:
    """Оптимизатор портфеля по методу Марковица"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация оптимизатора
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
        self.results = None
        self.weights_record = None
    
    def build_covariance_matrix(self, betas: np.ndarray) -> np.ndarray:
        """
        Построение ковариационной матрицы на основе beta
        
        Args:
            betas: массив beta коэффициентов
            
        Returns:
            Ковариационная матрица
        """
        n = len(betas)
        market_var = self.config.MARKET_VOLATILITY ** 2
        idio_var = self.config.IDIOSYNCRATIC_VOLATILITY ** 2
        
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = betas[i]**2 * market_var + idio_var
                else:
                    cov_matrix[i, j] = betas[i] * betas[j] * market_var
        
        return cov_matrix
    
    def _apply_portfolio_constraints(self, weights: np.ndarray, 
                                     min_weight: float = 0.0,
                                     max_weight: float = 1.0,
                                     max_assets: Optional[int] = None,
                                     min_assets: int = 1) -> np.ndarray:
        """
        Применение ограничений к портфелю
        
        Args:
            weights: исходные веса
            min_weight: минимальная доля акции
            max_weight: максимальная доля акции
            max_assets: максимальное количество акций в портфеле
            min_assets: минимальное количество акций в портфеле
            
        Returns:
            Веса с примененными ограничениями
        """
        n = len(weights)
        
        # Если нет ограничений, возвращаем исходные веса
        if min_weight == 0 and max_weight == 1 and max_assets is None and min_assets <= 1:
            return weights
        
        # Копируем веса
        constrained_weights = weights.copy()
        
        # Применяем ограничение на максимальное количество акций
        if max_assets is not None and max_assets < n:
            # Оставляем только топ-max_assets акций
            top_indices = np.argsort(constrained_weights)[-max_assets:]
            mask = np.zeros(n, dtype=bool)
            mask[top_indices] = True
            constrained_weights[~mask] = 0
        
        # Применяем ограничение на минимальное количество акций
        if min_assets > 1:
            # Убеждаемся, что как минимум min_assets акций имеют ненулевой вес
            nonzero_count = np.sum(constrained_weights > 1e-6)
            if nonzero_count < min_assets:
                # Добавляем акции с наименьшими весами, пока не достигнем min_assets
                zero_indices = np.where(constrained_weights <= 1e-6)[0]
                
                # Сортируем нулевые акции по их исходным весам
                sorted_zero = sorted(zero_indices, key=lambda i: weights[i], reverse=True)
                
                # Добавляем акции, пока не достигнем min_assets
                assets_to_add = min_assets - nonzero_count
                for i in range(min(assets_to_add, len(sorted_zero))):
                    idx = sorted_zero[i]
                    constrained_weights[idx] = weights[idx]
        
        # Применяем ограничения на минимальную и максимальную долю
        if min_weight > 0 or max_weight < 1:
            # Обрезаем веса
            constrained_weights = np.clip(constrained_weights, min_weight, max_weight)
        
        # Перенормировка
        total = np.sum(constrained_weights)
        if total > 0:
            constrained_weights = constrained_weights / total
        else:
            # Если все веса стали нулевыми, равномерно распределяем между min_assets акциями
            constrained_weights = np.zeros(n)
            top_indices = np.argsort(weights)[-min_assets:]
            for idx in top_indices:
                constrained_weights[idx] = 1.0 / min_assets
        
        return constrained_weights
    
    def portfolio_stats(self, weights: np.ndarray, returns: np.ndarray, 
                       cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Расчет статистик портфеля
        
        Args:
            weights: веса активов
            returns: ожидаемые доходности
            cov_matrix: ковариационная матрица
            
        Returns:
            Кортеж (доходность, волатильность, коэффициент Шарпа)
        """
        port_return = np.sum(returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - self.config.RISK_FREE_RATE) / port_volatility
        
        return port_return, port_volatility, sharpe_ratio
    
    def optimize(self, returns: np.ndarray, cov_matrix: np.ndarray,
                min_weight: float = 0.0,
                max_weight: float = 1.0,
                max_assets: Optional[int] = None,
                min_assets: int = 1) -> Dict[str, PortfolioResult]:
        """
        Оптимизация портфеля с ограничениями
        
        Args:
            returns: ожидаемые доходности активов
            cov_matrix: ковариационная матрица
            min_weight: минимальная доля акции в портфеле
            max_weight: максимальная доля акции в портфеле
            max_assets: максимальное количество акций в портфеле
            min_assets: минимальное количество акций в портфеле
            
        Returns:
            Словарь с результатами оптимизации
        """
        n = len(returns)
        
        # Валидация параметров
        if min_weight < 0 or min_weight > 1:
            raise ValueError("min_weight должен быть между 0 и 1")
        if max_weight < 0 or max_weight > 1:
            raise ValueError("max_weight должен быть между 0 и 1")
        if min_weight > max_weight:
            raise ValueError("min_weight не может быть больше max_weight")
        if max_assets is not None and (max_assets < 1 or max_assets > n):
            raise ValueError(f"max_assets должен быть между 1 и {n}")
        if min_assets < 1 or min_assets > n:
            raise ValueError(f"min_assets должен быть между 1 и {n}")
        if min_assets * min_weight > 1:
            raise ValueError(f"Невозможно выполнить ограничения: минимальная сумма весов = {min_assets * min_weight} > 1")
        
        # Корректируем max_weight если нужно
        effective_max_weight = min(max_weight, 1.0 - (min_assets - 1) * min_weight)
        
        print(f"\nОптимизация с ограничениями:")
        print(f"  Мин. доля акции: {min_weight:.2%}")
        print(f"  Макс. доля акции: {effective_max_weight:.2%}")
        print(f"  Мин. кол-во акций: {min_assets}")
        print(f"  Макс. кол-во акций: {max_assets if max_assets else 'нет'}")
        
        # Генерация случайных портфелей
        num_portfolios = self.config.NUM_PORTFOLIOS
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Генерируем случайные веса
            weights = np.random.random(n)
            weights = weights / np.sum(weights)  # Сначала нормализуем
            
            # Применяем ограничения
            weights = self._apply_portfolio_constraints(
                weights, 
                min_weight=min_weight,
                max_weight=effective_max_weight,
                max_assets=max_assets,
                min_assets=min_assets
            )
            
            weights_record.append(weights)
            
            port_return, port_volatility, sharpe_ratio = self.portfolio_stats(
                weights, returns, cov_matrix
            )
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = sharpe_ratio
        
        self.results = results
        self.weights_record = weights_record
        
        # Поиск портфеля с максимальным Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_weights = weights_record[max_sharpe_idx]
        max_sharpe_return, max_sharpe_vol, max_sharpe = self.portfolio_stats(
            max_sharpe_weights, returns, cov_matrix
        )
        
        # Поиск портфеля с минимальной волатильностью
        min_vol_idx = np.argmin(results[1])
        min_vol_weights = weights_record[min_vol_idx]
        min_vol_return, min_vol_vol, min_vol_sharpe = self.portfolio_stats(
            min_vol_weights, returns, cov_matrix
        )
        
        # Подсчет количества активов в портфелях
        max_sharpe_assets = np.sum(max_sharpe_weights > 1e-6)
        min_vol_assets = np.sum(min_vol_weights > 1e-6)
        
        return {
            'max_sharpe': PortfolioResult(
                weights=max_sharpe_weights,
                expected_return=max_sharpe_return,
                volatility=max_sharpe_vol,
                sharpe_ratio=max_sharpe,
                num_assets=max_sharpe_assets,
                max_weight=np.max(max_sharpe_weights),
                min_weight=np.min(max_sharpe_weights[max_sharpe_weights > 1e-6]) if max_sharpe_assets > 0 else 0
            ),
            'min_volatility': PortfolioResult(
                weights=min_vol_weights,
                expected_return=min_vol_return,
                volatility=min_vol_vol,
                sharpe_ratio=min_vol_sharpe,
                num_assets=min_vol_assets,
                max_weight=np.max(min_vol_weights),
                min_weight=np.min(min_vol_weights[min_vol_weights > 1e-6]) if min_vol_assets > 0 else 0
            )
        }
    
    def optimize_with_target_return(self, returns: np.ndarray, cov_matrix: np.ndarray,
                                   target_return: float,
                                   min_weight: float = 0.0,
                                   max_weight: float = 1.0,
                                   max_assets: Optional[int] = None,
                                   min_assets: int = 1) -> Optional[PortfolioResult]:
        """
        Оптимизация портфеля с целевой доходностью
        
        Args:
            returns: ожидаемые доходности активов
            cov_matrix: ковариационная матрица
            target_return: целевая доходность
            min_weight: минимальная доля акции
            max_weight: максимальная доля акции
            max_assets: максимальное количество акций
            min_assets: минимальное количество акций
            
        Returns:
            Портфель с минимальной волатильностью для целевой доходности
        """
        n = len(returns)
        
        # Проверяем, достижима ли целевая доходность
        max_possible_return = np.max(returns)
        min_possible_return = np.min(returns)
        
        if target_return > max_possible_return or target_return < min_possible_return:
            print(f"Предупреждение: целевая доходность {target_return:.2%} вне диапазона "
                  f"[{min_possible_return:.2%}, {max_possible_return:.2%}]")
            return None
        
        # Генерируем портфели и ищем с доходностью близкой к целевой
        num_portfolios = self.config.NUM_PORTFOLIOS * 2  # Увеличиваем для точности
        best_portfolio = None
        min_volatility = float('inf')
        
        for i in range(num_portfolios):
            weights = np.random.random(n)
            weights = weights / np.sum(weights)
            weights = self._apply_portfolio_constraints(
                weights, min_weight, max_weight, max_assets, min_assets
            )
            
            port_return, port_volatility, _ = self.portfolio_stats(weights, returns, cov_matrix)
            
            # Проверяем близость к целевой доходности
            if abs(port_return - target_return) < 0.001:  # допуск 0.1%
                if port_volatility < min_volatility:
                    min_volatility = port_volatility
                    best_portfolio = weights
        
        if best_portfolio is not None:
            port_return, port_volatility, sharpe = self.portfolio_stats(
                best_portfolio, returns, cov_matrix
            )
            num_assets = np.sum(best_portfolio > 1e-6)
            
            return PortfolioResult(
                weights=best_portfolio,
                expected_return=port_return,
                volatility=port_volatility,
                sharpe_ratio=sharpe,
                num_assets=num_assets,
                max_weight=np.max(best_portfolio),
                min_weight=np.min(best_portfolio[best_portfolio > 1e-6]) if num_assets > 0 else 0
            )
        
        return None
    
    def get_all_portfolios(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Получение всех сгенерированных портфелей"""
        return self.results, self.weights_record
    
    def get_portfolio_statistics(self, portfolio_result: PortfolioResult) -> Dict:
        """
        Получение расширенной статистики портфеля
        
        Args:
            portfolio_result: результат оптимизации портфеля
            
        Returns:
            Словарь со статистикой
        """
        weights = portfolio_result.weights
        nonzero_weights = weights[weights > 1e-6]
        
        return {
            'num_assets': portfolio_result.num_assets,
            'max_weight': portfolio_result.max_weight,
            'min_weight': portfolio_result.min_weight,
            'avg_weight': np.mean(nonzero_weights) if len(nonzero_weights) > 0 else 0,
            'std_weights': np.std(nonzero_weights) if len(nonzero_weights) > 0 else 0,
            'concentration': np.sum(weights**2),  # Индекс Херфиндаля
            'top_3_concentration': np.sum(np.sort(weights)[-3:])  # Доля топ-3 позиций
        }