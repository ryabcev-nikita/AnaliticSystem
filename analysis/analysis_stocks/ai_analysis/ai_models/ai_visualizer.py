# file: visualizer.py
"""
Модуль для визуализации результатов анализа
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ai_models.ai_config import AnalysisConfig
from ai_models.ai_optimizer import PortfolioResult


class ResultVisualizer:
    """Визуализатор результатов анализа"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация визуализатора
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
        plt.style.use('default')
    
    def plot_efficient_frontier(self, results: np.ndarray, 
                               max_sharpe: PortfolioResult,
                               min_volatility: PortfolioResult,
                               save: bool = True) -> None:
        """
        Построение эффективной границы
        
        Args:
            results: результаты всех портфелей (3 x N)
            max_sharpe: портфель с максимальным Sharpe ratio
            min_volatility: портфель с минимальной волатильностью
            save: сохранять ли график в файл
        """
        plt.figure(figsize=(12, 8))
        
        # Все портфели
        plt.scatter(results[1, :], results[0, :], 
                   c=results[2, :], cmap='viridis', 
                   marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        
        # Оптимальные портфели
        plt.scatter(max_sharpe.volatility, max_sharpe.expected_return, 
                   marker='*', color='r', s=500, label='Max Sharpe Ratio')
        plt.scatter(min_volatility.volatility, min_volatility.expected_return, 
                   marker='*', color='b', s=500, label='Min Volatility')
        
        plt.xlabel('Волатильность')
        plt.ylabel('Ожидаемая доходность')
        plt.title('Эффективная граница Марковица')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.config.OUTPUT_EFFICIENT_FRONTIER, 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_nn_history(self, history: Dict[str, List[float]], 
                       save: bool = True) -> None:
        """
        Построение графика истории обучения нейронной сети
        
        Args:
            history: словарь с историей обучения
            save: сохранять ли график в файл
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('История обучения нейронной сети')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.config.OUTPUT_NN_HISTORY, 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weights_comparison(self, tickers: List[str], 
                               max_sharpe_weights: np.ndarray,
                               min_volatility_weights: np.ndarray) -> None:
        """
        Построение сравнения весов портфелей
        
        Args:
            tickers: тикеры акций
            max_sharpe_weights: веса для портфеля с макс. Sharpe
            min_volatility_weights: веса для портфеля с мин. волатильностью
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Сортировка по весам для лучшей визуализации
        sorted_idx = np.argsort(max_sharpe_weights)[::-1]
        sorted_tickers = [tickers[i] for i in sorted_idx]
        
        # Max Sharpe портфель
        ax1.bar(range(len(sorted_tickers)), 
                max_sharpe_weights[sorted_idx])
        ax1.set_xticks(range(len(sorted_tickers)))
        ax1.set_xticklabels(sorted_tickers, rotation=45, ha='right')
        ax1.set_title('Max Sharpe Ratio Portfolio')
        ax1.set_ylabel('Weight')
        ax1.grid(True, alpha=0.3)
        
        # Min Volatility портфель
        ax2.bar(range(len(sorted_tickers)), 
                min_volatility_weights[sorted_idx])
        ax2.set_xticks(range(len(sorted_tickers)))
        ax2.set_xticklabels(sorted_tickers, rotation=45, ha='right')
        ax2.set_title('Min Volatility Portfolio')
        ax2.set_ylabel('Weight')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()