# file: report_generator.py
"""
Модуль для генерации отчетов
"""
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

from ai_models.ai_config import AnalysisConfig
from ai_models.ai_optimizer import PortfolioResult


class ReportGenerator:
    """Генератор отчетов"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация генератора отчетов
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
    
    def save_selected_stocks(self, df: pd.DataFrame) -> None:
        """
        Сохранение отобранных акций в Excel
        
        Args:
            df: DataFrame с отобранными акциями
        """
        output_cols = [col for col in self.config.OUTPUT_COLUMNS if col in df.columns]
        df[output_cols].to_excel(self.config.OUTPUT_SELECTED_STOCKS, index=False)
        print(f"Сохранен файл: {self.config.OUTPUT_SELECTED_STOCKS}")
    
    def save_portfolio_weights(self, tickers: List[str], 
                              max_sharpe: PortfolioResult,
                              min_volatility: PortfolioResult) -> None:
        """
        Сохранение весов портфелей в Excel
        
        Args:
            tickers: тикеры акций
            max_sharpe: портфель с максимальным Sharpe ratio
            min_volatility: портфель с минимальной волатильностью
        """
        weights_df = pd.DataFrame({
            'ticker': tickers,
            'max_sharpe_weights': max_sharpe.weights,
            'min_volatility_weights': min_volatility.weights
        })
        weights_df.to_excel(self.config.OUTPUT_PORTFOLIO_WEIGHTS, index=False)
        print(f"Сохранен файл: {self.config.OUTPUT_PORTFOLIO_WEIGHTS}")
    
    def generate_text_report(self, 
                            undervalued_df: pd.DataFrame,
                            train_loss: float,
                            val_loss: float,
                            max_sharpe: PortfolioResult,
                            min_volatility: PortfolioResult) -> None:
        """
        Генерация текстового отчета
        
        Args:
            undervalued_df: DataFrame с недооцененными акциями
            train_loss: финальная ошибка на обучении
            val_loss: финальная ошибка на валидации
            max_sharpe: портфель с максимальным Sharpe ratio
            min_volatility: портфель с минимальной волатильностью
        """
        with open(self.config.OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("АНАЛИЗ НЕДООЦЕНЕННЫХ АКЦИЙ И ОПТИМИЗАЦИЯ ПОРТФЕЛЯ\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Результаты нейронной сети
            f.write("1. РЕЗУЛЬТАТЫ НЕЙРОННОЙ СЕТИ\n")
            f.write("-"*40 + "\n")
            f.write(f"Финальная ошибка на обучении (MSE): {train_loss:.4f}\n")
            f.write(f"Финальная ошибка на валидации (MSE): {val_loss:.4f}\n\n")
            
            # Топ-10 недооцененных акций
            f.write("2. ТОП-10 НЕДООЦЕНЕННЫХ АКЦИЙ\n")
            f.write("-"*40 + "\n")
            top10 = undervalued_df.nsmallest(10, 'residual')
            for idx, row in top10.iterrows():
                f.write(f"{row['ticker']}: {row['name']}\n")
                f.write(f"  P/E: {row['pe']:.2f}, ROE: {row['roe']*100:.1f}%, P/B: {row['pb']:.2f}\n")
                f.write(f"  Div Yield: {row['dividend_yield']:.2f}%, g: {row['g']*100:.2f}%\n")
                f.write(f"  Residual: {row['residual']:.2f}\n\n")
            
            # Оптимальные портфели
            f.write("3. ОПТИМАЛЬНЫЕ ПОРТФЕЛИ\n")
            f.write("-"*40 + "\n")
            
            f.write("\nПортфель с максимальным Sharpe ratio:\n")
            f.write(f"  Ожидаемая доходность: {max_sharpe.expected_return*100:.2f}%\n")
            f.write(f"  Волатильность: {max_sharpe.volatility*100:.2f}%\n")
            f.write(f"  Sharpe ratio: {max_sharpe.sharpe_ratio:.2f}\n")
            f.write("\n  Топ-5 позиций:\n")
            top_weights_idx = np.argsort(max_sharpe.weights)[-5:][::-1]
            for idx in top_weights_idx:
                f.write(f"    {undervalued_df.iloc[idx]['ticker']}: {max_sharpe.weights[idx]*100:.2f}%\n")
            
            f.write("\nПортфель с минимальной волатильностью:\n")
            f.write(f"  Ожидаемая доходность: {min_volatility.expected_return*100:.2f}%\n")
            f.write(f"  Волатильность: {min_volatility.volatility*100:.2f}%\n")
            f.write(f"  Sharpe ratio: {min_volatility.sharpe_ratio:.2f}\n")
            f.write("\n  Топ-5 позиций:\n")
            top_weights_idx = np.argsort(min_volatility.weights)[-5:][::-1]
            for idx in top_weights_idx:
                f.write(f"    {undervalued_df.iloc[idx]['ticker']}: {min_volatility.weights[idx]*100:.2f}%\n")
        
        print(f"Сохранен файл: {self.config.OUTPUT_SUMMARY}")