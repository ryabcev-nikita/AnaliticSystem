# file: config.py
"""
Конфигурационный файл с настройками анализа
"""
from dataclasses import dataclass
import os
from typing import Optional, Tuple, List

def find_root_dir(marker="main.py"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, marker)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Достигли корня файловой системы
            return None
        current_dir = parent_dir

parent_dir = find_root_dir()
ai_analysis_dir = f"{parent_dir}/data/ai_analysis/"
os.makedirs(ai_analysis_dir, exist_ok=True)
        
@dataclass
class AnalysisConfig:
    """Конфигурация анализа"""
    
    
    # Пути к файлам
    INPUT_FILE: str = f"{parent_dir}/data/" + 'fundamentals_shares.xlsx'
    OUTPUT_SELECTED_STOCKS: str = ai_analysis_dir + 'selected_stocks.xlsx'
    OUTPUT_PORTFOLIO_WEIGHTS: str = ai_analysis_dir + 'portfolio_weights.xlsx'
    OUTPUT_EFFICIENT_FRONTIER: str = ai_analysis_dir + 'efficient_frontier.png'
    OUTPUT_NN_HISTORY: str = ai_analysis_dir + 'nn_training_history.png'
    OUTPUT_SUMMARY: str = ai_analysis_dir + 'summary.txt'
    
    # Фильтры для отбора акций
    MAX_PE: float = 100.0
    MAX_EV_EBITDA: float = 100.0
    MIN_EV_EBITDA: float = 0.1
    MIN_PE: float = 0.1
    MAX_PB: float = 20.0
    MIN_PB: float = 0.1
    
    # Параметры нейронной сети
    NN_HIDDEN_LAYERS: Tuple[int, ...] = (10,)
    NN_ACTIVATION: str = 'relu'
    NN_SOLVER: str = 'adam'
    NN_MAX_ITER: int = 500
    NN_EPOCHS: int = 50
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Параметры оптимизации портфеля
    RISK_FREE_RATE: float = 0.155
    MARKET_VOLATILITY: float = 0.20  # 20%
    IDIOSYNCRATIC_VOLATILITY: float = 0.10  # 10%
    NUM_PORTFOLIOS: int = 10000
    
    # Параметры ограничений портфеля (можно задать значения по умолчанию)
    PORTFOLIO_MIN_WEIGHT: float = 0.01  # минимальная доля 1%
    PORTFOLIO_MAX_WEIGHT: float = 0.15  # максимальная доля 30%
    PORTFOLIO_MIN_ASSETS: int = 10        # минимум 5 акций
    PORTFOLIO_MAX_ASSETS: Optional[int] = 20  # максимум 15 акций
    
    # Колонки для анализа
    NUMERIC_COLUMNS: List[str] = None
    FEATURE_COLUMNS: List[str] = None
    OUTPUT_COLUMNS: List[str] = None
    
    def __post_init__(self):
        self.NUMERIC_COLUMNS = [
            'market_cap', 'ev', 'revenue', 'net_income', 'ebitda', 'pe', 'pb', 'ps',
            'pfcf', 'roe', 'roa', 'roic', 'ev_ebitda', 'ev_s', 'fcf', 'cagr_sales',
            'avg_dividend_yield', 'avg_cagr_dividend', 'current_ratio', 'payout_ratio',
            'npm', 'debt', 'debt_capital', 'net_debt_ebitda', 'debt_ebitda', 'eps',
            'dividend_yield', 'beta', 'dividend_per_share'
        ]
        
        self.FEATURE_COLUMNS = ['roe', 'pb', 'avg_dividend_yield', 'g', 'beta']
        
        self.OUTPUT_COLUMNS = [
            'ticker', 'name', 'pe', 'roe', 'pb', 'dividend_yield', 
            'g', 'residual', 'expected_return', 'beta'
        ]