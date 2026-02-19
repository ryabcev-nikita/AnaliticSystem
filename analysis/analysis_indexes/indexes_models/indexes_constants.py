# analysis_indexes_constants.py
import os

# ============= Базовые пути =============
# Определяем корневую директорию проекта
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MARKET_DATA_DIR = os.path.join(DATA_DIR, "market_data")
RESULTS_DIR = os.path.join(DATA_DIR, "world_indexes")

# ============= Параметры анализа =============
START_DATE = "2025-02-12"
END_DATE = "2026-02-13"

# Параметры оптимизации Марковица
RISK_FREE_RATE = 0.05  # 5% годовых
TRADING_DAYS = 252  # Количество торговых дней в году
NUM_PORTFOLIOS = 10000  # Количество портфелей для симуляции
EF_POINTS = 100  # Количество точек на эффективной границе

# Ограничения на веса в портфеле
MIN_WEIGHT = 0.05  # Минимальный вес (0 = можно не включать)
MAX_WEIGHT = 0.3  # Максимальный вес (30% на один актив)
WEIGHT_THRESHOLD = 0.001  # Порог для отображения весов
SIGNIFICANT_WEIGHT_THRESHOLD = 0.01  # Порог для значимых весов

# Параметры для VaR и CVaR
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]

# ============= Файлы данных =============
MARKET_DATA_FILES = {
    "ibovespa": "Brazil_IBOVESPA.csv",
    "dow_jones": "Dow_Jones.csv",
    "hang_seng": "Hang_Seng.csv",
    "nasdaq": "NASDAQ.csv",
    "nifty": "Nifty_50_Индия.csv",
    "nikkei": "Nikkei_225.csv",
    "semiconductor": "Philadelphia_Semiconductor.csv",
    "sp500": "SandP_500.csv",
    "saudi": "Saudi_Arabia_TASI.csv",
    "shanghai": "Shanghai_Composite.csv",
    "south_africa": "South_Africa_Top_40.csv",
}

# ============= Названия индексов для отображения =============
INDEX_NAMES = {
    "ibovespa": "Brazil IBOVESPA",
    "dow_jones": "Dow Jones",
    "hang_seng": "Hang Seng",
    "nasdaq": "NASDAQ",
    "nifty": "Nifty 50 (India)",
    "nikkei": "Nikkei 225",
    "semiconductor": "Philadelphia Semiconductor",
    "sp500": "S&P 500",
    "shanghai": "Shanghai Composite",
    "south_africa": "South Africa Top 40",
    "saudi": "Saudi Arabia TASI",
}

# ============= Цвета для визуализации =============
COLORS = {
    "ibovespa": "#0095B6",
    "dow_jones": "#0072B2",
    "hang_seng": "#E69F00",
    "nasdaq": "#56B4E9",
    "nifty": "#F0E442",
    "nikkei": "#CC79A7",
    "semiconductor": "#D55E00",
    "sp500": "#999999",
    "shanghai": "#009E73",
    "south_africa": "#FF6B6B",
    "saudi": "#6B5B95",
    "efficient_frontier": "#FF5733",
    "max_sharpe": "#2ECC71",
    "min_vol": "#3498DB",
}

# ============= Параметры визуализации =============
PLOT_STYLE = {
    "figure_size": {"large": (14, 12), "medium": (14, 9), "small": (16, 8)},
    "dpi": 300,
    "title_fontsize": 16,
    "label_fontsize": 12,
    "annotation_fontsize": 9,
    "legend_fontsize": 11,
}

# ============= Рыночный индекс для бета-коэффициента =============
BENCHMARK_INDEX = "sp500"

# ============= Имена выходных файлов =============
OUTPUT_FILES = {
    "correlation_matrix": "correlation_matrix",
    "efficient_frontier": "efficient_frontier",
    "weights_comparison": "portfolio_weights_comparison",
    "optimization_results": "portfolio_optimization_results",
    "portfolio_weights": "portfolio_weights",
    "full_report": "portfolio_analysis_full_report",
}
