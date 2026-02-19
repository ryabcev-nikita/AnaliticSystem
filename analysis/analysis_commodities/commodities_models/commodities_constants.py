# metals_constants.py
import os
from pathlib import Path

# ============= Базовые пути =============
# Определяем корневую директорию проекта
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MARKET_DATA_DIR = os.path.join(DATA_DIR, "market_data")
RESULTS_DIR = os.path.join(DATA_DIR, "metals_analysis")

# ============= Параметры анализа =============
START_DATE = "2025-02-13"
END_DATE = "2026-02-13"

# Параметры оптимизации Марковица
RISK_FREE_RATE = 0.05  # 5% годовых
TRADING_DAYS = 252  # Количество торговых дней в году
NUM_PORTFOLIOS = 10000  # Количество портфелей для симуляции
EF_POINTS = 100  # Количество точек на эффективной границе

# Ограничения на веса в портфеле
MIN_WEIGHT = 0.0  # Минимальный вес (0 = можно не включать)
MAX_WEIGHT = (
    0.5  # Максимальный вес (50% на один металл - для сырьевых товаров можно больше)
)
WEIGHT_THRESHOLD = 0.001  # Порог для отображения весов
SIGNIFICANT_WEIGHT_THRESHOLD = 0.01  # Порог для значимых весов

# Параметры для VaR и CVaR
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]

# ============= Файлы данных по металлам =============
METALS_DATA_FILES = {
    "gold": "Gold.csv",
    "silver": "Silver.csv",
    "platinum": "Platinum.csv",
    "palladium": "Palladium.csv",
    "copper": "Copper.csv",
    "aluminum": "Aluminum.csv",
    "oil": "Oil_Brent.csv",  # Добавляем нефть как сырьевой товар для сравнения
}

# ============= Названия металлов для отображения =============
METAL_NAMES = {
    "gold": "Gold",
    "silver": "Silver",
    "platinum": "Platinum",
    "palladium": "Palladium",
    "copper": "Copper",
    "aluminum": "Aluminum",
    "oil": "Brent Oil",
}

# ============= Цвета для визуализации =============
COLORS = {
    "gold": "#FFD700",  # Золотой
    "silver": "#C0C0C0",  # Серебряный
    "platinum": "#E5E4E2",  # Платиновый
    "palladium": "#8A9597",  # Палладий
    "copper": "#B87333",  # Медный
    "aluminum": "#848789",  # Алюминиевый
    "oil": "#3A5F40",  # Темно-зеленый для нефти
    "efficient_frontier": "#FF5733",
    "max_sharpe": "#2ECC71",
    "min_vol": "#3498DB",
}

# ============= Группы металлов для анализа =============
PRECIOUS_METALS = ["gold", "silver", "platinum", "palladium"]
INDUSTRIAL_METALS = ["copper", "aluminum"]
ENERGY = ["oil"]

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
# Для металлов используем золото как "безрисковый" актив или нефть как бенчмарк
BENCHMARK_INDEX = "gold"  # Можно изменить на "oil" или другой

# ============= Имена выходных файлов =============
OUTPUT_FILES = {
    "correlation_matrix": "metals_correlation_matrix",
    "efficient_frontier": "metals_efficient_frontier",
    "weights_comparison": "metals_portfolio_weights_comparison",
    "optimization_results": "metals_portfolio_optimization_results",
    "portfolio_weights": "metals_portfolio_weights",
    "full_report": "metals_analysis_full_report",
    "prices_chart": "metals_prices_normalized",
    "returns_distribution": "metals_returns_distribution",
}
