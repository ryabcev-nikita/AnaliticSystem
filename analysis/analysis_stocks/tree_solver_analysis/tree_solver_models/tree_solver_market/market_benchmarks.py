# ==================== КЛАССЫ ДАННЫХ ====================
from dataclasses import dataclass


@dataclass
class MarketBenchmarks:
    """Рыночные бенчмарки на основе медианных значений"""

    pe_median: float
    pb_median: float
    ps_median: float
    roe_median: float
    div_yield_median: float
    debt_capital_median: float
    beta_median: float
