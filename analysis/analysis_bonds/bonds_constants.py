# bonds_constants.py
import os
from datetime import datetime

# ============= Базовые пути =============
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "bonds_analysis")

# ============= Дата анализа =============
ANALYSIS_DATE = datetime.now().strftime("%Y-%m-%d")

# ============= Параметры оптимизации =============
RISK_FREE_RATE = 0.155  # 16% - ключевая ставка ЦБ РФ
INFLATION_RATE = 0.055  # 7.5% - прогноз инфляции
TAX_RATE = 0.13  # 13% - НДФЛ на купоны

# Параметры для разных валют
CURRENCY_PARAMS = {
    "rub": {
        "risk_free_rate": 0.155,
        "min_yield": 0.12,
        "max_yield": 0.35,
        "min_duration": 0.1,
        "max_duration": 10,
        "liquidity_premium": 0.01,
    },
    "cny": {
        "risk_free_rate": 0.03,
        "min_yield": 0.04,
        "max_yield": 0.12,
        "min_duration": 0.1,
        "max_duration": 7,
        "liquidity_premium": 0.02,
    },
    "usd": {
        "risk_free_rate": 0.045,
        "min_yield": 0.05,
        "max_yield": 0.10,
        "min_duration": 0.1,
        "max_duration": 10,
        "liquidity_premium": 0.01,
    },
    "eur": {
        "risk_free_rate": 0.03,
        "min_yield": 0.035,
        "max_yield": 0.08,
        "min_duration": 0.1,
        "max_duration": 8,
        "liquidity_premium": 0.015,
    },
}

# ============= Риск-уровни =============
RISK_LEVELS = {
    0: {
        "name": "Минимальный риск",
        "description": "ОФЗ, муниципальные, квазисуверенные",
        "max_weight_per_issuer": 0.20,
        "min_credit_quality": "AAA",
        "max_yield_spread": 0.02,
        "max_duration": 5,
        "min_liquidity": 0.8,
    },
    1: {
        "name": "Низкий риск",
        "description": "Крупные банки, системно значимые компании",
        "max_weight_per_issuer": 0.15,
        "min_credit_quality": "AA",
        "max_yield_spread": 0.04,
        "max_duration": 7,
        "min_liquidity": 0.7,
    },
    2: {
        "name": "Средний риск",
        "description": "Крупные корпоративные эмитенты",
        "max_weight_per_issuer": 0.10,
        "min_credit_quality": "A",
        "max_yield_spread": 0.07,
        "max_duration": 10,
        "min_liquidity": 0.6,
    },
    3: {
        "name": "Повышенный риск",
        "description": "Малые и средние компании, ВДО",
        "max_weight_per_issuer": 0.05,
        "min_credit_quality": "BBB",
        "max_yield_spread": 0.15,
        "max_duration": 15,
        "min_liquidity": 0.4,
    },
}

# ============= Сектора =============
SECTORS = {
    "government": {"name": "Государственные", "max_weight": 0.40, "min_weight": 0.10},
    "municipal": {"name": "Муниципальные", "max_weight": 0.25, "min_weight": 0.05},
    "financial": {"name": "Финансовый сектор", "max_weight": 0.30, "min_weight": 0.05},
    "energy": {"name": "Энергетика", "max_weight": 0.25, "min_weight": 0.0},
    "materials": {"name": "Сырьевой сектор", "max_weight": 0.25, "min_weight": 0.0},
    "industrials": {"name": "Промышленность", "max_weight": 0.20, "min_weight": 0.0},
    "consumer": {
        "name": "Потребительский сектор",
        "max_weight": 0.20,
        "min_weight": 0.0,
    },
    "telecom": {"name": "Телекоммуникации", "max_weight": 0.15, "min_weight": 0.0},
    "it": {"name": "IT", "max_weight": 0.10, "min_weight": 0.0},
    "utilities": {"name": "Коммунальные услуги", "max_weight": 0.15, "min_weight": 0.0},
    "real_estate": {"name": "Недвижимость", "max_weight": 0.15, "min_weight": 0.0},
    "health_care": {"name": "Здравоохранение", "max_weight": 0.10, "min_weight": 0.0},
    "other": {"name": "Прочее", "max_weight": 0.10, "min_weight": 0.0},
}

# ============= Параметры оптимизации портфеля =============
OPTIMIZATION_PARAMS = {
    "min_bonds": 15,  # Минимальное количество облигаций
    "max_bonds": 50,  # Максимальное количество облигаций
    "min_weight_per_bond": 0.01,  # 1% - минимальный вес
    "max_weight_per_bond": 0.15,  # 15% - максимальный вес
    "max_weight_single_issuer": 0.20,  # 20% - на одного эмитента
    "target_duration": 3.5,  # Целевая дюрация
    "min_current_yield": 0.12,  # Минимальная текущая доходность
    "max_nominal_deviation": 0.30,  # Максимальное отклонение номинала
}

# ============= Параметры скоринга =============
SCORING_WEIGHTS = {
    "yield_score": 0.30,  # Доходность
    "risk_score": 0.25,  # Риск
    "liquidity_score": 0.15,  # Ликвидность
    "duration_score": 0.10,  # Дюрация
    "sector_score": 0.10,  # Сектор
    "currency_score": 0.10,  # Валюта
}

# ============= Цвета для визуализации =============
COLORS = {
    "government": "#1f77b4",
    "municipal": "#ff7f0e",
    "financial": "#2ca02c",
    "energy": "#d62728",
    "materials": "#9467bd",
    "industrials": "#8c564b",
    "consumer": "#e377c2",
    "telecom": "#7f7f7f",
    "it": "#bcbd22",
    "utilities": "#17becf",
    "real_estate": "#aec7e8",
    "health_care": "#ffbb78",
    "other": "#98df8a",
    "rub": "#4daf4a",
    "cny": "#ff7f00",
    "usd": "#377eb8",
    "eur": "#984ea3",
    "risk_0": "#2ecc71",
    "risk_1": "#3498db",
    "risk_2": "#f39c12",
    "risk_3": "#e74c3c",
}

# ============= Параметры визуализации =============
PLOT_STYLE = {
    "figure_size": {"large": (16, 10), "medium": (14, 8), "small": (12, 6)},
    "dpi": 300,
    "title_fontsize": 16,
    "label_fontsize": 12,
    "annotation_fontsize": 10,
    "legend_fontsize": 11,
}

# ============= Имена выходных файлов =============
OUTPUT_FILES = {
    "full_report": "bonds_portfolio_analysis_full_report",
    "portfolio_structure": "bonds_portfolio_structure",
    "risk_analysis": "bonds_risk_analysis",
    "yield_curve": "bonds_yield_curve",
    "sector_allocation": "bonds_sector_allocation",
    "currency_allocation": "bonds_currency_allocation",
    "maturity_profile": "bonds_maturity_profile",
}
