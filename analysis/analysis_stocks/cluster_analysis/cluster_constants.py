"""
Константы для модуля кластерного анализа и оптимизации портфеля.
Все магические числа вынесены в именованные константы.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List


# ==================== КОНСТАНТЫ КЛАСТЕРИЗАЦИИ ====================


@dataclass(frozen=True)
class ClusterConstants:
    """Константы для кластерного анализа"""

    # Параметры кластеризации
    DEFAULT_N_CLUSTERS: int = 4
    MAX_CLUSTERS: int = 8
    MIN_CLUSTERS: int = 2
    RANDOM_STATE: int = 42
    N_INIT: int = 10

    # PCA компоненты
    PCA_COMPONENTS: int = 2

    # Признаки по умолчанию
    DEFAULT_CLUSTER_FEATURES: Tuple[str, ...] = ("PB", "ROE")

    # Минимальное количество данных для кластеризации
    MIN_DATA_FOR_CLUSTERING: int = 3

    # Размеры точек на графиках
    SCATTER_POINT_SIZE: int = 100
    CENTROID_POINT_SIZE: int = 300
    BAR_TEXT_OFFSET: float = 0.5

    # Альфа-прозрачность
    SCATTER_ALPHA: float = 0.7
    GRID_ALPHA: float = 0.3


@dataclass(frozen=True)
class ClusterThresholds:
    """Пороговые значения для кластеров"""

    # P/E пороги
    PE_DEEP_VALUE: float = 5.0
    PE_VALUE: float = 8.0
    PE_FAIR: float = 12.0
    PE_GROWTH: float = 15.0

    # P/B пороги
    PB_DEEP_VALUE: float = 0.8
    PB_VALUE: float = 1.2
    PB_FAIR: float = 1.5
    PB_GROWTH: float = 2.0

    # ROE пороги
    ROE_HIGH: float = 20.0
    ROE_GOOD: float = 12.0

    # Дивидендные пороги
    DIV_HIGH: float = 8.0
    DIV_GOOD: float = 5.0

    # Пороги для скоринга
    SCORE_PE_STRONG: float = 5.0
    SCORE_PE_MEDIUM: float = 8.0
    SCORE_PE_WEAK: float = 12.0
    SCORE_PE_OVER: float = 25.0

    SCORE_PB_STRONG: float = 0.7
    SCORE_PB_MEDIUM: float = 1.0
    SCORE_PB_WEAK: float = 1.5
    SCORE_PB_OVER: float = 3.0

    SCORE_ROE_HIGH: float = 25.0
    SCORE_ROE_GOOD: float = 20.0
    SCORE_ROE_MEDIUM: float = 15.0
    SCORE_ROE_LOW: float = 10.0

    SCORE_DEBT_LOW: float = 20.0
    SCORE_DEBT_MEDIUM: float = 40.0
    SCORE_DEBT_HIGH: float = 60.0
    SCORE_DEBT_CRITICAL: float = 80.0

    SCORE_DIV_HIGH: float = 12.0
    SCORE_DIV_GOOD: float = 10.0
    SCORE_DIV_MEDIUM: float = 8.0
    SCORE_DIV_LOW: float = 6.0
    SCORE_DIV_POOR: float = 4.0
    SCORE_DIV_MIN: float = 1.0


@dataclass(frozen=True)
class ClusterScores:
    """Баллы для скоринга кластеров"""

    # Баллы для P/E
    PE_DEEP_VALUE_SCORE: int = 2
    PE_VALUE_SCORE: int = 1

    # Баллы для P/B
    PB_DEEP_VALUE_SCORE: int = 2
    PB_VALUE_SCORE: int = 1

    # Баллы для ROE
    ROE_HIGH_SCORE: int = 2
    ROE_GOOD_SCORE: int = 1

    # Баллы для дивидендов
    DIV_HIGH_SCORE: int = 2
    DIV_GOOD_SCORE: int = 1

    # Пороги для рекомендаций
    AGGRESSIVE_BUY_THRESHOLD: int = 6
    BUY_THRESHOLD: int = 4
    HOLD_THRESHOLD: int = 2


# ==================== КОНСТАНТЫ ПОРТФЕЛЯ ====================


@dataclass(frozen=True)
class PortfolioClusterConstants:
    """Константы для оптимизации портфеля с кластерами"""

    # Ограничения на веса
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.15
    MIN_WEIGHT_LOOSE: float = 0.02
    MAX_WEIGHT_LOOSE: float = 0.20

    # Корреляции внутри и между кластерами
    INTRA_CLUSTER_CORRELATION: float = 0.4
    INTER_CLUSTER_CORRELATION: float = 0.15

    # Базовые значения доходности и риска
    BASE_EXPECTED_RETURN: float = 0.10
    BASE_RISK: float = 0.18
    MIN_RISK: float = 0.10
    MAX_RISK: float = 0.50

    # Пороги для отбора кандидатов
    MIN_MARKET_CAP: float = 1e9
    MIN_MARKET_CAP_LOOSE: float = 0.5e9
    MIN_EXPECTED_RETURN: float = 0.08
    MAX_RISK_THRESHOLD: float = 0.40

    # Максимальное количество кандидатов
    MAX_CANDIDATES: int = 20

    # Веса для общего скора
    VALUE_SCORE_WEIGHT: float = 0.3
    QUALITY_SCORE_WEIGHT: float = 0.3
    INCOME_SCORE_WEIGHT: float = 0.2
    RETURN_SCORE_WEIGHT: float = 0.2
    RETURN_SCORE_MULTIPLIER: float = 100.0

    # Стратегии портфелей по умолчанию
    DEFAULT_STRATEGIES: Tuple[str, ...] = (
        "balanced",
        "value",
        "dividend",
        "cluster_based",
    )

    # Топ позиций для отображения
    TOP_POSITIONS_N: int = 10
    TOP_POSITIONS_RECOMMEND: int = 5
    TOP_IN_CLUSTER_N: int = 2


@dataclass(frozen=True)
class ReturnPremiumsCluster:
    """Премии к доходности для кластерного анализа"""

    # Премии за P/E
    PE_DEEP_PREMIUM: float = 0.12
    PE_VALUE_PREMIUM: float = 0.08
    PE_FAIR_PREMIUM: float = 0.04

    # Премии за P/B
    PB_DEEP_PREMIUM: float = 0.08
    PB_VALUE_PREMIUM: float = 0.05
    PB_FAIR_PREMIUM: float = 0.02

    # Премии за ROE
    ROE_HIGH_PREMIUM: float = 0.06
    ROE_GOOD_PREMIUM: float = 0.04
    ROE_MEDIUM_PREMIUM: float = 0.02

    # Коэффициент для дивидендной премии
    DIVIDEND_PREMIUM_FACTOR: float = 0.7

    # Максимальная доходность
    MAX_RETURN: float = 0.30


@dataclass(frozen=True)
class RiskPremiumsCluster:
    """Премии за риск для кластерного анализа"""

    # Коэффициент для беты
    BETA_RISK_FACTOR: float = 0.08

    # Премии за долг
    DEBT_CRITICAL_PENALTY: float = 0.08
    DEBT_HIGH_PENALTY: float = 0.05
    DEBT_MEDIUM_PENALTY: float = 0.02

    # Премии за P/E
    PE_EXTREME_PENALTY: float = 0.05
    PE_MISSING_PENALTY: float = 0.03


# ==================== КОНСТАНТЫ СЕКТОРОВ ====================


@dataclass(frozen=True)
class SectorKeywordsCluster:
    """Ключевые слова для определения секторов (расширенные)"""

    BANKS: Tuple[str, ...] = ("банк", "сбер", "втб", "мтс-банк", "мкб", "совкомбанк")
    OIL_GAS: Tuple[str, ...] = (
        "нефть",
        "газ",
        "газпром",
        "лукойл",
        "роснефть",
        "татнефть",
        "сургутнефтегаз",
        "башнефть",
        "славнефть",
        "русснефть",
    )
    METALS: Tuple[str, ...] = (
        "металл",
        "золото",
        "норильск",
        "северсталь",
        "нлмк",
        "ммк",
        "полюс",
        "распадская",
        "мечел",
        "чмк",
    )
    ENERGY: Tuple[str, ...] = (
        "энерго",
        "росcети",
        "тгк",
        "мосэнерго",
        "интер рао",
        "юнипро",
        "форвард",
        "дэк",
    )
    TELECOM: Tuple[str, ...] = ("связь", "телеком", "ростелеком", "мтс", "таттелеком")
    RETAIL: Tuple[str, ...] = (
        "магнит",
        "лента",
        "пятерочка",
        "фикс прайс",
        "окей",
        "м.видео",
    )
    CHEMICAL: Tuple[str, ...] = (
        "хим",
        "фосагро",
        "акрон",
        "азот",
        "казаньоргсинтез",
        "нижнекамскнефтехим",
        "куйбышевазот",
    )
    IT: Tuple[str, ...] = (
        "яндекс",
        "циан",
        "вк",
        "астра",
        "софтлайн",
        "диасофт",
        "позитив",
    )

    DEFAULT: str = "Другие"


@dataclass(frozen=True)
class SectorNamesCluster:
    """Названия секторов"""

    BANKS: str = "Банки"
    OIL_GAS: str = "Нефть и газ"
    METALS: str = "Металлургия"
    ENERGY: str = "Энергетика"
    TELECOM: str = "Телекоммуникации"
    RETAIL: str = "Ритейл"
    CHEMICAL: str = "Химическая промышленность"
    IT: str = "IT и технологии"
    OTHER: str = "Другие"


# ==================== КОНСТАНТЫ СКОРИНГА ====================


@dataclass(frozen=True)
class ScoringConstants:
    """Константы для скоринга компаний"""

    # Начальное значение скора
    BASE_SCORE: int = 50
    MIN_SCORE: int = 0
    MAX_SCORE: int = 100

    # Баллы для P/E
    PE_DEEP_VALUE_BONUS: int = 30
    PE_VALUE_BONUS: int = 20
    PE_FAIR_BONUS: int = 10
    PE_OVER_PENALTY: int = -20

    # Баллы для P/B
    PB_DEEP_VALUE_BONUS: int = 25
    PB_VALUE_BONUS: int = 15
    PB_FAIR_BONUS: int = 5
    PB_OVER_PENALTY: int = -15

    # Баллы для ROE
    ROE_HIGH_BONUS: int = 30
    ROE_GOOD_BONUS: int = 20
    ROE_MEDIUM_BONUS: int = 10
    ROE_LOW_BONUS: int = 5
    ROE_NEGATIVE_PENALTY: int = -30

    # Баллы для долга
    DEBT_LOW_BONUS: int = 20
    DEBT_MEDIUM_BONUS: int = 10
    DEBT_HIGH_BONUS: int = 5
    DEBT_CRITICAL_PENALTY: int = -20

    # Баллы для дивидендов
    DIV_HIGH_BONUS: int = 30
    DIV_GOOD_BONUS: int = 20
    DIV_MEDIUM_BONUS: int = 15
    DIV_LOW_BONUS: int = 10
    DIV_POOR_BONUS: int = 5
    DIV_MIN_PENALTY: int = -10

    # Баллы для роста (P/S)
    PS_HIGH_BONUS: int = 15
    PS_GOOD_BONUS: int = 10
    PS_MEDIUM_BONUS: int = 5


# ==================== КОНСТАНТЫ ФАЙЛОВ ====================


@dataclass(frozen=True)
class ClusterFileConstants:
    """Константы для файлов кластерного анализа"""

    # Имена файлов
    CLUSTER_ANALYSIS_FILE: str = "cluster_analysis.png"
    CLUSTER_OPTIMIZATION_FILE: str = "cluster_optimization.png"
    PORTFOLIO_COMPARISON_FILE: str = "portfolio_comparison.png"
    CLUSTER_ALLOCATION_FILE: str = "cluster_allocation.png"
    CLUSTERED_COMPANIES_FILE: str = "clustered_companies.xlsx"
    INVESTMENT_CLUSTER_REPORT: str = "investment_cluster_report.xlsx"

    # Названия листов в Excel
    SHEET_PORTFOLIO_SUMMARY: str = "Сводка_портфелей"
    SHEET_CLUSTERS: str = "Кластеры"
    SHEET_ALL_COMPANIES: str = "Все_компании"

    # Размеры графиков
    FIGURE_SIZE_OPTIMIZATION: Tuple[int, int] = (14, 5)
    FIGURE_SIZE_CLUSTERS: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_COMPARISON: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_ALLOCATION: Tuple[int, int] = (14, 6)

    DPI: int = 300


# ==================== КОНСТАНТЫ ФОРМАТИРОВАНИЯ ====================


@dataclass(frozen=True)
class ClusterFormattingConstants:
    """Константы для форматирования вывода в кластерном анализе"""

    # Форматы чисел
    PERCENT_FORMAT: str = "{:.1%}"
    PERCENT_FORMAT_2D: str = "{:.2%}"
    FLOAT_FORMAT_1D: str = "{:.1f}"
    FLOAT_FORMAT_2D: str = "{:.2f}"

    # Форматы для matplotlib
    MATPLOTLIB_PERCENT: str = "%1.1f%%"
    MATPLOTLIB_PERCENT_2D: str = "%1.2f%%"

    # Строки для отображения N/A
    NA_STRING: str = "N/A"

    # Разделители
    SEPARATOR: str = "=" * 80
    SUB_SEPARATOR: str = "-" * 40

    # Цвета для графиков
    COLOR_CLUSTER_CMAP: str = "viridis"
    COLOR_SHARPE_CMAP: str = "RdYlGn"
    COLOR_DIVERSIFICATION: str = "skyblue"
    COLOR_PORTFOLIO_BG: str = "#f0f8ff"
    COLOR_PORTFOLIO_EDGE: str = "#4682b4"
    COLOR_CENTROID: str = "red"
    COLOR_SHARPE_TARGET: str = "r"

    # Размеры шрифтов
    TITLE_FONT_SIZE: int = 16
    SUBTITLE_FONT_SIZE: int = 14
    AXIS_FONT_SIZE: int = 12
    LABEL_FONT_SIZE: int = 11
    ANNOTATION_FONT_SIZE: int = 9
    BAR_TEXT_FONT_SIZE: int = 9

    # Эксплозия для pie chart
    PIE_EXPLODE_FACTOR: float = 0.05


# ==================== КОНСТАНТЫ ОТЧЕТОВ ====================


@dataclass(frozen=True)
class ClusterReportConstants:
    """Константы для отчетов кластерного анализа"""

    # Названия колонок в отчете
    COL_PORTFOLIO: str = "Портфель"
    COL_RETURN: str = "Доходность"
    COL_RISK: str = "Риск"
    COL_SHARPE: str = "Шарп"
    COL_DIVERSIFICATION: str = "Диверсификация"
    COL_N_POSITIONS: str = "Кол-во позиций"

    COL_CLUSTER: str = "Кластер"
    COL_CLUSTER_SIZE: str = "Кол-во компаний"
    COL_AVG_PE: str = "Средний P/E"
    COL_AVG_PB: str = "Средний P/B"
    COL_AVG_ROE: str = "Средний ROE"
    COL_AVG_DIV: str = "Средняя див.доходность"
    COL_RISK_CLUSTER: str = "Риск"
    COL_DESCRIPTION: str = "Описание"
    COL_RECOMMENDATION: str = "Рекомендация"

    # Описания кластеров
    DESC_DEEP_VALUE: str = "Глубоко недооцененные value-акции"
    DESC_VALUE: str = "Недооцененные value-акции"
    DESC_GROWTH_OVER: str = "Переоцененные growth-акции"
    DESC_HIGH_PROFIT: str = "Высокорентабельные компании"
    DESC_PROFIT: str = "Прибыльные компании"
    DESC_FAIR: str = "Справедливо оцененные компании"

    # Рекомендации
    REC_AGGRESSIVE_BUY: str = "Агрессивная покупка"
    REC_BUY: str = "Покупка"
    REC_HOLD: str = "Держать"
    REC_AVOID: str = "Избегать"


# Создаем экземпляры констант для удобного импорта
CLUSTER = ClusterConstants()
CLUSTER_THRESHOLDS = ClusterThresholds()
CLUSTER_SCORES = ClusterScores()
PORTFOLIO_CLUSTER = PortfolioClusterConstants()
RETURN_PREMIUMS_CLUSTER = ReturnPremiumsCluster()
RISK_PREMIUMS_CLUSTER = RiskPremiumsCluster()
SECTOR_KEYWORDS_CLUSTER = SectorKeywordsCluster()
SECTOR_NAMES_CLUSTER = SectorNamesCluster()
SCORING = ScoringConstants()
CLUSTER_FILES = ClusterFileConstants()
CLUSTER_FORMAT = ClusterFormattingConstants()
CLUSTER_REPORT = ClusterReportConstants()
