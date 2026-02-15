"""
Константы для модуля автоэнкодера и оптимизации портфеля.
Все магические числа вынесены в именованные константы.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List


# ==================== КОНСТАНТЫ АВТОЭНКОДЕРА ====================


@dataclass(frozen=True)
class AEArchitectureConstants:
    """Константы архитектуры автоэнкодера"""

    # Размеры слоев энкодера
    ENCODER_LAYER_1: int = 64
    ENCODER_LAYER_2: int = 32
    ENCODER_LAYER_3: int = 16
    ENCODER_LAYER_4: int = 8

    # Размеры слоев декодера
    DECODER_LAYER_1: int = 16
    DECODER_LAYER_2: int = 32
    DECODER_LAYER_3: int = 64

    # Параметры обучения
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 16
    N_EPOCHS: int = 100
    EPOCH_LOG_INTERVAL: int = 20

    # Функции активации
    ACTIVATION: str = "relu"


@dataclass(frozen=True)
class AEThresholdConstants:
    """Пороговые значения для автоэнкодера"""

    # Пороги для аномалий
    ANOMALY_IQR_MULTIPLIER: float = 1.5
    STRONG_ANOMALY_IQR_MULTIPLIER: float = 3.0

    # Пороги для признаков (P/E, P/B и т.д.)
    PE_STRONG_UNDERVALUED: float = 5.0
    PE_UNDERVALUED: float = 10.0
    PE_OVERVALUED: float = 25.0

    PS_STRONG_UNDERVALUED: float = 5.0
    PS_UNDERVALUED: float = 10.0
    PS_OVERVALUED: float = 25.0

    PB_STRONG_UNDERVALUED: float = 0.8
    PB_UNDERVALUED: float = 1.2

    ROE_HIGH: float = 20.0
    ROE_MEDIUM: float = 15.0

    # Процентили
    Q1_PERCENTILE: float = 25.0
    Q3_PERCENTILE: float = 75.0

    # Порог для значимых весов в портфеле
    SIGNIFICANT_WEIGHT_THRESHOLD: float = 0.01

    # Порог для ошибки реконструкции
    ERROR_MEDIAN_FACTOR: float = 0.5

    # Порог для скора недооцененности
    IDEAL_PROFILE_UNDERVALUED_FACTOR: float = 0.8
    IDEAL_PROFILE_OVERVAULED_FACTOR: float = 1.2

    # Порог для статистического скора
    STAT_SCORE_HIGH: float = 2.5


# ==================== КОНСТАНТЫ ПОРТФЕЛЯ ====================


@dataclass(frozen=True)
class AEPortfolioConstants:
    """Константы для оптимизации портфеля с автоэнкодером"""

    # Базовые значения
    BASE_RETURN: float = 0.10
    BASE_VOLATILITY: float = 0.18
    RISK_FREE_RATE: float = 0.08

    # Ограничения на веса
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.20

    # Премии за недооцененность
    UNDERVALUED_SCORE_PREMIUM: float = 0.15
    TOP_UNDERVALUED_PREMIUM: float = 0.05

    # Штрафы за аномалии
    ANOMALY_PENALTY: float = -0.08
    STRONG_ANOMALY_PENALTY: float = -0.15

    # Премии за P/E
    PE_STRONG_PREMIUM: float = 0.05
    PE_MEDIUM_PREMIUM: float = 0.03
    PE_OVER_PENALTY: float = -0.03

    PS_STRONG_PREMIUM: float = 0.05
    PS_MEDIUM_PREMIUM: float = 0.03
    PS_OVER_PENALTY: float = -0.03

    # Премии за P/B
    PB_STRONG_PREMIUM: float = 0.04
    PB_MEDIUM_PREMIUM: float = 0.02

    # Премии за ROE
    ROE_HIGH_PREMIUM: float = 0.04
    ROE_MEDIUM_PREMIUM: float = 0.02

    # Коэффициент для дивидендной премии
    DIVIDEND_PREMIUM_FACTOR: float = 0.8

    # Коэффициенты для волатильности
    ERROR_VOL_FACTOR: float = 0.5
    ANOMALY_VOL_PENALTY: float = 0.08
    STRONG_ANOMALY_VOL_PENALTY: float = 0.12
    TOP_UNDERVALUED_VOL_BONUS: float = -0.03

    BETA_VOL_FACTOR_MIN: float = 0.8
    BETA_VOL_FACTOR_MAX: float = 0.2

    DEBT_VOL_FACTOR_MIN: float = 0.9
    DEBT_VOL_FACTOR_MAX: float = 0.2
    DEBT_NORMALIZATION: float = 0.5

    # Максимальная/минимальная волатильность и доходность
    MAX_VOLATILITY: float = 0.50
    MIN_VOLATILITY: float = 0.10
    MAX_RETURN: float = 0.40
    MIN_RETURN: float = 0.05

    # Корреляции на основе ошибок реконструкции
    ERROR_CORR_HIGH: float = 0.6
    ERROR_CORR_MEDIUM: float = 0.4
    ERROR_CORR_LOW: float = 0.2
    ERROR_DIFF_LOW: float = 0.5
    ERROR_DIFF_MEDIUM: float = 1.0

    # VaR и CVaR коэффициенты
    VAR_95_COEFF: float = 1.645
    CVAR_95_COEFF: float = 2.063

    # Веса для комбинированного портфеля (с бустом недооцененных)
    SHARPE_WEIGHT_BOOST: float = 0.4
    MIN_RISK_WEIGHT_BOOST: float = 0.3
    MAX_RETURN_WEIGHT_BOOST: float = 0.3

    # Веса для комбинированного портфеля (без буста)
    SHARPE_WEIGHT_NORMAL: float = 0.5
    MIN_RISK_WEIGHT_NORMAL: float = 0.3
    MAX_RETURN_WEIGHT_NORMAL: float = 0.2

    # Параметры оптимизации
    OPTIMIZER_MAX_ITER: int = 1000

    # Пороги для отбора кандидатов
    MIN_CANDIDATES: int = 5
    MIN_CANDIDATES_LOOSE: int = 3
    MAX_CANDIDATES: int = 20
    MIN_EXPECTED_RETURN: float = 0.08
    MAX_VOLATILITY_THRESHOLD: float = 0.45
    MIN_PORTFOLIO_SIZE: int = 3

    # Веса для комбинированного скора
    UNDERVALUED_WEIGHT: float = 0.6
    RISK_WEIGHT: float = 0.4

    # Дивидендный порог
    DIVIDEND_YIELD_THRESHOLD: float = 0.03

    # Размер топ-недооцененных
    TOP_UNDERVALUED_N: int = 10


@dataclass(frozen=True)
class AEScoringConstants:
    """Константы для скоринга автоэнкодера"""

    # Баллы для признаков
    SCORE_STRONG_UNDERVALUED: int = 2
    SCORE_UNDERVALUED: int = 1

    # Коэффициент для комбинированного скора
    FUNDAMENTAL_SCORE_FACTOR: int = 2

    # Порог для хороших компаний
    GOOD_COMPANIES_MIN_COUNT: int = 5


# ==================== КОНСТАНТЫ ПРИЗНАКОВ ====================


@dataclass(frozen=True)
class AEFeatureConstants:
    """Константы для признаков автоэнкодера"""

    # Признаки по умолчанию
    DEFAULT_FEATURES: Tuple[str, ...] = (
        "dividend_yield",
        "P_E",
        "P_B",
        "P_S",
        "NPM",
        "EV_EBITDA",
        "ROE",
        "debt_capital",
    )

    # Признаки, где меньше = лучше
    LOWER_IS_BETTER_FEATURES: Tuple[str, ...] = (
        "P_E",
        "P_B",
        "P_S",
        "EV_EBITDA",
        "debt_capital",
    )

    # Признаки, где больше = лучше
    HIGHER_IS_BETTER_FEATURES: Tuple[str, ...] = ("dividend_yield", "ROE", "NPM")


@dataclass(frozen=True)
class AEColumnMapping:
    """Маппинг колонок для автоэнкодера"""

    # Маппинг Excel колонок в внутренние имена
    COLUMN_MAPPING: Dict[str, str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "COLUMN_MAPPING",
            {
                "P/E": "P_E",
                "P/B": "P_B",
                "P/S": "P_S",
                "EV/EBITDA": "EV_EBITDA",
                "Debt/Capital": "debt_capital",
            },
        )

    # Числовые колонки для конвертации
    NUMERIC_COLUMNS: Tuple[str, ...] = (
        "Рыночная капитализация",
        "Выручка",
        "Чистая прибыль",
        "EBITDA",
        "Свободный денежный поток",
        "Дивиденд на акцию",
    )

    # Суффиксы для конвертации
    BILLION_SUFFIX: str = "млрд"
    MILLION_SUFFIX: str = "млн"

    # Замена для конвертации
    BILLION_REPLACE: str = "e9"
    MILLION_REPLACE: str = "e6"


# ==================== КОНСТАНТЫ ФАЙЛОВ ====================


@dataclass(frozen=True)
class AEFileConstants:
    """Константы для файлов автоэнкодера"""

    # Имена файлов
    AE_PORTFOLIO_RESULTS: str = "ae_portfolio_results.xlsx"
    AE_ANOMALY_ANALYSIS: str = "ae_anomaly_analysis.png"
    AE_PORTFOLIO_COMPARISON: str = "ae_portfolio_comparison.png"
    AE_PORTFOLIO_OPTIMAL: str = "ae_portfolio_optimal"
    AE_PORTFOLIO_SUMMARY: str = "ae_portfolio_summary.png"

    # Названия листов в Excel
    SHEET_AE_RESULTS: str = "AE_Результаты"
    SHEET_CANDIDATES: str = "Кандидаты"
    SHEET_PORTFOLIO_SUMMARY: str = "Сводка_портфелей"
    SHEET_BEST_PORTFOLIO: str = "Лучший_портфель"

    # Размеры графиков
    FIGURE_SIZE_SUMMARY: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_COMPARISON: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_ANOMALY: Tuple[int, int] = (16, 12)

    DPI: int = 300

    # Количество портфелей для границы эффективности
    N_EFFICIENT_PORTFOLIOS: int = 3000


# ==================== КОНСТАНТЫ ФОРМАТИРОВАНИЯ ====================


@dataclass(frozen=True)
class AEFormattingConstants:
    """Константы для форматирования вывода в автоэнкодере"""

    # Форматы чисел
    PERCENT_FORMAT: str = "{:.1%}"
    PERCENT_FORMAT_2D: str = "{:.2%}"
    FLOAT_FORMAT_1D: str = "{:.1f}"
    FLOAT_FORMAT_2D: str = "{:.2f}"
    FLOAT_FORMAT_3D: str = "{:.3f}"
    FLOAT_FORMAT_4D: str = "{:.4f}"
    FLOAT_FORMAT_6D: str = "{:.6f}"

    # Форматы для matplotlib
    MATPLOTLIB_PERCENT: str = "%1.1f%%"
    MATPLOTLIB_PERCENT_2D: str = "%1.2f%%"

    # Строки для отображения N/A
    NA_STRING: str = "N/A"

    # Символы
    STAR_SYMBOL: str = "★"

    # Разделители
    SEPARATOR: str = "=" * 80
    SUB_SEPARATOR: str = "-" * 40

    # Цвета для графиков
    COLOR_ANOMALY: str = "#e74c3c"  # Красный
    COLOR_TOP_UNDERVALUED: str = "#2ecc71"  # Зеленый
    COLOR_NORMAL: str = "#3498db"  # Синий
    COLOR_ANOMALY_BG: str = "#f0f8ff"  # Светло-голубой
    COLOR_ANOMALY_EDGE: str = "#4682b4"  # Стальной синий

    COLOR_OPTIMAL_MARKER: str = "red"
    COLOR_CONFIDENCE_CMAP: str = "viridis"
    COLOR_SHARPE_CMAP: str = "RdYlGn"
    COLOR_RISK_CONTRIBUTION_CMAP: str = "RdYlGn_r"
    COLOR_DIVERSIFICATION: str = "skyblue"

    # Размеры шрифтов
    TITLE_FONT_SIZE: int = 16
    SUBTITLE_FONT_SIZE: int = 14
    AXIS_FONT_SIZE: int = 12
    LABEL_FONT_SIZE: int = 11
    ANNOTATION_FONT_SIZE: int = 9
    BAR_TEXT_FONT_SIZE: int = 9

    # Эксплозия для pie chart
    PIE_EXPLODE_FACTOR: float = 0.05

    # Топ позиции для отображения
    TOP_POSITIONS_SUMMARY: int = 5
    TOP_POSITIONS_BEST: int = 5
    TOP_RISK_CONTRIBUTION: int = 8

    # Количество бинов для гистограммы
    HISTOGRAM_BINS: int = 50

    # Размеры точек
    SCATTER_POINT_SIZE: int = 50
    SCATTER_POINT_SIZE_LARGE: int = 200
    SCATTER_POINT_SIZE_PORTFOLIO: int = 300
    WEIGHT_SCALE_FACTOR: int = 3000


# ==================== КОНСТАНТЫ РЕКОМЕНДАЦИЙ ====================


@dataclass(frozen=True)
class AERecommendationConstants:
    """Константы для рекомендаций автоэнкодера"""

    # Названия категорий
    CATEGORY_ANOMALIES: str = "Аномалии"
    CATEGORY_TOP_UNDERVALUED: str = "Топ_недооцененные"
    CATEGORY_NORMAL: str = "Обычные"

    # Цвета категорий
    CATEGORY_COLORS: Dict[str, str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "CATEGORY_COLORS",
            {
                self.CATEGORY_ANOMALIES: "#e74c3c",
                self.CATEGORY_TOP_UNDERVALUED: "#2ecc71",
                self.CATEGORY_NORMAL: "#3498db",
            },
        )


# Создаем экземпляры констант
AE_ARCH = AEArchitectureConstants()
AE_THRESHOLD = AEThresholdConstants()
AE_PORTFOLIO = AEPortfolioConstants()
AE_SCORING = AEScoringConstants()
AE_FEATURE = AEFeatureConstants()
AE_COLUMN = AEColumnMapping()
AE_FILES = AEFileConstants()
AE_FORMAT = AEFormattingConstants()
AE_REC = AERecommendationConstants()
