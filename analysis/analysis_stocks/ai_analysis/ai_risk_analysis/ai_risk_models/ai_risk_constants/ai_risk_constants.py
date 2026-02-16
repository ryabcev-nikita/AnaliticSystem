"""
Константы для модуля нейросетевого анализа рисков и оптимизации портфеля.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ==================== КОНСТАНТЫ НЕЙРОСЕТИ ====================


@dataclass(frozen=True)
class NNArchitectureConstants:
    """Константы архитектуры нейросети"""

    # Размеры слоев
    DENSE_LAYER_1: int = 128
    DENSE_LAYER_2: int = 64
    DENSE_LAYER_3: int = 32
    DENSE_LAYER_4: int = 16

    # Для residual сети
    RESIDUAL_LAYER_1: int = 256
    RESIDUAL_LAYER_2: int = 128

    # Широкая сеть
    WIDE_LAYER_1: int = 256
    WIDE_LAYER_2: int = 128
    WIDE_LAYER_3: int = 64
    WIDE_LAYER_4: int = 32

    # Dropout коэффициенты
    DROPOUT_HIGH: float = 0.4
    DROPOUT_MEDIUM: float = 0.3
    DROPOUT_LOW: float = 0.2

    # L2 регуляризация
    L2_REGULARIZER: float = 0.001

    # Параметры обучения
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 16
    EPOCHS: int = 30

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 8
    REDUCE_LR_PATIENCE: int = 5
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-6

    # Кросс-валидация
    N_FOLDS: int = 3
    RANDOM_STATE: int = 42


@dataclass(frozen=True)
class NNFeatureConstants:
    """Константы для признаков нейросети"""

    # Пороги для ограничения экстремальных значений
    PE_MAX: float = 100.0
    PE_MIN: float = -50.0

    PS_MAX: float = 100.0
    PS_MIN: float = -50.0

    EV_EBITDA_MAX: float = 100.0
    EV_EBITDA_MIN: float = -50.0

    ROE_MAX: float = 100.0
    ROE_MIN: float = -100.0

    ROA_MAX: float = 100.0
    ROA_MIN: float = -100.0

    ROIC_MAX: float = 100.0
    ROIC_MIN: float = -100.0

    NPM_MAX: float = 100.0
    NPM_MIN: float = -100.0

    DEBT_CAPITAL_MAX: float = 10.0
    DEBT_CAPITAL_MIN: float = 0.0

    DEBT_EBITDA_MAX: float = 10.0
    DEBT_EBITDA_MIN: float = 0.0

    # Квантили для очистки выбросов
    Q1_PERCENTILE: float = 25.0
    Q3_PERCENTILE: float = 75.0
    IQR_MULTIPLIER: float = 3.0


@dataclass(frozen=True)
class NNThresholdConstants:
    """Пороговые значения для нейросетевого анализа"""

    # Пороги для P/E, P/B и т.д. (чем меньше, тем лучше)
    PE_RISK_FREE: float = 0.0
    PE_LOW_RISK: float = 0.7  # 70% от медианы
    PE_MEDIUM_RISK: float = 1.0  # 100% от медианы
    PE_HIGH_RISK: float = 1.5  # 150% от медианы

    # Пороги для ROE, ROA и т.д. (чем больше, тем лучше)
    ROE_LOW_RISK: float = 1.3  # 130% от медианы
    ROE_MEDIUM_RISK: float = 1.0  # 100% от медианы
    ROE_HIGH_RISK: float = 0.7  # 70% от медианы

    # Пороги для Beta
    BETA_VERY_LOW: float = 0.5
    BETA_LOW: float = 1.0
    BETA_MEDIUM: float = 1.5
    BETA_HIGH: float = 2.0

    # Пороги для отклонения от медианы
    DEVIATION_LOW: float = 0.5
    DEVIATION_MEDIUM: float = 1.0
    DEVIATION_HIGH: float = 2.0

    # Пороги для статистического скора
    STAT_SCORE_HIGH: float = 2.5


# ==================== КОНСТАНТЫ РИСКА ====================


@dataclass(frozen=True)
class RiskCategoryConstants:
    """Константы категорий риска"""

    # Коды категорий
    RISK_A_CODE: int = 0  # Низкий риск
    RISK_B_CODE: int = 1  # Средний риск
    RISK_C_CODE: int = 2  # Высокий риск
    RISK_D_CODE: int = 3  # Очень высокий риск

    # Названия категорий
    RISK_A_NAME: str = "A: Низкий риск"
    RISK_B_NAME: str = "B: Средний риск"
    RISK_C_NAME: str = "C: Высокий риск"
    RISK_D_NAME: str = "D: Очень высокий риск"

    # Описания категорий
    RISK_A_DESC: str = "Низкий риск"
    RISK_B_DESC: str = "Средний риск"
    RISK_C_DESC: str = "Высокий риск"
    RISK_D_DESC: str = "Очень высокий риск"

    # Маппинг кодов в названия
    CATEGORY_MAP: Dict[int, str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "CATEGORY_MAP",
            {
                self.RISK_A_CODE: self.RISK_A_NAME,
                self.RISK_B_CODE: self.RISK_B_NAME,
                self.RISK_C_CODE: self.RISK_C_NAME,
                self.RISK_D_CODE: self.RISK_D_NAME,
            },
        )


@dataclass(frozen=True)
class RiskScoreConstants:
    """Константы для скоринга риска"""

    # Уровни риска
    RISK_VERY_LOW: int = 0
    RISK_LOW: int = 1
    RISK_MEDIUM: int = 2
    RISK_HIGH: int = 3
    RISK_VERY_HIGH: int = 4

    # Веса для признаков
    WEIGHT_HIGH: float = 2.0
    WEIGHT_MEDIUM: float = 1.5
    WEIGHT_ABOVE_AVG: float = 1.2
    WEIGHT_BASE: float = 1.0

    # Список признаков с высокой важностью
    HIGH_IMPORTANCE_FEATURES: Tuple[str, ...] = (
        "P/E",
        "ROE",
        "dividend_yield",
        "Debt/EBITDA",
    )
    MEDIUM_IMPORTANCE_FEATURES: Tuple[str, ...] = (
        "P/B",
        "P/S",
        "ROA",
        "NPM",
        "EV/EBITDA",
    )
    ABOVE_AVG_IMPORTANCE_FEATURES: Tuple[str, ...] = ("Beta", "Бета", "EPS")

    # Вес для автоэнкодера
    AE_WEIGHT: float = 2.0
    AE_RISK_SCORE: int = 3


# ==================== КОНСТАНТЫ ПОРТФЕЛЯ ====================


@dataclass(frozen=True)
class NNRiskPortfolioConstants:
    """Константы для оптимизации портфеля с нейросетевым риском"""

    # Базовые значения
    BASE_RETURN: float = 0.155
    BASE_VOLATILITY: float = 0.10
    RISK_FREE_RATE: float = 0.155

    # Ограничения на веса
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.15

    # Премии за категории риска
    RISK_A_PREMIUM: float = 0.02
    RISK_B_PREMIUM: float = 0.04
    RISK_C_PREMIUM: float = 0.07
    RISK_D_PREMIUM: float = 0.10

    # Премии за ROE
    ROE_HIGH_PREMIUM: float = 0.03
    ROE_MEDIUM_PREMIUM: float = 0.02
    ROE_HIGH_THRESHOLD: float = 20.0
    ROE_MEDIUM_THRESHOLD: float = 15.0

    # Коэффициент для дивидендной премии
    DIVIDEND_PREMIUM_FACTOR: float = 0.8

    # Коэффициент уверенности
    CONFIDENCE_MIN_FACTOR: float = 0.8
    CONFIDENCE_MAX_FACTOR: float = 0.4
    CONFIDENCE_SCALE: float = 0.8

    # Максимальная/минимальная доходность
    MAX_RETURN: float = 0.50
    MIN_RETURN: float = 0.05

    # Базовая волатильность по категориям
    VOLATILITY_A: float = 0.12
    VOLATILITY_B: float = 0.18
    VOLATILITY_C: float = 0.25
    VOLATILITY_D: float = 0.35

    # Коэффициенты для Beta
    BETA_VOL_FACTOR_MIN: float = 0.7
    BETA_VOL_FACTOR_MAX: float = 0.3

    # Коэффициенты для долга
    DEBT_VOL_FACTOR_MIN: float = 0.9
    DEBT_VOL_FACTOR_MAX: float = 0.2
    DEBT_NORMALIZATION: float = 50.0

    # Максимальная/минимальная волатильность
    MAX_VOLATILITY: float = 0.50
    MIN_VOLATILITY: float = 0.10

    # Корреляции внутри и между категориями риска
    INTRA_CATEGORY_CORRELATION: float = 0.5
    INTER_CATEGORY_CORRELATION: float = 0.25

    # VaR и CVaR коэффициенты
    VAR_95_COEFF: float = 1.645
    CVAR_95_COEFF: float = 2.063

    # Веса для комбинированного портфеля
    SHARPE_PORTFOLIO_WEIGHT: float = 0.5
    MIN_RISK_PORTFOLIO_WEIGHT: float = 0.3
    MAX_RETURN_PORTFOLIO_WEIGHT: float = 0.2

    # Параметры оптимизации
    OPTIMIZER_MAX_ITER: int = 1000

    # Пороги для отбора кандидатов
    MIN_CONFIDENCE: float = 0.5
    MIN_EXPECTED_RETURN: float = 0.155
    MAX_VOLATILITY_THRESHOLD: float = 0.50
    MAX_CANDIDATES: int = 30

    # Веса для скора кандидатов
    RISK_SCORE_WEIGHT: float = 0.4
    CONFIDENCE_WEIGHT: float = 0.3
    RETURN_WEIGHT: float = 0.3
    RETURN_NORMALIZATION: float = 0.2


# ==================== КОНСТАНТЫ ПРИЗНАКОВ ====================


@dataclass(frozen=True)
class NNFeatureAliases:
    """Алиасы признаков для нейросети"""

    PE_ALIASES: Tuple[str, ...] = ("P/E", "P_E", "PE", "Price/Earnings")
    PB_ALIASES: Tuple[str, ...] = ("P/B", "P_B", "PB", "Price/Book")
    PS_ALIASES: Tuple[str, ...] = ("P/S", "P_S", "PS", "Price/Sales")
    PFCF_ALIASES: Tuple[str, ...] = ("P/FCF", "P_FCF", "PFCF", "Price/FCF")
    EV_EBITDA_ALIASES: Tuple[str, ...] = (
        "EV/EBITDA",
        "EV_EBITDA",
        "EV/EBIT",
        "EV_EBIT",
    )
    EV_S_ALIASES: Tuple[str, ...] = ("EV/S", "EV_S", "EV/Sales", "EV_Sales")
    ROE_ALIASES: Tuple[str, ...] = ("ROE", "Рентабельность капитала")
    ROA_ALIASES: Tuple[str, ...] = ("ROA", "Рентабельность активов")
    ROIC_ALIASES: Tuple[str, ...] = ("ROIC", "Рентабельность инвестиций")
    NPM_ALIASES: Tuple[str, ...] = ("NPM", "Чистая маржа", "Net Profit Margin")
    EBITDA_MARGIN_ALIASES: Tuple[str, ...] = (
        "EBITDA margin",
        "Рентабельность EBITDA",
        "EBITDA_Margin",
    )
    DEBT_CAPITAL_ALIASES: Tuple[str, ...] = (
        "Debt/Capital",
        "debt_capital",
        "Debt_to_Capital",
        "Долг/Капитал",
    )
    DEBT_EBITDA_ALIASES: Tuple[str, ...] = ("Debt/EBITDA", "Debt_EBITDA", "Долг/EBITDA")
    NET_DEBT_EBITDA_ALIASES: Tuple[str, ...] = (
        "Net_Debt/EBITDA",
        "Net_Debt_EBITDA",
        "Чистый долг/EBITDA",
    )
    DIV_YIELD_ALIASES: Tuple[str, ...] = (
        "dividend_yield",
        "Averange_dividend_yield",
        "Дивидендная доходность",
    )
    EPS_ALIASES: Tuple[str, ...] = ("EPS", "Прибыль на акцию")
    BETA_ALIASES: Tuple[str, ...] = ("Beta", "Бета", "beta", "БЕТА", "b")

    # Полный маппинг признаков
    FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "FEATURE_ALIASES",
            {
                "P/E": self.PE_ALIASES,
                "P/B": self.PB_ALIASES,
                "P/S": self.PS_ALIASES,
                # "P/FCF": self.PFCF_ALIASES,
                # "EV/EBITDA": self.EV_EBITDA_ALIASES,
                # "EV/S": self.EV_S_ALIASES,
                "ROE": self.ROE_ALIASES,
                # "ROA": self.ROA_ALIASES,
                # "ROIC": self.ROIC_ALIASES,
                "NPM": self.NPM_ALIASES,
                # "EBITDA margin": self.EBITDA_MARGIN_ALIASES,
                "Debt/Capital": self.DEBT_CAPITAL_ALIASES,
                # "Debt/EBITDA": self.DEBT_EBITDA_ALIASES,
                # "Net Debt/EBITDA": self.NET_DEBT_EBITDA_ALIASES,
                "dividend_yield": self.DIV_YIELD_ALIASES,
                # "EPS": self.EPS_ALIASES,
                "Beta": self.BETA_ALIASES,
            },
        )


@dataclass(frozen=True)
class NNRiskFileConstants:
    """Константы для файлов нейросетевого анализа рисков"""

    # Имена файлов
    NN_RISK_PORTFOLIO_BASE: str = "nn_risk_portfolio"
    NN_RISK_EFFICIENT_FRONTIER: str = "nn_risk_efficient_frontier.png"
    NN_RISK_PORTFOLIO_COMPARISON: str = "nn_risk_portfolio_comparison.png"
    NN_RISK_PORTFOLIO_RESULTS: str = "nn_risk_portfolio_results.xlsx"

    # Названия листов в Excel
    SHEET_STOCKS_WITH_RISK: str = "Акции_с_рисками"
    SHEET_CANDIDATES: str = "Кандидаты"
    SHEET_PORTFOLIO_SUMMARY: str = "Сводка_портфелей"
    SHEET_BEST_PORTFOLIO: str = "Лучший_портфель"

    # Размеры графиков
    FIGURE_SIZE_SUMMARY: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_COMPARISON: Tuple[int, int] = (16, 12)
    FIGURE_SIZE_FRONTIER: Tuple[int, int] = (14, 8)

    DPI: int = 300

    # Количество портфелей для границы эффективности
    N_EFFICIENT_PORTFOLIOS: int = 3000


@dataclass(frozen=True)
class NNRiskFormattingConstants:
    """Константы для форматирования вывода в нейросетевом анализе"""

    # Форматы чисел
    PERCENT_FORMAT: str = "{:.1%}"
    PERCENT_FORMAT_2D: str = "{:.2%}"
    FLOAT_FORMAT_1D: str = "{:.1f}"
    FLOAT_FORMAT_2D: str = "{:.2f}"
    FLOAT_FORMAT_3D: str = "{:.3f}"

    # Форматы для matplotlib
    MATPLOTLIB_PERCENT: str = "%1.1f%%"
    MATPLOTLIB_PERCENT_2D: str = "%1.2f%%"

    # Строки для отображения N/A
    NA_STRING: str = "N/A"

    # Разделители
    SEPARATOR: str = "=" * 80
    SUB_SEPARATOR: str = "-" * 40

    # Цвета для графиков риска
    COLOR_RISK_A: str = "#2ecc71"  # Зеленый
    COLOR_RISK_B: str = "#f39c12"  # Оранжевый
    COLOR_RISK_C: str = "#e67e22"  # Темно-оранжевый
    COLOR_RISK_D: str = "#e74c3c"  # Красный
    COLOR_RISK_DEFAULT: str = "#3498db"  # Синий

    COLOR_CONFIDENCE_CMAP: str = "RdYlGn"
    COLOR_RISK_CONTRIBUTION_CMAP: str = "RdYlGn_r"
    COLOR_EFFICIENT_CMAP: str = "viridis"
    COLOR_PORTFOLIO_CMAP: str = "Set1"

    COLOR_PORTFOLIO_BG: str = "#f0f8ff"
    COLOR_PORTFOLIO_EDGE: str = "#4682b4"
    COLOR_OPTIMAL_MARKER: str = "red"
    COLOR_MARKET_MARKER: str = "white"

    # Размеры шрифтов
    TITLE_FONT_SIZE: int = 16
    SUBTITLE_FONT_SIZE: int = 14
    AXIS_FONT_SIZE: int = 12
    LABEL_FONT_SIZE: int = 11
    ANNOTATION_FONT_SIZE: int = 9
    BAR_TEXT_FONT_SIZE: int = 9

    # Эксплозия для pie chart
    PIE_EXPLODE_FACTOR: float = 0.05

    # Топ позиций для отображения
    TOP_POSITIONS_SUMMARY: int = 5
    TOP_POSITIONS_BEST: int = 5
    TOP_RISK_CONTRIBUTION: int = 8


@dataclass(frozen=True)
class NNRiskRecommendationConstants:
    """Константы для рекомендаций по риску"""

    # Рекомендации для категории A
    RISK_A_RECOMMENDATION: str = (
        "Рекомендуется для консервативного портфеля. Показатели значительно лучше медианных."
    )
    RISK_A_ALLOCATION: str = "5-15%"
    RISK_A_MONITORING: str = "Ежеквартально"

    # Рекомендации для категории B
    RISK_B_RECOMMENDATION: str = (
        "Подходит для сбалансированного портфеля. Показатели вблизи медианных значений."
    )
    RISK_B_ALLOCATION: str = "3-8%"
    RISK_B_MONITORING: str = "Ежемесячно"

    # Рекомендации для категории C
    RISK_C_RECOMMENDATION: str = (
        "Только для агрессивных инвесторов. Значительные отклонения от нормы."
    )
    RISK_C_ALLOCATION: str = "1-3%"
    RISK_C_MONITORING: str = "Еженедельно"

    # Рекомендации для категории D
    RISK_D_RECOMMENDATION: str = (
        "Спекулятивная позиция. Экстремальные отклонения от рыночных норм."
    )
    RISK_D_ALLOCATION: str = "0-1%"
    RISK_D_MONITORING: str = "Ежедневно"

    # Текст аномалии
    ANOMALY_NOTE: str = "Значительное отклонение от медианных значений"


# Создаем экземпляры констант
NN_ARCH = NNArchitectureConstants()
NN_FEATURE = NNFeatureConstants()
NN_THRESHOLD = NNThresholdConstants()
RISK_CAT = RiskCategoryConstants()
RISK_SCORE = RiskScoreConstants()
NN_PORTFOLIO = NNRiskPortfolioConstants()
NN_FEATURE_ALIASES = NNFeatureAliases()
NN_FILES = NNRiskFileConstants()
NN_FORMAT = NNRiskFormattingConstants()
NN_REC = NNRiskRecommendationConstants()
