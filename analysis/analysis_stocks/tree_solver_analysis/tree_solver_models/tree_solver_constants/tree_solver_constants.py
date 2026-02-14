"""
Константы для модуля построения деревья решений и оптимизации портфеля.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ==================== ФИНАНСОВЫЕ КОНСТАНТЫ ====================


@dataclass(frozen=True)
class FinancialConstants:
    """Финансовые константы и пороговые значения"""

    # Безрисковая ставка и базовые доходности
    RISK_FREE_RATE: float = 0.155  # 8% - базовая безрисковая ставка
    BASE_RETURN: float = 0.155  # Базовая ожидаемая доходность

    # Пороги для оценки недооцененности
    STRONGLY_UNDERVALUED_THRESHOLD: float = 0.7  # 70% от медианы
    UNDERVALUED_THRESHOLD: float = 1.0  # 100% от медианы
    OVERVALUED_THRESHOLD: float = 1.5  # 150% от медианы

    # Специальные пороги для P/B
    PB_STRONG_THRESHOLD: float = 0.8  # 80% от медианы
    PB_OVERVAULED_THRESHOLD: float = 1.5  # 150% от медианы

    # Пороги для ROE
    ROE_STRONG_THRESHOLD: float = 1.5  # 150% от медианы
    ROE_GOOD_THRESHOLD: float = 1.0  # 100% от медианы

    # Пороги для дивидендов
    DIVIDEND_STRONG_THRESHOLD: float = 2.0  # 200% от медианы
    DIVIDEND_GOOD_THRESHOLD: float = 1.0  # 100% от медианы

    # Пороги для беты
    BETA_HIGH_THRESHOLD: float = 1.3  # 130% от медианы
    BETA_LOW_THRESHOLD: float = 0.7  # 70% от медианы

    # Пороги для долга
    DEBT_HIGH_THRESHOLD: float = 1.5  # 150% от медианы
    DEBT_MEDIUM_THRESHOLD: float = 1.0  # 100% от медианы


@dataclass(frozen=True)
class ValuationScores:
    """Баллы для оценки стоимости"""

    STRONG_BUY: int = 2  # Сильно недооценена
    BUY: int = 1  # Недооценена
    HOLD: int = 0  # Справедливая оценка
    SELL: int = -1  # Переоценена

    # Пороги для итоговой оценки
    STRONG_BUY_THRESHOLD: int = 4  # Сумма баллов для сильной недооценки
    BUY_THRESHOLD: int = 2  # Сумма баллов для недооценки
    SELL_THRESHOLD: int = -2  # Сумма баллов для переоценки


@dataclass(frozen=True)
class ReturnPremiums:
    """Премии к доходности"""

    STRONG_PE_PREMIUM: float = 0.15  # 15% за сильную недооценку по P/E
    PE_PREMIUM: float = 0.08  # 8% за недооценку по P/E

    STRONG_PS_PREMIUM: float = 0.15  # 15% за сильную недооценку по P/S
    PS_PREMIUM: float = 0.08  # 8% за недооценку по P/S

    STRONG_PB_PREMIUM: float = 0.10  # 10% за сильную недооценку по P/B
    PB_PREMIUM: float = 0.05  # 5% за недооценку по P/B

    STRONG_ROE_PREMIUM: float = 0.12  # 12% за высокий ROE
    ROE_PREMIUM: float = 0.06  # 6% за ROE выше медианы

    STRONG_DIVIDEND_PREMIUM: float = 0.08  # 8% за высокую дивидендную доходность
    DIVIDEND_PREMIUM: float = 0.04  # 4% за дивдоходность выше медианы

    MODEL_STRONG_PREMIUM: float = 0.15  # 15% за прогноз "сильно недооценена"
    MODEL_PREMIUM: float = 0.08  # 8% за прогноз "недооценена"


@dataclass(frozen=True)
class RiskPremiums:
    """Премии за риск"""

    BASE_RISK: float = 0.155  # 15% - базовый риск

    BETA_HIGH_PENALTY: float = 0.06  # +6% за высокую бету
    BETA_MEDIUM_PENALTY: float = 0.03  # +3% за бету выше медианы
    BETA_LOW_BONUS: float = -0.03  # -3% за низкую бету

    DEBT_HIGH_PENALTY: float = 0.05  # +5% за высокий долг
    DEBT_MEDIUM_PENALTY: float = 0.02  # +2% за долг выше медианы

    OVERVALUED_PENALTY: float = 0.05  # +5% за переоцененность

    MIN_RISK: float = 0.155  # 8% - минимальный риск
    MAX_RISK: float = 0.45  # 45% - максимальный риск


# ==================== КОНСТАНТЫ МОДЕЛИ ====================


@dataclass(frozen=True)
class ModelConstants:
    """Константы для модели дерева решений"""

    # Параметры модели
    MAX_DEPTH: int = 4
    MIN_SAMPLES_SPLIT: int = 10
    MIN_SAMPLES_LEAF: int = 5
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42

    # Пороги для целевой переменной
    PE_STRONG_BUY_THRESHOLD: float = 5.0
    PE_BUY_THRESHOLD: float = 10.0
    PE_SELL_THRESHOLD: float = 20.0

    PB_STRONG_BUY_THRESHOLD: float = 1.0
    PB_BUY_THRESHOLD: float = 1.5
    PB_SELL_THRESHOLD: float = 3.0


@dataclass(frozen=True)
class TargetMapping:
    """Маппинг целевых значений"""

    STRONG_UNDERVALUED: int = 0
    UNDERVALUED: int = 1
    FAIR_VALUE: int = 2
    OVERVALUED: int = 3

    LABELS: Dict[int, str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "LABELS",
            {
                self.STRONG_UNDERVALUED: "Сильно недооценена",
                self.UNDERVALUED: "Недооценена",
                self.FAIR_VALUE: "Справедливая оценка",
                self.OVERVALUED: "Переоценена",
            },
        )


# ==================== КОНСТАНТЫ ПОРТФЕЛЯ ====================


@dataclass(frozen=True)
class PortfolioConstants:
    """Константы для оптимизации портфеля"""

    # Ограничения на веса
    MIN_WEIGHT: float = 0.01  # 1% - минимальная доля
    MAX_WEIGHT: float = 0.20  # 20% - максимальная доля
    MAX_WEIGHT_LOOSE: float = 0.25  # 25% - расширенный лимит

    # Корреляции
    INTRASECTOR_CORRELATION: float = 0.5  # Корреляция внутри сектора
    INTERSECTOR_CORRELATION: float = 0.25  # Корреляция между секторами

    # Коэффициенты для комбинированного портфеля
    SHARPE_PORTFOLIO_WEIGHT: float = 0.7  # 70% - портфель с макс. Шарпом
    MIN_RISK_PORTFOLIO_WEIGHT: float = 0.3  # 30% - портфель с мин. риском

    # Количество портфелей для границы эффективности
    N_EFFICIENT_PORTFOLIOS: int = 2000

    # Пороги для отбора кандидатов
    MIN_MARKET_CAP: float = 5e9  # 5 млрд ₽
    MIN_MARKET_CAP_LOOSE: float = 3e9  # 3 млрд ₽ (расширенный)
    MIN_EXPECTED_RETURN: float = 0.08  # 8%
    MIN_EXPECTED_RETURN_LOOSE: float = 0.06  # 6% (расширенный)
    MAX_RISK: float = 0.40  # 40%

    # Максимальное количество кандидатов
    MAX_CANDIDATES: int = 15

    # Топ позиций для отображения
    TOP_POSITIONS_N: int = 10
    TOP_PIE_N: int = 8
    TOP_RECOMMENDATIONS_N: int = 3

    # Порог для аннотаций на графике
    ANNOTATION_WEIGHT_THRESHOLD: float = 0.02  # 2%

    # Стоп-лосс и тейк-профит
    STOP_LOSS_THRESHOLD: float = -0.15  # -15%
    TAKE_PROFIT_THRESHOLD: float = 0.30  # +30%


# ==================== КОНСТАНТЫ ФАЙЛОВ ====================


@dataclass(frozen=True)
class FileConstants:
    """Константы для файлов и путей"""

    # Имена файлов
    DECISION_TREE_FILE: str = "decision_tree.png"
    EFFICIENT_FRONTIER_FILE: str = "efficient_frontier.png"
    INVEST_PORTFOLIO_REPORT: str = "investment_portfolio_report.xlsx"
    OPTIMAL_PORTFOLIO_FILE: str = "optimal_portfolio.png"
    PORTFOLIO_SUMMARY_FILE: str = "portfolio_summary.png"

    # Названия листов в Excel
    SHEET_PORTFOLIO: str = "Портфель"
    SHEET_SECTORS: str = "Сектора"
    SHEET_METRICS: str = "Метрики"
    SHEET_BENCHMARKS: str = "Бенчмарки"

    # Размеры графиков
    FIGURE_SIZE_TREE: Tuple[int, int] = (20, 12)
    FIGURE_SIZE_FRONTIER: Tuple[int, int] = (12, 8)
    FIGURE_SIZE_SUMMARY: Tuple[int, int] = (16, 12)

    DPI: int = 300


# ==================== КОНСТАНТЫ ФУНДАМЕНТАЛЬНОГО АНАЛИЗА ====================


@dataclass(frozen=True)
class SectorKeywords:
    """Ключевые слова для определения секторов"""

    BANKS: Tuple[str, ...] = ("банк", "сбер", "втб", "мтс", "мкб", "совкомбанк")
    OIL_GAS: Tuple[str, ...] = (
        "нефть",
        "газ",
        "газпром",
        "лукойл",
        "роснефть",
        "татнефть",
    )
    METALS: Tuple[str, ...] = (
        "металл",
        "золото",
        "норильск",
        "северсталь",
        "нлмк",
        "мкк",
        "полюс",
    )
    ENERGY: Tuple[str, ...] = (
        "энерго",
        "росcети",
        "тгк",
        "мосэнерго",
        "интер рао",
        "юнипро",
    )
    TELECOM: Tuple[str, ...] = ("связь", "телеком", "ростелеком", "мтс")
    RETAIL: Tuple[str, ...] = ("ритейл", "магнит", "лента", "пятерочка", "фикс прайс")
    CHEMICAL: Tuple[str, ...] = ("хим", "фосагро", "акрон", "азот")
    IT: Tuple[str, ...] = ("техно", "софт", "яндекс", "циан", "вк", "ast", "софтлайн")

    DEFAULT: str = "Другие"


@dataclass(frozen=True)
class SectorNames:
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


# ==================== КОНСТАНТЫ ФОРМАТИРОВАНИЯ ====================


@dataclass(frozen=True)
class FormattingConstants:
    """Константы для форматирования вывода"""

    # Форматы чисел для отображения (готовые строки)
    PERCENT_FORMAT: str = "{:.1%}"  # для format()
    PERCENT_FORMAT_2D: str = "{:.2%}"  # для format()
    FLOAT_FORMAT_1D: str = "{:.1f}"  # для format()
    FLOAT_FORMAT_2D: str = "{:.2f}"  # для format()
    BILLIONS_FORMAT: str = "{:.1f} млрд ₽"  # для format()

    # Форматы для matplotlib (специальные строки с %)
    MATPLOTLIB_PERCENT: str = "%1.1f%%"  # для autopct в pie chart
    MATPLOTLIB_PERCENT_2D: str = "%1.2f%%"  # для autopct с 2 знаками

    # Строки для отображения N/A
    NA_STRING: str = "N/A"

    # Разделители
    SEPARATOR: str = "=" * 80
    SUB_SEPARATOR: str = "-" * 40

    # Цвета для графиков (HEX)
    COLOR_PORTFOLIO_MARKER: str = "#4682b4"
    COLOR_PORTFOLIO_BG: str = "#f0f8ff"
    COLOR_OPTIMAL_MARKER: str = "red"
    COLOR_SECTOR_CMAP: str = "Paired"
    COLOR_RISK_RETURN_CMAP: str = "viridis"
    COLOR_PIE_CMAP: str = "Set3"

    # Размеры шрифтов
    TITLE_FONT_SIZE: int = 16
    SUBTITLE_FONT_SIZE: int = 14
    AXIS_FONT_SIZE: int = 12
    LABEL_FONT_SIZE: int = 11
    ANNOTATION_FONT_SIZE: int = 9
    TREE_FONT_SIZE: int = 10


# ==================== КОНСТАНТЫ ДЛЯ КОНВЕРТАЦИИ ====================


@dataclass(frozen=True)
class ConversionConstants:
    """Константы для конвертации величин"""

    BILLION: int = 1_000_000_000
    MILLION: int = 1_000_000

    # Регулярные выражения
    BILLION_PATTERN: str = "млрд"
    MILLION_PATTERN: str = "млн"

    # Разделители в числах
    THOUSAND_SEPARATOR: str = " "
    DECIMAL_SEPARATOR: str = ","


# ==================== КОНСТАНТЫ ДЛЯ ОТЧЕТОВ ====================


@dataclass(frozen=True)
class ReportConstants:
    """Константы для генерации отчетов"""

    # Названия колонок в отчете
    COLUMN_TICKER: str = "Тикер"
    COLUMN_NAME: str = "Название"
    COLUMN_SECTOR: str = "Сектор"
    COLUMN_WEIGHT: str = "Вес"
    COLUMN_EXPECTED_RETURN: str = "Ожид. доходность"
    COLUMN_RISK: str = "Риск"
    COLUMN_PE: str = "P/E"
    COLUMN_PB: str = "P/B"
    COLUMN_ROE: str = "ROE"
    COLUMN_DIV_YIELD: str = "Див. доходность"
    COLUMN_RATING: str = "Оценка"
    COLUMN_CONFIDENCE: str = "Уверенность"

    # Названия метрик
    METRIC_EXPECTED_RETURN: str = "Ожидаемая доходность"
    METRIC_RISK: str = "Риск (волатильность)"
    METRIC_SHARPE: str = "Коэффициент Шарпа"
    METRIC_DIVERSIFICATION: str = "Индекс диверсификации"
    METRIC_N_POSITIONS: str = "Количество позиций"
    METRIC_MAX_WEIGHT: str = "Максимальная доля"
    METRIC_MIN_WEIGHT: str = "Минимальная доля ( >1% )"

    # Названия бенчмарков
    BENCHMARK_PE: str = "Медианный P/E"
    BENCHMARK_PB: str = "Медианный P/B"
    BENCHMARK_PS: str = "Медианный P/S"
    BENCHMARK_ROE: str = "Медианный ROE"
    BENCHMARK_DIV_YIELD: str = "Медианная дивидендная доходность"
    BENCHMARK_DEBT: str = "Медианная долг/капитал"
    BENCHMARK_BETA: str = "Медианная бета"


# Создаем экземпляры констант для удобного импорта
FINANCIAL = FinancialConstants()
VALUATION_SCORES = ValuationScores()
RETURN_PREMIUMS = ReturnPremiums()
RISK_PREMIUMS = RiskPremiums()
MODEL_CONSTANTS = ModelConstants()
TARGET_MAPPING = TargetMapping()
PORTFOLIO_CONSTANTS = PortfolioConstants()
FILE_CONSTANTS = FileConstants()
SECTOR_KEYWORDS = SectorKeywords()
SECTOR_NAMES = SectorNames()
FORMATTING = FormattingConstants()
CONVERSION = ConversionConstants()
REPORT = ReportConstants()
