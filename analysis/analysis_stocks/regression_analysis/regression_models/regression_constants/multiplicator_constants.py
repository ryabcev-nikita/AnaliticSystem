"""
Константы для регрессионного анализа акций
"""

# Настройки путей
import os


def find_root_dir(marker="main.py"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, marker)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Достигли корня файловой системы
            return None
        current_dir = parent_dir


ROOT_DIR = find_root_dir()

DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "regression_analysis")
PATHS = {
    "input_file": DATA_DIR + "/fundamentals_shares.xlsx",
    "output_dir": RESULTS_DIR,
}

# Настройки обработки данных
DATA_PROCESSING = {
    "numeric_columns": [
        "PE",
        "PBV",
        "PS",
        "EV_EBITDA",
        "ROE",
        "ROA",
        "ROIC",
        "NPM",
        "Revenue",
        "Net_Income",
        "Market_Cap",
        "Dividend_per_share",
        "EPS",
    ],
    "clip_lower_bounds": {
        "PE": 0.1,
        "PBV": 0.1,
        "PS": 0.01,
        "EV_EBITDA": 0.1,
        "PEG": 0.1,
        "g": 0.001,
    },
    "outlier_threshold": 2.5,  # уменьшили порог для более агрессивного удаления выбросов
    "payout_ratio_clip": (0, 1),
    "min_data_points": 30,  # минимальное количество точек для регрессии
}

# Колонки для анализа
ANALYSIS_COLUMNS = {
    "correlation": [
        "PE",
        "PBV",
        "PS",
        "EV_EBITDA",
        "ROE",
        "ROA",
        "ROIC",
        "NPM",
        "g",
        "PEG",
    ],
    "regression_pe": ["PE", "g"],
    "regression_pbv": ["PBV", "ROE"],
    "regression_ps": ["PS", "NPM"],
    "regression_peg": ["PEG", "g"],
    "regression_ev": ["EV_EBITDA", "g", "ROIC"],
}

# Веса для композитного скора
COMPOSITE_SCORE_WEIGHTS = {
    "PE_ratio": 0.3,
    "PBV_ratio": 0.25,
    "PS_ratio": 0.2,
    "EV_ratio": 0.25,
}

# Настройки портфеля
PORTFOLIO_CONFIG = {
    "n_selected_stocks": 20,
    "n_portfolio_stocks": 10,
    "n_portfolios_simulation": 10000,
    "risk_free_rate": 0.0,
}

# Настройки визуализации
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "small_figure": (12, 5),
    "dpi": 300,
    "correlation_figure": (12, 10),
    "diagnostics_figure": (12, 10),
}

# Переименование колонок
COLUMN_MAPPING = {
    "P/E": "PE",
    "P/B": "PBV",
    "P/S": "PS",
    "EV/EBITDA": "EV_EBITDA",
    "ROE": "ROE",
    "ROA": "ROA",
    "ROIC": "ROIC",
    "NPM": "NPM",
    "Выручка": "Revenue",
    "Чистая прибыль": "Net_Income",
    "Рыночная капитализация": "Market_Cap",
    "Дивиденд на акцию": "Dividend_per_share",
    "EPS": "EPS",
}

# Названия файлов для сохранения
OUTPUT_FILES = {
    "correlation_matrix": "correlation_matrix.png",
    "pe_scatter": "pe_vs_g_scatter.png",
    "pbv_scatter": "pbv_vs_roe_scatter.png",
    "ps_scatter": "ps_vs_npm_scatter.png",
    "peg_scatter": "peg_vs_g_scatter.png",
    "pe_diagnostics": "pe_regression_diagnostics.png",
    "pe_robust_diagnostics": "pe_robust_regression_diagnostics.png",
    "efficient_frontier": "markowitz_efficient_frontier.png",
    "selected_stocks": "selected_stocks.csv",
    "optimal_portfolio": "optimal_portfolio.csv",
    "regression_results": "regression_results.txt",
    "boxplots_before_after": "boxplots_before_after_outliers.png",
}

# Настройки робастной регрессии
ROBUST_REGRESSION = {
    "max_iter": 1000,
    "tolerance": 1e-6,
    "scale_estimator": "mad",  # можно также ' HuberScale'
}
