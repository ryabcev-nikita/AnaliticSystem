from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": [
        "os", "sys", "logging", "analysis", "api", "shared"
    ],
    "includes": [
        "numpy", "pandas", "matplotlib", "openpyxl", "requests", "yfinance",
        "seaborn", "sklearn", "statsmodels", "tensorflow", "bs4"
    ],
    "include_files": [
        "shared/", "analysis/", "api/"
    ],
    "excludes": [],
}

setup(
    name="AnalyticSystem",
    version="1.0.1",
    description="Аналитическая система",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=None, target_name="AnalyticSystem.exe")]
)