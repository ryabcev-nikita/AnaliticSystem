# mport sys
from cx_Freeze import setup, Executable


setup(
    name="AnalyticSystem",
    version="1.0.1",
    description="Аналитическая система фондового рынка на основе методов предиктивной аналитики",
    executables=[
        Executable(
            "main.py",
            icon="icon.png",  # Иконка для exe файла
            target_name="AnalyticSystem.exe",  # Имя выходного файла
        )
    ],
)
