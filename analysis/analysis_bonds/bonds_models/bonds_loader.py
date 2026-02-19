from datetime import datetime

import pandas as pd

from analysis.analysis_bonds.bonds_models.bonds_models import Bond


class BondsDataLoader:
    """Класс для загрузки и предобработки данных об облигациях"""

    @staticmethod
    def load_bonds_data(file_path: str) -> pd.DataFrame:
        """Загрузка данных из Excel файла"""
        print(f"   Загрузка данных из: {file_path}")

        df = pd.read_excel(file_path, sheet_name=0, header=0)

        # Переименовываем колонки
        df.columns = [
            "ticker",
            "name",
            "sector",
            "currency",
            "maturity_date",
            "nominal",
            "risk_level",
            "floating_coupon",
            "coupon_rate",
        ]

        # Обработка пропусков
        df["sector"] = df["sector"].fillna("other")
        df["currency"] = df["currency"].fillna("rub")
        df["nominal"] = df["nominal"].fillna(1000)
        df["risk_level"] = df["risk_level"].fillna(2)
        df["floating_coupon"] = df["floating_coupon"].fillna(False)
        df["coupon_rate"] = df["coupon_rate"].fillna(0)

        # Конвертация дат
        df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")

        # Удаляем строки без даты погашения
        df = df.dropna(subset=["maturity_date"])

        # Фильтруем только будущие даты погашения
        today = datetime.now()
        df = df[df["maturity_date"] > today]

        return df


class BondsAnalyzer:
    """Класс для анализа облигаций"""

    def __init__(self, bonds_df: pd.DataFrame):
        self.bonds_df = bonds_df
        self.bonds_list = []
        self.process_bonds()

    def process_bonds(self):
        """Обработка облигаций и расчет метрик"""
        for _, row in self.bonds_df.iterrows():
            try:
                bond = Bond(
                    ticker=row["ticker"],
                    name=row["name"],
                    sector=row["sector"],
                    currency=row["currency"],
                    maturity_date=row["maturity_date"].date(),
                    nominal=row["nominal"],
                    risk_level=(
                        int(row["risk_level"]) if not pd.isna(row["risk_level"]) else 2
                    ),
                    floating_coupon=bool(row["floating_coupon"]),
                    coupon_rate=(
                        float(row["coupon_rate"])
                        if not pd.isna(row["coupon_rate"])
                        else 0
                    ),
                )
                bond.calculate_metrics()
                self.bonds_list.append(bond)
            except Exception as e:
                print(f"   Ошибка обработки {row.get('ticker', 'Unknown')}: {e}")
                continue

    def get_bonds_dataframe(self) -> pd.DataFrame:
        """Получение DataFrame с рассчитанными метриками"""
        data = []
        for bond in self.bonds_list:
            data.append(
                {
                    "ticker": bond.ticker,
                    "name": (
                        bond.name[:30] + "..." if len(bond.name) > 30 else bond.name
                    ),
                    "sector": bond.sector,
                    "currency": bond.currency,
                    "years_to_maturity": round(bond.years_to_maturity, 2),
                    "nominal": bond.nominal,
                    "risk_level": bond.risk_level,
                    "coupon_rate": bond.coupon_rate,
                    "current_yield": round(bond.current_yield * 100, 2),
                    "yield_to_maturity": round(bond.yield_to_maturity * 100, 2),
                    "modified_duration": round(bond.modified_duration, 2),
                    "credit_spread": round(bond.credit_spread * 100, 2),
                    "liquidity_score": round(bond.liquidity_score, 2),
                    "floating_coupon": bond.floating_coupon,
                    "maturity_date": bond.maturity_date,
                }
            )

        return pd.DataFrame(data)

    def get_statistics_by_risk(self) -> pd.DataFrame:
        """Статистика по уровням риска"""
        df = self.get_bonds_dataframe()
        stats = (
            df.groupby("risk_level")
            .agg(
                {
                    "current_yield": ["mean", "min", "max", "std"],
                    "modified_duration": ["mean", "min", "max"],
                    "years_to_maturity": "mean",
                    "ticker": "count",
                }
            )
            .round(2)
        )

        stats.columns = [
            "Avg Yield",
            "Min Yield",
            "Max Yield",
            "Yield Std",
            "Avg Duration",
            "Min Duration",
            "Max Duration",
            "Avg Maturity",
            "Count",
        ]
        return stats

    def get_statistics_by_currency(self) -> pd.DataFrame:
        """Статистика по валютам"""
        df = self.get_bonds_dataframe()
        stats = (
            df.groupby("currency")
            .agg(
                {
                    "current_yield": ["mean", "min", "max"],
                    "modified_duration": "mean",
                    "years_to_maturity": "mean",
                    "ticker": "count",
                    "nominal": "sum",
                }
            )
            .round(2)
        )

        stats.columns = [
            "Avg Yield",
            "Min Yield",
            "Max Yield",
            "Avg Duration",
            "Avg Maturity",
            "Count",
            "Total Nominal",
        ]
        return stats
