from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Bond:
    """Модель данных облигации"""

    ticker: str
    name: str
    sector: str = ""
    currency: str = ""
    floating_coupon_flag: Optional[bool] = None
    amortization_flag: Optional[bool] = None
    perpetual_flag: Optional[bool] = None
    maturity_date: Optional[str] = None
    nominal: float = 0.0
    risk_level: str = ""
    coupon_rate: Optional[float] = None
    instrument_id: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Bond":
        """Создает объект Bond из ответа API"""
        return cls(
            ticker=data.get("ticker", ""),
            name=data.get("name", ""),
            sector=data.get("sector", ""),
            currency=data.get("currency", ""),
            floating_coupon_flag=data.get("floating_coupon_flag"),
            amortization_flag=data.get("amortization_flag"),
            perpetual_flag=data.get("perpetual_flag"),
            maturity_date=cls._parse_date(data.get("maturity_date", "")),
            nominal=cls._parse_nominal(data.get("nominal", "")),
            risk_level=data.get("risk_level", ""),
            instrument_id=data.get("figi") or data.get("uid"),
        )

    @staticmethod
    def _parse_nominal(nominal_str: str) -> float:
        """Парсит номинал из строки"""
        try:
            content = nominal_str.replace("MoneyValue(", "").rstrip(")")
            parts = content.split(", ")
            units = 0
            nano = 0
            for part in parts:
                if part.startswith("units"):
                    units = int(part.split("=")[1])
                elif part.startswith("nano"):
                    nano = int(part.split("=")[1])
            return units + nano / 1e9
        except:
            return 0.0

    @staticmethod
    def _parse_date(date_str: str) -> Optional[str]:
        """Парсит дату в формат YYYY-MM-DD"""
        if not date_str or date_str == "1970-01-01 00:00:00+00:00":
            return None
        try:
            if "+" in date_str:
                date_str = date_str.split("+")[0]
            dt = datetime.fromisoformat(date_str)
            return dt.replace(tzinfo=None).strftime("%Y-%m-%d")
        except:
            return None
