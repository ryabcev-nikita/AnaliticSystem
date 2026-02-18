from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Share:
    """Модель данных акции"""

    uid: str
    asset_uid: str
    figi: str
    ticker: str
    name: str
    lot: int
    currency: str
    sector: str
    class_code: str

    @classmethod
    def from_api_response(cls, instrument) -> "Share":
        """Создает объект Share из ответа API"""
        return cls(
            uid=instrument.uid,
            asset_uid=instrument.asset_uid,
            figi=instrument.figi,
            ticker=instrument.ticker,
            name=instrument.name,
            lot=instrument.lot,
            currency=instrument.currency,
            sector=getattr(instrument, "sector", "Не указан"),
            class_code=instrument.class_code,
        )


@dataclass
class FundamentalData:
    """Модель фундаментальных данных"""

    ticker: str
    name: str
    asset_uid: str
    figi: str
    currency: str
    market_capitalization: Any = None
    total_enterprise_value_mrq: Any = None
    revenue_ttm: Any = None
    net_income_ttm: Any = None
    ebitda_ttm: Any = None
    pe_ratio_ttm: Any = None
    price_to_book_ttm: Any = None
    price_to_sales_ttm: Any = None
    price_to_free_cash_flow_ttm: Any = None
    roe: Any = None
    roa: Any = None
    roic: Any = None
    ev_to_ebitda_mrq: Any = None
    ev_to_sales: Any = None
    free_cash_flow_ttm: Any = None
    five_year_annual_revenue_growth_rate: Any = None
    five_years_average_dividend_yield: Any = None
    five_year_annual_dividend_growth_rate: Any = None
    current_ratio_mrq: Any = None
    dividend_payout_ratio_fy: Any = None
    net_margin_mrq: Any = None
    total_debt_mrq: Any = None
    total_debt_to_equity_mrq: Any = None
    net_debt_to_ebitda: Any = None
    total_debt_to_ebitda_mrq: Any = None
    eps_ttm: Any = None
    dividend_yield_daily_ttm: Any = None
    beta: Any = None
    dividends_per_share: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для DataFrame"""
        return {
            "Тикер": self.ticker,
            "Название": self.name,
            "Asset UID": self.asset_uid,
            "FIGI": self.figi,
            "Валюта": self.currency,
            "Рыночная капитализация": self._format_number(self.market_capitalization),
            "EV": self._format_number(self.total_enterprise_value_mrq),
            "Выручка": self._format_number(self.revenue_ttm),
            "Чистая прибыль": self._format_number(self.net_income_ttm),
            "EBITDA": self._format_number(self.ebitda_ttm),
            "P/E": self.pe_ratio_ttm,
            "P/B": self.price_to_book_ttm,
            "P/S": self.price_to_sales_ttm,
            "P/FCF": self.price_to_free_cash_flow_ttm,
            "ROE": self.roe,
            "ROA": self.roa,
            "ROIC": self.roic,
            "EV/EBITDA": self.ev_to_ebitda_mrq,
            "EV/S": self.ev_to_sales,
            "FCF": self.free_cash_flow_ttm,
            "CAGR_Sales": self.five_year_annual_revenue_growth_rate,
            "Average_dividend_yield": self.five_years_average_dividend_yield,
            "Average_cagr_dividend_yield": self.five_year_annual_dividend_growth_rate,
            "Current_ratio_mr": self.current_ratio_mrq,
            "Payot Ratio": self.dividend_payout_ratio_fy,
            "NPM": self.net_margin_mrq,
            "Debt": self._format_number(self.total_debt_mrq),
            "Debt/Capital": self.total_debt_to_equity_mrq,
            "Net_Debt/EBITDA": self.net_debt_to_ebitda,
            "Debt/EBITDA": self.total_debt_to_ebitda_mrq,
            "EPS": self.eps_ttm,
            "Дивидендная доходность": self.dividend_yield_daily_ttm,
            "Бета": self.beta,
            "Дивиденд на акцию": self.dividends_per_share,
        }

    @staticmethod
    def _format_number(val):
        """Форматирует большие числа"""
        if isinstance(val, (int, float)):
            if val is None or val == 0:
                return ""
            if abs(val) >= 1_000_000_000:
                return f"{val/1_000_000_000:.2f} млрд"
            elif abs(val) >= 1_000_000:
                return f"{val/1_000_000:.2f} млн"
            elif abs(val) >= 1_000:
                return f"{val/1_000:.1f} тыс"
        return val
