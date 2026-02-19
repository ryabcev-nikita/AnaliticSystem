from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from bonds_models.bonds_constants import (
    CURRENCY_PARAMS,
    INFLATION_RATE,
    RISK_FREE_RATE,
    TAX_RATE,
)


@dataclass
class Bond:
    """Класс для хранения информации об облигации"""

    ticker: str
    name: str
    sector: str
    currency: str
    maturity_date: date
    nominal: float
    risk_level: int
    floating_coupon: bool
    coupon_rate: float

    # Расчетные параметры
    years_to_maturity: float = 0
    current_yield: float = 0
    yield_to_maturity: float = 0
    modified_duration: float = 0
    convexity: float = 0
    credit_spread: float = 0
    liquidity_score: float = 0
    tax_equivalent_yield: float = 0
    real_yield: float = 0

    # Скоринговые параметры
    total_score: float = 0
    yield_score: float = 0
    risk_score: float = 0
    liquidity_score_norm: float = 0
    duration_score: float = 0
    sector_score: float = 0
    currency_score: float = 0

    def calculate_metrics(self, base_rate: float = RISK_FREE_RATE):
        """Расчет метрик облигации"""
        # Дней до погашения
        today = datetime.now().date()
        days_to_maturity = (self.maturity_date - today).days
        self.years_to_maturity = max(days_to_maturity / 365, 0.1)

        # Текущая доходность
        if self.nominal > 0:
            self.current_yield = self.coupon_rate / 100

        # Доходность к погашению (упрощенная)
        if self.years_to_maturity > 0:
            if self.floating_coupon:
                # Для флоатеров: текущая ставка + спред
                self.yield_to_maturity = base_rate + (self.coupon_rate / 100)
            else:
                # Для фиксированных: приближение
                self.yield_to_maturity = self.current_yield

        # Кредитный спред
        self.credit_spread = (
            self.yield_to_maturity - CURRENCY_PARAMS[self.currency]["risk_free_rate"]
        )

        # Дюрация (упрощенная)
        if not self.floating_coupon:
            self.modified_duration = self.years_to_maturity / (
                1 + self.yield_to_maturity
            )
        else:
            self.modified_duration = 0.1  # Флоатеры имеют низкую дюрацию

        # Выпуклость (упрощенная)
        self.convexity = self.years_to_maturity**2 / 100

        # Ликвидность (на основе номинала и риск-уровня)
        self.liquidity_score = min(self.nominal / 1000, 1) * (1 - self.risk_level * 0.2)

        # Налоговый эквивалент
        self.tax_equivalent_yield = self.current_yield / (1 - TAX_RATE)

        # Реальная доходность
        self.real_yield = self.current_yield - INFLATION_RATE
