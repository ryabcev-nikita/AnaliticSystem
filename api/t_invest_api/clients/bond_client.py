import json
import time
from typing import List, Optional
from datetime import datetime

import pandas as pd
from t_tech.invest import Client
from t_tech.invest.schemas import (
    GetBondEventsRequest,
    InstrumentExchangeType,
)
from t_tech.invest.exceptions import RequestError

from constants.config import (
    TOKEN,
    BONDS_JSON_PATH,
    CHUNK_SIZE,
    DELAY_BETWEEN_CHUNKS,
    DELAY_BETWEEN_REQUESTS,
)
from models.bond import Bond
from utils.utils import APIUtils, ExcelFormatter


class BondsClient:
    """Клиент для работы с облигациями"""

    def __init__(self, token: str = TOKEN):
        self.token = token
        self.api_utils = APIUtils()

    def get_bonds_list(self) -> List[Bond]:
        """Получает список облигаций"""
        print("📊 Загрузка списка облигаций...")

        with Client(self.token) as client:
            response = client.instruments.bonds(
                instrument_exchange=InstrumentExchangeType.INSTRUMENT_EXCHANGE_UNSPECIFIED
            )

            # Сохраняем в JSON
            data_to_save = {
                "source": "T-Bank API",
                "generated_at": datetime.now().isoformat(),
                "instruments": [
                    {
                        "figi": instrument.figi,
                        "uid": instrument.uid,
                        "ticker": instrument.ticker,
                        "name": instrument.name,
                        "sector": getattr(instrument, "sector", ""),
                        "currency": instrument.currency,
                        "floating_coupon_flag": getattr(
                            instrument, "floating_coupon_flag", None
                        ),
                        "amortization_flag": getattr(
                            instrument, "amortization_flag", None
                        ),
                        "perpetual_flag": getattr(instrument, "perpetual_flag", None),
                        "maturity_date": str(getattr(instrument, "maturity_date", "")),
                        "nominal": str(getattr(instrument, "nominal", "")),
                        "risk_level": getattr(instrument, "risk_level", ""),
                    }
                    for instrument in response.instruments
                ],
            }

            with open(BONDS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=str)

            print(f"✅ Загружено {len(response.instruments)} облигаций")

            # Конвертируем в объекты Bond
            bonds = []
            for instrument_data in data_to_save["instruments"]:
                bond = Bond.from_api_response(instrument_data)
                if bond.instrument_id:  # Только с валидным ID
                    bonds.append(bond)

            return bonds

    def _get_coupon_rate(self, client, instrument_id: str) -> Optional[float]:
        """Получает ставку купона по инструменту"""
        try:
            request = GetBondEventsRequest(instrument_id=instrument_id)
            response = client.instruments.get_bond_events(request=request)

            if response.events:
                for event in response.events:
                    if event.coupon_interest_rate:
                        rate = self.api_utils.quotation_to_float(
                            event.coupon_interest_rate
                        )
                        if rate > 0:
                            return round(rate, 2)
            return None

        except RequestError:
            return None
        except Exception:
            return None

    def enrich_with_coupon_rates(self, bonds: List[Bond]) -> List[Bond]:
        """Обогащает данные облигаций ставками купонов"""
        print(f"\n🎯 Запрашиваем ставки купонов для {len(bonds)} облигаций")
        print(f"📦 Чанк: {CHUNK_SIZE} шт, интервал: {DELAY_BETWEEN_CHUNKS} сек")

        chunks = [bonds[i : i + CHUNK_SIZE] for i in range(0, len(bonds), CHUNK_SIZE)]
        success_count = 0

        with Client(self.token) as client:
            for chunk_idx, chunk in enumerate(chunks, 1):
                print(f"\n🔄 Чанк {chunk_idx}/{len(chunks)} ({len(chunk)} шт)")

                for item_idx, bond in enumerate(chunk, 1):
                    ticker_short = (
                        bond.ticker[:12] + ".."
                        if len(bond.ticker) > 12
                        else bond.ticker
                    )
                    name_short = (
                        bond.name[:25] + ".." if len(bond.name) > 25 else bond.name
                    )

                    print(
                        f"   [{item_idx:2d}/{len(chunk)}] {ticker_short:12} — {name_short:25}",
                        end=" ",
                    )

                    rate = self.api_utils.retry_on_rate_limit(
                        self._get_coupon_rate, client, bond.instrument_id
                    )

                    if rate:
                        bond.coupon_rate = rate
                        success_count += 1
                        print(f"✅ {rate}%")
                    else:
                        bond.coupon_rate = 0
                        print(f"❌")

                    time.sleep(DELAY_BETWEEN_REQUESTS)

                if chunk_idx < len(chunks):
                    print(f"   ⏳ Ожидание {DELAY_BETWEEN_CHUNKS} сек...")
                    time.sleep(DELAY_BETWEEN_CHUNKS)

        print(
            f"\n✅ Найдено ставок: {success_count}/{len(bonds)} ({success_count/len(bonds)*100:.1f}%)"
        )
        return bonds

    def save_to_excel(self, bonds: List[Bond], filepath: str):
        """Сохраняет данные облигаций в Excel"""
        data = [
            {
                "ticker": bond.ticker,
                "name": bond.name,
                "sector": bond.sector,
                "currency": bond.currency,
                "floating_coupon_flag": bond.floating_coupon_flag,
                "amortization_flag": bond.amortization_flag,
                "perpetual_flag": bond.perpetual_flag,
                "maturity_date": bond.maturity_date,
                "nominal": bond.nominal,
                "risk_level": bond.risk_level,
                "coupon_rate": bond.coupon_rate,
            }
            for bond in bonds
        ]

        df = pd.DataFrame(data)
        if "maturity_date" in df.columns:
            df = df.sort_values("maturity_date")

        ExcelFormatter.save_to_excel(data, filepath)

        # Выводим статистику
        self._print_statistics(df)

    def _print_statistics(self, df: pd.DataFrame):
        """Выводит статистику по облигациям"""
        rates_df = df[df["coupon_rate"].notna()]

        print(f"\n📊 Статистика:")
        print(f"   Всего облигаций: {len(df)}")

        if not rates_df.empty:
            print(
                f"   Найдено ставок: {len(rates_df)} ({len(rates_df)/len(df)*100:.1f}%)"
            )
            print(f"   Средняя ставка: {rates_df['coupon_rate'].mean():.2f}%")
            print(f"   Мин ставка: {rates_df['coupon_rate'].min():.2f}%")
            print(f"   Макс ставка: {rates_df['coupon_rate'].max():.2f}%")

            # Топ-5 секторов по доходности
            sector_avg = (
                rates_df.groupby("sector")["coupon_rate"]
                .agg(["mean", "count"])
                .round(2)
            )
            sector_avg = (
                sector_avg[sector_avg["count"] >= 5]
                .sort_values("mean", ascending=False)
                .head(5)
            )

            if not sector_avg.empty:
                print(f"\n🏆 Топ-5 секторов по доходности:")
                for sector, row in sector_avg.iterrows():
                    print(f"   {sector:15}: {row['mean']}% ({row['count']} шт)")
