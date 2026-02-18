import json
import time
from typing import List
from datetime import datetime

from t_tech.invest import Client
from t_tech.invest.schemas import (
    GetAssetFundamentalsRequest,
)

from ..models.share import FundamentalData, Share
from ..utils.utils import APIUtils, ExcelFormatter
from ..constants.config import (
    CHUNK_SIZE,
    DELAY_BETWEEN_REQUESTS,
    SHARES_JSON_PATH,
    TARGET_CURRENCY,
    TOKEN,
)


class SharesClient:
    """Клиент для работы с акциями"""

    def __init__(self, token: str = TOKEN):
        self.token = token
        self.api_utils = APIUtils()

    def get_shares_list(self) -> List[Share]:
        """Получает список акций через API"""
        print("\n📊 Загрузка списка акций...")

        with Client(self.token) as client:
            response = client.instruments.shares()

            # Фильтруем по валюте
            filtered_instruments = [
                instrument
                for instrument in response.instruments
                if instrument.currency.lower() == TARGET_CURRENCY
            ]

            print(f"✅ Получено рублевых акций: {len(filtered_instruments)}")

            # Сохраняем в JSON
            shares = []
            data_to_save = {
                "source": "T-Bank API",
                "generated_at": datetime.now().isoformat(),
                "instruments": [],
            }

            for instrument in filtered_instruments:
                share = Share.from_api_response(instrument)
                shares.append(share)

                data_to_save["instruments"].append(
                    {
                        "uid": share.uid,
                        "asset_uid": share.asset_uid,
                        "figi": share.figi,
                        "ticker": share.ticker,
                        "name": share.name,
                        "lot": share.lot,
                        "currency": share.currency,
                        "sector": share.sector,
                        "class_code": share.class_code,
                    }
                )

            with open(SHARES_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=str)

            return shares

    def get_fundamentals(self, shares: List[Share]) -> List[FundamentalData]:
        """Получает фундаментальные показатели акций"""
        print(f"\n📈 Загрузка фундаментальных показателей для {len(shares)} акций")

        asset_uids = [share.asset_uid for share in shares if share.asset_uid]
        if not asset_uids:
            print("❌ Нет asset_uid для запроса")
            return []

        fundamentals_data = []
        chunks = [
            asset_uids[i : i + CHUNK_SIZE]
            for i in range(0, len(asset_uids), CHUNK_SIZE)
        ]
        shares_chunks = [
            shares[i : i + CHUNK_SIZE] for i in range(0, len(shares), CHUNK_SIZE)
        ]

        for chunk_idx, (asset_chunk, shares_chunk) in enumerate(
            zip(chunks, shares_chunks), 1
        ):
            print(f"\n🔄 Чанк {chunk_idx}/{len(chunks)} ({len(asset_chunk)} активов)")

            try:
                with Client(self.token) as client:
                    request = GetAssetFundamentalsRequest(assets=asset_chunk)
                    response = client.instruments.get_asset_fundamentals(
                        request=request
                    )

                    print(f"   Получено показателей: {len(response.fundamentals)}")

                    # Сопоставляем с акциями
                    for fundamental in response.fundamentals:
                        matching_share = next(
                            (
                                s
                                for s in shares_chunk
                                if s.asset_uid == fundamental.asset_uid
                            ),
                            None,
                        )

                        if not matching_share:
                            continue

                        # Создаем объект FundamentalData
                        fund_data = FundamentalData(
                            ticker=matching_share.ticker,
                            name=matching_share.name,
                            asset_uid=matching_share.asset_uid,
                            figi=matching_share.figi,
                            currency=matching_share.currency,
                            market_capitalization=getattr(
                                fundamental, "market_capitalization", None
                            ),
                            total_enterprise_value_mrq=getattr(
                                fundamental, "total_enterprise_value_mrq", None
                            ),
                            revenue_ttm=getattr(fundamental, "revenue_ttm", None),
                            net_income_ttm=getattr(fundamental, "net_income_ttm", None),
                            ebitda_ttm=getattr(fundamental, "ebitda_ttm", None),
                            pe_ratio_ttm=getattr(fundamental, "pe_ratio_ttm", None),
                            price_to_book_ttm=getattr(
                                fundamental, "price_to_book_ttm", None
                            ),
                            price_to_sales_ttm=getattr(
                                fundamental, "price_to_sales_ttm", None
                            ),
                            price_to_free_cash_flow_ttm=getattr(
                                fundamental, "price_to_free_cash_flow_ttm", None
                            ),
                            roe=getattr(fundamental, "roe", None),
                            roa=getattr(fundamental, "roa", None),
                            roic=getattr(fundamental, "roic", None),
                            ev_to_ebitda_mrq=getattr(
                                fundamental, "ev_to_ebitda_mrq", None
                            ),
                            ev_to_sales=getattr(fundamental, "ev_to_sales", None),
                            free_cash_flow_ttm=getattr(
                                fundamental, "free_cash_flow_ttm", None
                            ),
                            five_year_annual_revenue_growth_rate=getattr(
                                fundamental,
                                "five_year_annual_revenue_growth_rate",
                                None,
                            ),
                            five_years_average_dividend_yield=getattr(
                                fundamental, "five_years_average_dividend_yield", None
                            ),
                            five_year_annual_dividend_growth_rate=getattr(
                                fundamental,
                                "five_year_annual_dividend_growth_rate",
                                None,
                            ),
                            current_ratio_mrq=getattr(
                                fundamental, "current_ratio_mrq", None
                            ),
                            dividend_payout_ratio_fy=getattr(
                                fundamental, "dividend_payout_ratio_fy", None
                            ),
                            net_margin_mrq=getattr(fundamental, "net_margin_mrq", None),
                            total_debt_mrq=getattr(fundamental, "total_debt_mrq", None),
                            total_debt_to_equity_mrq=getattr(
                                fundamental, "total_debt_to_equity_mrq", None
                            ),
                            net_debt_to_ebitda=getattr(
                                fundamental, "net_debt_to_ebitda", None
                            ),
                            total_debt_to_ebitda_mrq=getattr(
                                fundamental, "total_debt_to_ebitda_mrq", None
                            ),
                            eps_ttm=getattr(fundamental, "eps_ttm", None),
                            dividend_yield_daily_ttm=getattr(
                                fundamental, "dividend_yield_daily_ttm", None
                            ),
                            beta=getattr(fundamental, "beta", None),
                            dividends_per_share=getattr(
                                fundamental, "dividends_per_share", None
                            ),
                        )
                        fundamentals_data.append(fund_data)

                    time.sleep(DELAY_BETWEEN_REQUESTS)

            except Exception as e:
                print(f"   ❌ Ошибка в пачке: {e}")
                continue

        return fundamentals_data

    def save_fundamentals_to_excel(
        self, fundamentals: List[FundamentalData], filepath: str
    ):
        """Сохраняет фундаментальные данные в Excel"""
        data = [fund.to_dict() for fund in fundamentals]
        ExcelFormatter.save_to_excel(data, filepath)
        print(f"📊 Сохранено {len(fundamentals)} записей")
