import json
import os
import pandas as pd
from datetime import datetime
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill
import openpyxl

# from shared.config import TOKEN

# Импорты для API Т-Банка
from t_tech.invest import Client
from t_tech.invest.schemas import GetAssetFundamentalsRequest

TOKEN = "t.V4QVXUA5khrTJcMQNsCCDC3IfD94uJA5Yj_FpR8UfaMs3KxSY0tlIlSDe3ix6G7CcKYMbfQTNlLSWR2l1aHQjQ"
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(parent_dir, "data", "fundamentals_shares.xlsx")
os.makedirs(os.path.dirname(file_path), exist_ok=True)

file_path_json = os.path.join(parent_dir, "data", "shares.json")


def get_bonds(token):
    pass


def get_shares(token):
    """Получает список акций через API Т-Банка"""
    try:
        with Client(token) as client:
            print("[Т-Банк] Запрашиваю список акций...")
            instruments_response = client.instruments.shares()

            # Фильтрация (только рублевые)
            filtered_instruments = [
                instrument
                for instrument in instruments_response.instruments
                if instrument.currency.lower() == "rub"
            ]

            print(f"[Т-Банк] Получено рублевых акций: {len(filtered_instruments)}")

            # Подготавливаем данные для сохранения
            data_to_save = {
                "source": "T-Bank API",
                "generated_at": datetime.now().isoformat(),
                "instruments": [
                    {
                        "uid": instrument.uid,
                        "asset_uid": instrument.asset_uid,
                        "figi": instrument.figi,
                        "ticker": instrument.ticker,
                        "name": instrument.name,
                        "lot": instrument.lot,
                        "currency": instrument.currency,
                        "sector": getattr(instrument, "sector", "Не указан"),
                        "class_code": instrument.class_code,
                    }
                    for instrument in filtered_instruments
                ],
            }
            # Сохраняем в JSON
            with open(file_path_json, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=str)

            print(f"✅ Данные сохранены в 'tbank_shares.json'")
            return True

    except Exception as e:
        print(f"❌ Ошибка при получении акций: {e}")
        return False


def get_fundamentals_assets_to_excel(token, excel_filename=file_path):
    """
    Получает фундаментальные показатели акций Т-Банк и сохраняет в Excel
    """
    try:
        # Загружаем список инструментов
        with open(file_path_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        instruments_list = json_data["instruments"]
        asset_uids = [
            inst["asset_uid"] for inst in instruments_list if inst.get("asset_uid")
        ]

        if not asset_uids:
            print("❌ Нет asset_uid для запроса")
            return

        print(f"[Т-Банк] Загружено {len(asset_uids)} активов для запроса")

        all_fundamentals = []
        chunk_size = 30  # Можно регулировать в зависимости от лимитов API

        # Обработка чанками
        for i in range(0, len(asset_uids), chunk_size):
            chunk = asset_uids[i : i + chunk_size]
            instruments_chunk = instruments_list[i : i + chunk_size]

            print(
                f"🔄 Пачка {i//chunk_size + 1}/{(len(asset_uids)-1)//chunk_size + 1} ({len(chunk)} активов)"
            )
            try:
                with Client(token) as client:
                    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: передаем assets как массив
                    request = GetAssetFundamentalsRequest(assets=chunk)
                    response = client.instruments.get_asset_fundamentals(
                        request=request
                    )
                    print(f"   Получено показателей: {len(response.fundamentals)}")

                    # Сопоставляем фундаментальные данные с инструментами
                    for fundamental in response.fundamentals:
                        # Ищем соответствующий инструмент по asset_uid
                        matching_instrument = next(
                            (
                                inst
                                for inst in instruments_chunk
                                if inst.get("asset_uid") == fundamental.asset_uid
                            ),
                            None,
                        )

                        if not matching_instrument:
                            continue

                        # Функция безопасного получения значений
                        def safe_get(field, default=None):
                            try:
                                val = getattr(fundamental, field, default)
                                if val is None:
                                    return default
                                # Обработка дат
                                if field.endswith("_date") and hasattr(
                                    val, "ToDatetime"
                                ):
                                    try:
                                        return val.ToDatetime().strftime("%Y-%m-%d")
                                    except:
                                        return default
                                return val
                            except AttributeError:
                                return default

                        # Форматирование чисел
                        def fmt_num(val):
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

                        # Собираем строку данных
                        row = {
                            "Тикер": matching_instrument.get("ticker", ""),
                            "Название": matching_instrument.get("name", ""),
                            "Asset UID": matching_instrument.get("asset_uid", ""),
                            "FIGI": matching_instrument.get("figi", ""),
                            "Валюта": matching_instrument.get("currency", ""),
                            "Рыночная капитализация": fmt_num(
                                safe_get("market_capitalization")
                            ),
                            "EV": fmt_num(safe_get("total_enterprise_value_mrq")),
                            "Выручка": fmt_num(safe_get("revenue_ttm")),
                            "Чистая прибыль": fmt_num(safe_get("net_income_ttm")),
                            "EBITDA": fmt_num(safe_get("ebitda_ttm")),
                            "P/E": safe_get("pe_ratio_ttm"),
                            "P/B": safe_get("price_to_book_ttm"),
                            "P/S": safe_get("price_to_sales_ttm"),
                            "P/FCF": safe_get("price_to_free_cash_flow_ttm"),
                            "ROE": safe_get("roe"),
                            "ROA": safe_get("roa"),
                            "ROIC": safe_get("roic"),
                            "EV/EBITDA": safe_get("ev_to_ebitda_mrq"),
                            "EV/S": safe_get("ev_to_sales"),
                            "Payot Ratio": safe_get("dividend_payout_ratio_fy"),
                            "NPM": safe_get("net_margin_mrq"),
                            "Debt": safe_get("total_debt_mrq"),
                            "Debt/Capital": safe_get("total_debt_to_equity_mrq"),
                            "Net_Debt/EBITDA": safe_get("net_debt_to_ebitda"),
                            "Debt/EBITDA": safe_get("total_debt_to_ebitda_mrq"),
                            "EPS": safe_get("eps_ttm"),
                            "Дивидендная доходность": safe_get(
                                "dividend_yield_daily_ttm"
                            ),
                            "Свободный денежный поток": fmt_num(
                                safe_get("free_cash_flow_ttm")
                            ),
                            "Бета": safe_get("beta"),
                            "Дивиденд на акцию": safe_get("dividends_per_share"),
                        }
                        all_fundamentals.append(row)

                    print(f"   ✓ Обработано записей: {len(response.fundamentals)}")

            except Exception as e:
                print(f"   ❌ Ошибка в пачке: {e}")
                continue

        # Сохраняем в Excel
        if all_fundamentals:
            print(all_fundamentals)
            save_to_excel(all_fundamentals, excel_filename)
        else:
            print("⚠️ Не получено данных для сохранения")

    except Exception as e:
        print(f"❌ Общая ошибка: {e}")


def save_to_excel(data, filename):
    """Сохраняет данные в Excel с форматированием"""
    try:
        df = pd.DataFrame(data)

        if df.empty:
            print("❌ Нет данных для сохранения")
            return

        print(f"\n💾 Сохраняю {len(df)} записей в '{filename}'...")

        # Сохраняем в Excel
        df.to_excel(filename, index=False, engine="openpyxl")

        # Применяем форматирование
        format_excel_file(filename, df)

        print(f"✅ Данные успешно сохранены!")
        print(f"📊 Статистика: {len(df)} строк, {len(df.columns)} столбцов")

    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


def format_excel_file(filename, df):
    """Форматирует Excel файл"""
    try:
        wb = openpyxl.load_workbook(filename)
        ws = wb.active

        # Стиль заголовков
        header_fill = PatternFill(
            start_color="366092", end_color="366092", fill_type="solid"
        )
        header_font = Font(color="FFFFFF", bold=True)
        header_alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )

        # Форматируем заголовки
        for col in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment

        # Автоподбор ширины столбцов
        for column in ws.columns:
            max_len = 0
            col_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    cell_len = len(str(cell.value))
                    if cell_len > max_len:
                        max_len = cell_len
                except:
                    pass
            adjusted_width = min(max_len + 2, 50)
            ws.column_dimensions[col_letter].width = adjusted_width

        # Добавляем фильтры
        ws.auto_filter.ref = ws.dimensions

        wb.save(filename)
        print("   ✓ Форматирование применено")

    except Exception as e:
        print(f"   ⚠️ Ошибка форматирования: {e}")


# Основной запуск
if __name__ == "__main__":
    print("🚀 Запуск скрипта для работы с API Т-Банк Инвестиции")
    print("=" * 50)

    success = get_shares(TOKEN)
    if success:
        get_fundamentals_assets_to_excel(TOKEN)
