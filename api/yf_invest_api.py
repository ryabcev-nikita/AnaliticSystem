import os
import yfinance as yf
import pandas as pd
from datetime import datetime


class YF_InvestApi:
    def __init__(self):
        self.commodities = {
            "Gold": "GC=F",
            "Silver": "SI=F",
            "Platinum": "PL=F",
            "Palladium": "PA=F",
            "Copper": "HG=F",
            "Oil_WTI": "CL=F",
            "Oil_Brent": "BZ=F",
            "Natural_gas": "NG=F",
            "Aluminum": "ALI=F",  # Алюминий на LME
        }

        self.crypto = {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD",
            "Litecoin": "LTC-USD",
            "XRP": "XRP-USD",
            "Cardano": "ADA-USD",
        }

        self.forex = {
            "EUR/USD": "EURUSD=X",
            "USD/JPY": "JPY=X",
            "USD/CHF": "CHF=X",
            "USD/CNY": "CNY=X",
            "DXY": "DX-Y.NYB",
        }

        self.indices = {
            # США
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Philadelphia Semiconductor": "^SOX",  # Индекс производителей полупроводников
            # Азия
            "Nikkei 225": "^N225",
            "Hang Seng": "^HSI",
            "Shanghai Composite": "000001.SS",
            "Nifty 50 (Индия)": "^NSEI",
            # Другие рынки
            "Brazil IBOVESPA": "^BVSP",
            "South Africa Top 40": "^JN0U.JO",
            "Saudi Arabia TASI": "^TASI.SR",
        }

        self.market_data_dir = None

    def setup_directories(self, market_data_dir):
        """Создание необходимых директорий"""
        self.market_data_dir = market_data_dir
        # Создаем директорию, если её нет
        os.makedirs(self.market_data_dir, exist_ok=True)
        print(f"Директория для сохранения: {os.path.abspath(self.market_data_dir)}")

    def _download_and_save(self, name, ticker, category=""):
        """Общий метод для загрузки и сохранения данных"""
        try:
            print(f"   {name} ({ticker})...", end="", flush=True)

            # Загружаем данные
            data = yf.download(ticker, period="1y", interval="1d", progress=False)

            if data.empty:
                print(f" ✗ Нет данных")
                return False

            # Обрабатываем мультииндекс
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Формируем имя файла
            clean_name = (
                name.replace("/", "_")
                .replace(" ", "_")
                .replace("&", "and")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
                .replace("__", "_")
            )

            filename = f"{self.market_data_dir}/{clean_name}.csv"
            data.to_csv(filename)

            # Проверяем, создался ли файл
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f" ✓ {len(data)} строк ({file_size} байт)")
                return True
            else:
                print(f" ✗ Файл не создан")
                return False

        except Exception as e:
            print(f" ✗ Ошибка: {e}")
            return False

    def get_forex_exchange(self):
        print("\nЗагрузка валютных пар...")
        print("-" * 50)
        for name, ticker in self.forex.items():
            self._download_and_save(name, ticker, "forex")

    def get_commodities(self):
        print("\nЗагрузка товарных активов...")
        print("-" * 50)
        for name, ticker in self.commodities.items():
            self._download_and_save(name, ticker, "commodities")

    def get_cryptocurrencies(self):
        print("\nЗагрузка криптовалют...")
        print("-" * 50)
        for name, ticker in self.crypto.items():
            self._download_and_save(name, ticker, "crypto")

    def get_indexes(self):
        print("\nЗагрузка фондовых индексов...")
        print("-" * 50)
        for name, ticker in self.indices.items():
            self._download_and_save(name, ticker, "indices")

    def download_all(self):
        """Загрузить все активы"""
        print("=" * 60)
        print(f"НАЧАЛО ЗАГРУЗКИ РЫНОЧНЫХ ДАННЫХ")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        self.get_commodities()
        self.get_cryptocurrencies()
        self.get_indexes()
        self.get_forex_exchange()

        print("\n" + "=" * 60)
        print("ЗАГРУЗКА ЗАВЕРШЕНА!")
        print(f"Файлы сохранены в: {os.path.abspath(self.market_data_dir)}")
        print("=" * 60)

        # Показываем список сохраненных файлов
        if os.path.exists(self.market_data_dir):
            files = [f for f in os.listdir(self.market_data_dir) if f.endswith(".csv")]
            print(f"\nСохранено файлов: {len(files)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(__file__))
    market_data_dir = os.path.join(current_dir, "data", "market_data")

    yf_finance = YF_InvestApi()
    yf_finance.setup_directories(market_data_dir)
    yf_finance.download_all()
