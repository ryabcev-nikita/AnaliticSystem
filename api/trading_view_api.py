import requests
import pandas as pd
import json

class TradingViewResourceAnalyzer:
    """Получение цен на ресурсы через TradingView"""
    
    def __init__(self):
        self.resource_symbols = {
            'Нефть Brent': 'UKOIL',
            'Нефть WTI': 'USOIL',
            'Золото': 'XAUUSD',
            'Серебро': 'XAGUSD',
            'Медь': 'COPPER',
            'Алюминий': 'ALUMINUM',
            'Палладий': 'XPDUSD',
            'Платина': 'XPTUSD',
            'Газ природный': 'NATURALGAS',
            'Пшеница': 'WHEAT',
            'Кукуруза': 'CORN',
            'Соя': 'SOYBEAN',
            'Сахар': 'SUGAR',
            'Кофе': 'COFFEE',
            'Какао': 'COCOA',
            'Хлопок': 'COTTON',
            'Уголь': 'COAL',
            'Никель': 'NICKEL',
            'Олово': 'TIN',
            'Свинец': 'LEAD',
            'Цинк': 'ZINC'
        }
    
    def get_prices_from_tradingview(self):
        """Получение цен через TradingView API"""
        resource_data = {}
        
        for resource_name, symbol in self.resource_symbols.items():
            try:
                # TradingView API endpoint
                url = f"https://scanner.tradingview.com/global/scan"
                
                payload = {
                    "symbols": {
                        "tickers": [f"OANDA:{symbol}"]  # Для Forex commodities
                    },
                    "columns": ["close", "change", "change_abs", "volume"]
                }
                
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['data']:
                        close_price = data['data'][0]['d'][0]
                        change = data['data'][0]['d'][1]
                        
                        resource_data[resource_name] = {
                            'Цена': close_price,
                            'Изменение_%': change,
                            'Символ': symbol
                        }
                
            except Exception as e:
                print(f"Ошибка для {resource_name}: {e}")
                continue
        
        return pd.DataFrame(resource_data).T

tradingview = TradingViewResourceAnalyzer()
result = tradingview.get_prices_from_tradingview()
print(result)