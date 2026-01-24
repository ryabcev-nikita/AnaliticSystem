import pandas as pd


class MoexResourceAnalyzer:
    """Данные с Московской биржи"""
    
    def __init__(self):
        self.moex_tickers = {
            'Нефть Urals': 'IMOEX',
            'Золото': 'GLDRUB_TOM',
            'Серебро': 'SLVRUB_TOM',
            'Платина': 'PLTNRUB_TOM',
            'Палладий': 'PD RUB_TOD',
            'Никель': 'NICKEL',
            'Медь': 'CU-3.14',
            'Алюминий': 'AL-2.85'
        }
        
        self.moex_base_url = "https://iss.moex.com/iss"
    
    def get_moex_prices(self):
        """Получение данных с MOEX"""
        resource_data = {}
        
        for resource_name, ticker in self.moex_tickers.items():
            try:
                # Для фьючерсов на ресурсы
                url = f"{self.moex_base_url}/engines/futures/markets/forts/securities/{ticker}.json"
                
                response = requests.get(url)
                data = response.json()
                
                # Извлекаем последнюю цену
                if 'marketdata' in data:
                    marketdata = data['marketdata']['data'][0]
                    last_price = marketdata[12]  # LAST цена
                    change = marketdata[13]  # CHANGE
                    
                    resource_data[resource_name] = {
                        'Цена': last_price,
                        'Изменение': change,
                        'Тикер': ticker
                    }
                
            except Exception as e:
                print(f"Ошибка для {resource_name}: {e}")
                continue
        
        return pd.DataFrame(resource_data).T