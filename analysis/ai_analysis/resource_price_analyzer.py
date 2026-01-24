import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

class ResourcePriceAnalyzer:
    """Анализ влияния цен на ресурсы на акции компаний"""
    
    def __init__(self):
        self.resource_tickers = {
            'Нефть Brent': 'BZ=F',
            'Нефть WTI': 'CL=F',
            'Золото': 'GC=F',
            'Серебро': 'SI=F',
            'Медь': 'HG=F',
            'Алюминий': 'ALI=F',
            'Никель': 'NICKELUSD',
            'Палладий': 'PA=F',
            'Платина': 'PL=F',
            'Уголь': 'MTF=F',
            'Газ природный': 'NG=F',
            'Пшеница': 'ZW=F',
            'Кукуруза': 'ZC=F',
            'Соя': 'ZS=F'
        }
        
        self.resource_sectors = {
            'Нефтегаз': ['Нефть Brent', 'Нефть WTI', 'Газ природный'],
            'Металлургия': ['Золото', 'Серебро', 'Медь', 'Алюминий', 'Никель', 'Палладий', 'Платина'],
            'Энергетика': ['Уголь', 'Газ природный'],
            'Сельхоз': ['Пшеница', 'Кукуруза', 'Соя']
        }
    
    def get_resource_prices(self, period='6mo'):
        """Получение текущих цен на ресурсы"""
        resource_data = {}
        
        for resource_name, ticker in self.resource_tickers.items():
            try:
                resource = yf.Ticker(ticker)
                hist = resource.history(period=period)
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_change_1d = ((current_price / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
                    price_change_1m = ((current_price / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else 0
                    price_change_3m = ((current_price / hist['Close'].iloc[-63]) - 1) * 100 if len(hist) > 63 else 0
                    
                    resource_data[resource_name] = {
                        'Цена': current_price,
                        'Изменение_1д': price_change_1d,
                        'Изменение_1м': price_change_1m,
                        'Изменение_3м': price_change_3m,
                        'Волатильность': hist['Close'].pct_change().std() * 100
                    }
            except Exception as e:
                print(f"Ошибка при получении данных для {resource_name}: {e}")
                continue
        
        return pd.DataFrame(resource_data).T
    
    def map_companies_to_resources(self, df_companies):
        """Сопоставление компаний с ресурсами"""
        company_resource_map = {}
        
        # Словарь ключевых слов для определения зависимости
        resource_keywords = {
            'Нефть Brent': ['нефт', 'газ', 'нефте', 'нефтян', 'нефтегаз'],
            'Нефть WTI': ['нефт', 'газ', 'нефте'],
            'Золото': ['золот', 'gold', 'аур', 'драж'],
            'Серебро': ['серебр', 'silver', 'аргент'],
            'Медь': ['мед', 'copper', 'купрум'],
            'Алюминий': ['алюмин', 'alum', 'боксит'],
            'Никель': ['никел', 'nickel'],
            'Палладий': ['паллад', 'pallad'],
            'Платина': ['платин', 'platinum'],
            'Уголь': ['уголь', 'coal', 'кокс'],
            'Газ природный': ['газ', 'газов', 'природный газ'],
            'Пшеница': ['зерн', 'пшениц', 'агро', 'сельхоз'],
            'Кукуруза': ['зерн', 'кукур', 'агро'],
            'Соя': ['соя', 'маслич', 'агро']
        }
        
        for idx, row in df_companies.iterrows():
            company_name = str(row.get('Название', '')).lower()
            company_resources = []
            
            for resource, keywords in resource_keywords.items():
                if any(keyword in company_name for keyword in keywords):
                    company_resources.append(resource)
            
            # Особые случаи для известных компаний
            special_cases = {
                'роснефть': ['Нефть Brent', 'Нефть WTI'],
                'лукойл': ['Нефть Brent', 'Нефть WTI'],
                'газпром': ['Газ природный'],
                'норильск': ['Никель', 'Медь', 'Палладий'],
                'полюс': ['Золото'],
                'алроса': ['Алмазы'],
                'фосагро': ['Фосфаты'],
                'уралкалий': ['Калий'],
                'акрон': ['Азот']
            }
            
            for case, resources in special_cases.items():
                if case in company_name:
                    company_resources.extend(resources)
            
            company_resource_map[idx] = list(set(company_resources))
        
        return company_resource_map
    
    def calculate_resource_sensitivity(self, df_companies, resource_prices):
        """Расчет чувствительности компаний к ценам на ресурсы"""
        
        company_resource_map = self.map_companies_to_resources(df_companies)
        sensitivity_scores = {}
        
        for company_idx, resources in company_resource_map.items():
            if not resources:
                sensitivity_scores[company_idx] = {
                    'resource_sensitivity_score': 0,
                    'avg_resource_change_1m': 0,
                    'resource_exposure_count': 0,
                    'primary_resource': None
                }
                continue
            
            # Рассчитываем среднее изменение цен на связанные ресурсы
            resource_changes = []
            for resource in resources:
                if resource in resource_prices.index:
                    change = resource_prices.loc[resource, 'Изменение_1м']
                    resource_changes.append(change)
            
            if resource_changes:
                avg_change = np.mean(resource_changes)
                exposure_count = len(resources)
                
                # Определяем основной ресурс (с максимальным изменением)
                max_change_idx = np.argmax([abs(c) for c in resource_changes])
                primary_resource = resources[max_change_idx] if resources else None
                
                # Оценка чувствительности (0-100)
                sensitivity_score = min(100, max(0, 50 + avg_change * 2))
                
                sensitivity_scores[company_idx] = {
                    'resource_sensitivity_score': sensitivity_score,
                    'avg_resource_change_1m': avg_change,
                    'resource_exposure_count': exposure_count,
                    'primary_resource': primary_resource,
                    'resource_changes': resource_changes
                }
            else:
                sensitivity_scores[company_idx] = {
                    'resource_sensitivity_score': 50,  # нейтральная
                    'avg_resource_change_1m': 0,
                    'resource_exposure_count': 0,
                    'primary_resource': None
                }
        
        return sensitivity_scores