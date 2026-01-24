import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
from bs4 import BeautifulSoup
import warnings
import time
import json
import re
warnings.filterwarnings('ignore')
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Скачиваем необходимые ресурсы для nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

class GeopoliticalNewsAnalyzer:
    """Анализ геополитических новостей и их влияния на акции Мосбиржи"""
    
    def __init__(self):
        # Инициализация NLP моделей
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            self.classifier = pipeline("zero-shot-classification", 
                                     model="facebook/bart-large-mnli")
        except:
            print("Используется локальная модель для классификации")
            self.classifier = None
        
        # Геополитические категории
        self.geopolitical_categories = [
            "санкции", "торговые войны", "валютные риски", 
            "политическая нестабильность", "регуляторные изменения",
            "международные конфликты", "энергетическая безопасность",
            "кибербезопасность", "экологические риски", "транспортные ограничения",
            "нефть и газ", "финансовые рынки", "сельскохозяйственные риски"
        ]
        
        # Ключевые страны и регионы
        self.key_regions = {
            'Россия': ['росси', 'рф', 'москв', 'кремл', 'путин'],
            'США': ['сша', 'америк', 'вашингтон', 'байден'],
            'ЕС': ['евросоюз', 'европ', 'брюссел', 'германи', 'франц'],
            'Китай': ['кита', 'пекин', 'си цзиньпин'],
            'Ближний Восток': ['ближний восток', 'сауд', 'иран', 'сири', 'израил'],
            'Украина': ['украин', 'киев', 'зеленск'],
            'Великобритания': ['великобритан', 'лондон'],
            'Турция': ['турц', 'эрдоган'],
            'Индия': ['инди', 'моди']
        }
        
        # Основные акции индекса Мосбиржи
        self.moex_companies = {
            'SBER': {'name': 'Сбербанк', 'ticker': 'SBER', 'industry': 'финансы'},
            'GAZP': {'name': 'Газпром', 'ticker': 'GAZP', 'industry': 'нефтегаз'},
            'LKOH': {'name': 'Лукойл', 'ticker': 'LKOH', 'industry': 'нефтегаз'},
            'GMKN': {'name': 'Норникель', 'ticker': 'GMKN', 'industry': 'металлургия'},
            'ROSN': {'name': 'Роснефть', 'ticker': 'ROSN', 'industry': 'нефтегаз'},
            'MGNT': {'name': 'Магнит', 'ticker': 'MGNT', 'industry': 'ритейл'},
            'YNDX': {'name': 'Яндекс', 'ticker': 'YNDX', 'industry': 'технологии'},
            'VTBR': {'name': 'Банк ВТБ', 'ticker': 'VTBR', 'industry': 'финансы'},
            'ALRS': {'name': 'АЛРОСА', 'ticker': 'ALRS', 'industry': 'металлургия'},
            'PLZL': {'name': 'Полюс', 'ticker': 'PLZL', 'industry': 'металлургия'},
            'NVTK': {'name': 'Новатэк', 'ticker': 'NVTK', 'industry': 'нефтегаз'},
            'TATN': {'name': 'Татнефть', 'ticker': 'TATN', 'industry': 'нефтегаз'},
            'MOEX': {'name': 'Московская Биржа', 'ticker': 'MOEX', 'industry': 'финансы'},
            'AFKS': {'name': 'АФК Система', 'ticker': 'AFKS', 'industry': 'конгломерат'},
            'PHOR': {'name': 'ФосАгро', 'ticker': 'PHOR', 'industry': 'химия'},
            'RUAL': {'name': 'РУСАЛ', 'ticker': 'RUAL', 'industry': 'металлургия'},
            'MTSS': {'name': 'МТС', 'ticker': 'MTSS', 'industry': 'телеком'},
            'AFLT': {'name': 'Аэрофлот', 'ticker': 'AFLT', 'industry': 'транспорт'},
            'IRAO': {'name': 'Интер РАО', 'ticker': 'IRAO', 'industry': 'энергетика'},
            'RTKM': {'name': 'Ростелеком', 'ticker': 'RTKM', 'industry': 'телеком'}
        }
        
        # Ключевые слова для компаний
        self.company_keywords = {
            'SBER': ['сбер', 'сбербанк', 'герман греф', 'сб'],
            'GAZP': ['газпром', 'газ', 'миллер', 'гп'],
            'LKOH': ['лукойл', 'нефть', 'алекперов', 'лк'],
            'GMKN': ['норникель', 'никель', 'потанин', 'нн'],
            'ROSN': ['роснефть', 'сечин', 'рн'],
            'YNDX': ['яндекс', 'поиск', 'аркадий войц', 'я'],
            'VTBR': ['втб', 'банк втб', 'костин'],
            'NVTK': ['новатэк', 'гпз', 'михеельсон'],
            'MGNT': ['магнит', 'ритейл', 'галузица'],
            'TATN': ['татнефть', 'татарстан', 'нефть']
        }
        
        # Источники для парсинга финансовых данных
        self.finance_sources = {
            'moex': 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json',
            'investing': 'https://ru.investing.com/equities/',
            'finam': 'https://www.finam.ru/profile/mosbirzha-akcii/',
            'bcs': 'https://bcs-express.ru/kotirovki-i-grafiki/'
        }
        
    def fetch_real_news(self, days_back=7):
        """Получение реальных новостей с RSS-лент"""
        news_items = []
        
        try:
            # Список RSS-источников
            rss_sources = [
                {
                    'name': 'РБК',
                    'url': 'https://rssexport.rbc.ru/rbcnews/news/30/full.rss',
                    'parser': 'rbc'
                },
                {
                    'name': 'Интерфакс',
                    'url': 'https://www.interfax.ru/rss.asp',
                    'parser': 'interfax'
                },
                {
                    'name': 'Ведомости',
                    'url': 'https://www.vedomosti.ru/rss/news',
                    'parser': 'vedomosti'
                },
                {
                    'name': 'Коммерсант',
                    'url': 'https://www.kommersant.ru/RSS/news.xml',
                    'parser': 'kommersant'
                },
                {
                    'name': 'ТАСС',
                    'url': 'https://tass.ru/rss/v2.xml',
                    'parser': 'tass'
                }
            ]
            
            for source in rss_sources:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(source['url'], headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'xml')
                        items = soup.find_all('item')
                        
                        for item in items[:15]:  # Ограничиваем количество
                            title = item.title.text.strip() if item.title else ''
                            description = item.description.text.strip() if item.description else ''
                            pub_date = item.pubDate.text if item.pubDate else ''
                            
                            # Фильтруем по геополитическим темам
                            text = f"{title} {description}".lower()
                            geo_keywords = ['санкци', 'нефть', 'газ', 'рубл', 'доллар', 
                                          'войн', 'конфликт', 'экономик', 'рынок', 'акци',
                                          'бирж', 'инвест', 'фонд', 'банк', 'финанс']
                            
                            if any(keyword in text for keyword in geo_keywords):
                                news_items.append({
                                    'title': title[:200],
                                    'summary': description[:300],
                                    'date': pub_date[:20],
                                    'source': source['name'],
                                    'raw_text': f"{title} {description}"
                                })
                    
                    time.sleep(1)  # Задержка между запросами
                    
                except Exception as e:
                    print(f"Ошибка при парсинге {source['name']}: {str(e)[:50]}")
                    continue
                    
        except Exception as e:
            print(f"Ошибка получения новостей: {e}")
        
        # Если не получили новости, используем синтетические
        if len(news_items) < 5:
            print("Используем синтетические новости для анализа")
            news_items.extend(self.generate_synthetic_news(days_back))
        
        return news_items[:20]  # Ограничиваем 20 новостями
    
    def generate_synthetic_news(self, days_back=7):
        """Генерация синтетических новостей для тестирования"""
        synthetic_news = []
        
        news_templates = [
            {
                'title': 'Новые санкции США против российских банков могут ударить по Сбербанку и ВТБ',
                'summary': 'Администрация США рассматривает новые ограничительные меры против крупнейших российских финансовых институтов, что может повлиять на их котировки',
                'category': 'санкции',
                'impact': 0.8,
                'companies': ['SBER', 'VTBR', 'MOEX']
            },
            {
                'title': 'Цены на нефть Brent превысили $85 за баррель на фоне решения ОПЕК+',
                'summary': 'Сокращение добычи странами ОПЕК+ и геополитическая напряженность привели к росту цен на нефть на 5%, что позитивно для Газпрома и Лукойла',
                'category': 'энергетическая безопасность',
                'impact': 0.7,
                'companies': ['GAZP', 'LKOH', 'ROSN', 'NVTK', 'TATN']
            },
            {
                'title': 'ЕЦБ повысил ключевую ставку до 4.5%, рубль ослабевает',
                'summary': 'Европейский центральный банк продолжает борьбу с инфляцией, ужесточая денежно-кредитную политику, что оказывает давление на рубль',
                'category': 'валютные риски',
                'impact': 0.6,
                'companies': ['SBER', 'VTBR', 'MOEX']
            },
            {
                'title': 'Китай подписал новые контракты на поставку российского СПГ',
                'summary': 'Китайские компании заключили долгосрочные соглашения на поставку сжиженного природного газа из России',
                'category': 'торговые войны',
                'impact': 0.5,
                'companies': ['GAZP', 'NVTK']
            },
            {
                'title': 'Минфин может изменить дивидендную политику для госкомпаний',
                'summary': 'Рассматривается возможность пересмотра нормативов по выплате дивидендов компаниями с государственным участием',
                'category': 'регуляторные изменения',
                'impact': 0.7,
                'companies': ['GAZP', 'ROSN', 'TATN', 'ALRS']
            },
            {
                'title': 'Усиление контроля за трансграничными операциями бизнеса',
                'summary': 'Центральный банк и Росфинмониторинг вводят дополнительные проверки для экспортно-импортных сделок',
                'category': 'регуляторные изменения',
                'impact': 0.6,
                'companies': ['GMKN', 'ALRS', 'PHOR', 'RUAL']
            },
            {
                'title': 'Яндекс столкнулся с новыми регуляторными требованиями',
                'summary': 'Компания должна адаптировать свои сервисы под новые правила цифрового рынка',
                'category': 'регуляторные изменения',
                'impact': 0.5,
                'companies': ['YNDX']
            },
            {
                'title': 'Мировые цены на никель выросли на 10% за неделю',
                'summary': 'Дефицит предложения и увеличение спроса со стороны электромобильной отрасли подняли цены на никель',
                'category': 'торговые войны',
                'impact': 0.6,
                'companies': ['GMKN', 'NVTK']
            }
        ]
        
        for i, template in enumerate(news_templates):
            synthetic_news.append({
                'title': template['title'],
                'summary': template['summary'],
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'source': 'Синтетические данные',
                'raw_text': f"{template['title']} {template['summary']}",
                'template': template
            })
        
        return synthetic_news
    
    def analyze_news_sentiment(self, news_items):
        """Анализ тональности и категоризация новостей"""
        analyzed_news = []
        
        print(f"Анализируем {len(news_items)} новостей...")
        
        for news in news_items:
            text = news['raw_text']
            
            # Анализ тональности
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Определение категории (упрощенная версия если нет модели)
            categories = []
            scores = []
            
            if self.classifier:
                try:
                    classification = self.classifier(
                        text, 
                        candidate_labels=self.geopolitical_categories,
                        multi_label=True
                    )
                    categories = classification['labels'][:2]
                    scores = classification['scores'][:2]
                except:
                    categories = self.simple_category_detection(text)
                    scores = [0.7, 0.5]
            else:
                categories = self.simple_category_detection(text)
                scores = [0.7, 0.5] if categories else [0.5]
            
            # Определение затронутых регионов
            affected_regions = []
            for region, keywords in self.key_regions.items():
                if any(keyword in text.lower() for keyword in keywords):
                    affected_regions.append(region)
            
            # Определение потенциально затронутых компаний
            affected_companies = self.detect_affected_companies(text)
            
            # Если есть шаблон, берем компании из него
            if 'template' in news:
                affected_companies = list(set(affected_companies + news['template'].get('companies', [])))
            
            analyzed_news.append({
                **news,
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_compound': sentiment['compound'],
                'categories': categories[:2],
                'category_scores': scores[:2],
                'affected_regions': affected_regions,
                'affected_companies': affected_companies,
                'impact_score': self.calculate_news_impact_score(
                    sentiment['compound'], 
                    len(affected_regions),
                    len(affected_companies)
                )
            })
        
        return analyzed_news
    
    def simple_category_detection(self, text):
        """Упрощенное определение категорий по ключевым словам"""
        text_lower = text.lower()
        categories = []
        
        category_keywords = {
            'санкции': ['санкци', 'ограничен', 'запрет'],
            'нефть и газ': ['нефть', 'газ', 'энерг', 'баррел'],
            'валютные риски': ['валют', 'рубл', 'доллар', 'курс'],
            'торговые войны': ['торгов', 'импорт', 'экспорт', 'пошлин'],
            'регуляторные изменения': ['регулятор', 'закон', 'норматив', 'требован'],
            'финансовые рынки': ['бирж', 'акци', 'фонд', 'инвест']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories[:2] if categories else ['неопределено']
    
    def detect_affected_companies(self, text):
        """Определение затронутых компаний по ключевым словам"""
        text_lower = text.lower()
        affected = []
        
        for ticker, keywords in self.company_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                affected.append(ticker)
        
        # Дополнительная проверка по отраслям
        industry_keywords = {
            'SBER': ['банк', 'финанс', 'кредит'],
            'GAZP': ['газ', 'трубопровод', 'энерг'],
            'LKOH': ['нефт', 'бензин', 'заправк'],
            'YNDX': ['интернет', 'поиск', 'такси'],
            'MGNT': ['магазин', 'продукт', 'сеть']
        }
        
        for ticker, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords) and ticker not in affected:
                # Добавляем с меньшим приоритетом
                affected.append(ticker)
        
        return list(set(affected))
    
    def calculate_news_impact_score(self, sentiment, regions_count, companies_count):
        """Расчет оценки влияния новости"""
        # Более негативные новости имеют большее влияние
        base_impact = (1 - sentiment) * 50  # sentiment от -1 до 1
        
        # Учет количества затронутых регионов
        region_multiplier = 1 + (regions_count * 0.15)
        
        # Учет количества затронутых компаний
        company_multiplier = 1 + (companies_count * 0.1)
        
        impact = base_impact * region_multiplier * company_multiplier
        
        return min(100, max(0, round(impact, 1)))
    
    def analyze_company_exposure(self, analyzed_news):
        """Анализ подверженности компаний геополитическим рискам"""
        company_exposure = {ticker: {
            'name': info['name'],
            'industry': info['industry'],
            'total_impact': 0,
            'news_count': 0,
            'average_impact': 0,
            'max_impact': 0,
            'risk_categories': [],
            'affected_regions': [],
            'related_news': []
        } for ticker, info in self.moex_companies.items()}
        
        for news in analyzed_news:
            impact = news['impact_score']
            
            # Компании из списка затронутых
            for ticker in news.get('affected_companies', []):
                if ticker in company_exposure:
                    company_exposure[ticker]['total_impact'] += impact
                    company_exposure[ticker]['news_count'] += 1
                    company_exposure[ticker]['max_impact'] = max(
                        company_exposure[ticker]['max_impact'], 
                        impact
                    )
                    
                    # Добавляем категории рисков
                    for category in news['categories']:
                        if category not in company_exposure[ticker]['risk_categories']:
                            company_exposure[ticker]['risk_categories'].append(category)
                    
                    # Добавляем регионы
                    for region in news['affected_regions']:
                        if region not in company_exposure[ticker]['affected_regions']:
                            company_exposure[ticker]['affected_regions'].append(region)
                    
                    # Сохраняем связанные новости
                    company_exposure[ticker]['related_news'].append({
                        'title': news['title'][:100],
                        'impact': impact,
                        'sentiment': news['sentiment_compound'],
                        'date': news['date'],
                        'categories': news['categories']
                    })
            
            # Проверка по отраслевым ключевым словам
            for ticker, info in self.moex_companies.items():
                if ticker in news.get('affected_companies', []):
                    continue  # Уже учли выше
                    
                industry = info['industry']
                
                # Проверяем, связана ли новость с отраслью компании
                industry_keywords = {
                    'нефтегаз': ['нефт', 'газ', 'энерг', 'опек', 'баррел', 'топлив'],
                    'металлургия': ['метал', 'стал', 'никел', 'алюмин', 'золот', 'мед'],
                    'финансы': ['банк', 'финанс', 'ставк', 'рубл', 'валют', 'кредит'],
                    'ритейл': ['потреб', 'ритейл', 'продаж', 'товар', 'магазин'],
                    'технологии': ['технолог', 'цифр', 'софт', 'интернет', 'it'],
                    'транспорт': ['транспорт', 'логист', 'доставк', 'авиа', 'перевоз'],
                    'телеком': ['связь', 'телеком', 'интернет', 'мобильн'],
                    'химия': ['хими', 'удобрен', 'фосфат']
                }
                
                if industry in industry_keywords:
                    if any(keyword in news['raw_text'].lower() 
                           for keyword in industry_keywords[industry]):
                        # Добавляем с меньшим весом
                        company_exposure[ticker]['total_impact'] += impact * 0.5
                        company_exposure[ticker]['news_count'] += 0.5
        
        # Рассчитываем среднее воздействие
        for ticker in company_exposure:
            if company_exposure[ticker]['news_count'] > 0:
                company_exposure[ticker]['average_impact'] = (
                    company_exposure[ticker]['total_impact'] / 
                    company_exposure[ticker]['news_count']
                )
        
        return company_exposure
    
    def parse_stock_data_moex(self, tickers):
        """Парсинг данных с Московской биржи (MOEX API)"""
        price_data = {}
        
        try:
            # MOEX ISS API
            for ticker in tickers:
                try:
                    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Извлекаем данные о рыночной цене
                        marketdata = data.get('marketdata', {}).get('data', [])
                        securities = data.get('securities', {}).get('data', [])
                        
                        if marketdata and securities:
                            # Текущая цена
                            last_price = marketdata[0][12] if marketdata[0][12] else marketdata[0][4]
                            
                            # Изменение
                            change = marketdata[0][13] if marketdata[0][13] else 0
                            change_percent = marketdata[0][14] if marketdata[0][14] else 0
                            
                            # Объем торгов
                            volume = marketdata[0][9] if marketdata[0][9] else 0
                            
                            price_data[ticker] = {
                                'price': float(last_price) if last_price else 0,
                                'change': float(change) if change else 0,
                                'change_percent': float(change_percent) if change_percent else 0,
                                'volume': int(volume) if volume else 0,
                                'source': 'MOEX'
                            }
                    
                    time.sleep(0.5)  # Задержка между запросами
                    
                except Exception as e:
                    print(f"Ошибка парсинга {ticker}: {str(e)[:50]}")
                    continue
                    
        except Exception as e:
            print(f"Ошибка парсинга MOEX: {e}")
        
        # Если не получили данные с MOEX, используем фиктивные
        if not price_data:
            price_data = self.generate_mock_price_data(tickers)
        
        return price_data
    
    def parse_stock_data_investing(self, tickers):
        """Парсинг данных с Investing.com (альтернативный источник)"""
        price_data = {}
        
        try:
            # Маппинг тикеров для investing.com
            ticker_map = {
                'SBER': 'sberbank',
                'GAZP': 'gazprom',
                'LKOH': 'lukoil',
                'GMKN': 'mmk-norilsk-nickel',
                'ROSN': 'rosneft'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for ticker in tickers[:5]:  # Ограничиваем количество запросов
                if ticker in ticker_map:
                    try:
                        url = f"https://ru.investing.com/equities/{ticker_map[ticker]}"
                        response = requests.get(url, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            # Поиск цены (структура может меняться)
                            price_elem = soup.find('span', {'data-test': 'instrument-price-last'})
                            change_elem = soup.find('span', {'data-test': 'instrument-price-change'})
                            
                            if price_elem:
                                price_text = price_elem.text.replace(',', '.')
                                price = float(re.search(r'[\d.]+', price_text).group())
                                
                                change = 0
                                if change_elem:
                                    change_text = change_elem.text.replace(',', '.')
                                    change_match = re.search(r'([+-]?[\d.]+)', change_text)
                                    if change_match:
                                        change = float(change_match.group(1))
                                
                                price_data[ticker] = {
                                    'price': price,
                                    'change': change,
                                    'change_percent': round((change / price * 100), 2) if price > 0 else 0,
                                    'volume': 0,
                                    'source': 'Investing.com'
                                }
                        
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Ошибка парсинга {ticker}: {str(e)[:50]}")
                        continue
                        
        except Exception as e:
            print(f"Ошибка парсинга Investing.com: {e}")
        
        return price_data
    
    def generate_mock_price_data(self, tickers):
        """Генерация тестовых данных о ценах"""
        price_data = {}
        
        base_prices = {
            'SBER': 280.50,
            'GAZP': 165.30,
            'LKOH': 7100.80,
            'GMKN': 16750.40,
            'ROSN': 580.90,
            'MGNT': 5200.75,
            'YNDX': 2850.60,
            'VTBR': 0.0265,
            'ALRS': 79.40,
            'PLZL': 11700.25,
            'NVTK': 1645.80,
            'TATN': 630.45,
            'MOEX': 145.60,
            'AFKS': 17.85,
            'PHOR': 6800.90,
            'RUAL': 41.30,
            'MTSS': 270.45,
            'AFLT': 47.80,
            'IRAO': 2.45,
            'RTKM': 73.20
        }
        
        for ticker in tickers:
            base_price = base_prices.get(ticker, 100.0)
            
            # Генерируем случайное изменение от -10% до +10%
            import random
            change_percent = random.uniform(-10, 10)
            change = base_price * change_percent / 100
            price = base_price + change
            
            # Генерируем объем
            volume = random.randint(1000000, 10000000)
            
            price_data[ticker] = {
                'price': round(price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'volume': volume,
                'source': 'Моковые данные'
            }
        
        return price_data
    
    def get_stock_price_changes(self, tickers, days=5):
        """Получение данных об изменении цен акций"""
        print("Получение данных по акциям...")
        
        # Пытаемся получить реальные данные
        price_data = self.parse_stock_data_moex(tickers)
        
        # Если не получилось, пробуем альтернативный источник
        if len(price_data) < 5:
            investing_data = self.parse_stock_data_investing(tickers)
            price_data.update(investing_data)
        
        # Если все еще мало данных, используем моковые
        if len(price_data) < len(tickers) / 2:
            mock_data = self.generate_mock_price_data(tickers)
            # Обновляем только те тикеры, которых нет
            for ticker in tickers:
                if ticker not in price_data:
                    price_data[ticker] = mock_data.get(ticker, {
                        'price': 0,
                        'change': 0,
                        'change_percent': 0,
                        'volume': 0,
                        'source': 'Нет данных'
                    })
        
        return price_data
    
    def calculate_risk_levels(self, company_exposure):
        """Определение уровней риска для компаний"""
        risk_levels = {}
        
        for ticker, data in company_exposure.items():
            avg_impact = data['average_impact']
            
            # Определение уровня риска
            if avg_impact >= 70:
                risk_level = 'КРИТИЧЕСКИЙ'
                color = '🔴'
                risk_value = 5
            elif avg_impact >= 50:
                risk_level = 'ВЫСОКИЙ'
                color = '🟠'
                risk_value = 4
            elif avg_impact >= 30:
                risk_level = 'СРЕДНИЙ'
                color = '🟡'
                risk_value = 3
            elif avg_impact >= 10:
                risk_level = 'НИЗКИЙ'
                color = '🟢'
                risk_value = 2
            else:
                risk_level = 'МИНИМАЛЬНЫЙ'
                color = '⚪'
                risk_value = 1
            
            risk_levels[ticker] = {
                'Название': data['name'],
                'Отрасль': data['industry'],
                'Уровень риска': f"{color} {risk_level}",
                'Риск (число)': risk_value,
                'Среднее воздействие': round(avg_impact, 1),
                'Макс воздействие': round(data['max_impact'], 1),
                'Кол-во новостей': int(data['news_count']),
                'Основные риски': ', '.join(data['risk_categories'][:3]),
                'Затронутые регионы': ', '.join(data['affected_regions'][:3])
            }
        
        return risk_levels
    
    def generate_report(self, analyzed_news, company_risk_levels, price_changes):
        """Генерация сводного отчета"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_news_analyzed': len(analyzed_news),
            'top_risks': self.get_top_risks(analyzed_news),
            'company_analysis': {},
            'sector_analysis': {},
            'recommendations': []
        }
        
        # Анализ по компаниям
        for ticker, risk_data in company_risk_levels.items():
            company_info = {
                **risk_data,
                'price_data': price_changes.get(ticker, {})
            }
            
            # Генерация рекомендаций
            recommendation = self.generate_recommendation(ticker, risk_data, price_changes.get(ticker, {}))
            if recommendation:
                company_info['recommendation'] = recommendation
            
            report['company_analysis'][ticker] = company_info
        
        # Анализ по секторам
        report['sector_analysis'] = self.analyze_sectors(company_risk_levels)
        
        # Общие рекомендации
        report['recommendations'] = self.generate_general_recommendations(company_risk_levels)
        
        return report
    
    def analyze_sectors(self, company_risk_levels):
        """Анализ рисков по секторам"""
        sector_risks = {}
        
        for ticker, data in company_risk_levels.items():
            sector = data['Отрасль']
            if sector not in sector_risks:
                sector_risks[sector] = {
                    'companies': [],
                    'avg_impact': 0,
                    'max_impact': 0,
                    'total_news': 0
                }
            
            sector_risks[sector]['companies'].append(ticker)
            sector_risks[sector]['avg_impact'] += data['Среднее воздействие']
            sector_risks[sector]['max_impact'] = max(sector_risks[sector]['max_impact'], 
                                                   data['Макс воздействие'])
            sector_risks[sector]['total_news'] += data['Кол-во новостей']
        
        # Рассчитываем средние значения
        for sector in sector_risks:
            count = len(sector_risks[sector]['companies'])
            if count > 0:
                sector_risks[sector]['avg_impact'] = round(sector_risks[sector]['avg_impact'] / count, 1)
        
        return sector_risks
    
    def get_top_risks(self, analyzed_news):
        """Определение главных рисков"""
        risk_counter = {}
        
        for news in analyzed_news:
            for category in news['categories']:
                risk_counter[category] = risk_counter.get(category, 0) + 1
        
        return sorted(risk_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def generate_recommendation(self, ticker, risk_data, price_data):
        """Генерация рекомендации для конкретной акции"""
        avg_impact = risk_data['Среднее воздействие']
        risk_level = risk_data['Уровень риска']
        price_change = price_data.get('change_percent', 0)
        
        recommendations = []
        
        if '🔴' in risk_level:
            if price_change < -5:
                recommendations.append("КРИТИЧЕСКИЙ РИСК - продавать немедленно")
            else:
                recommendations.append("КРИТИЧЕСКИЙ РИСК - избегать покупки")
        elif '🟠' in risk_level:
            if price_change < -3:
                recommendations.append("ВЫСОКИЙ РИСК - сокращать позиции")
            else:
                recommendations.append("ВЫСОКИЙ РИСК - ограничить экспозицию")
        elif '🟡' in risk_level:
            recommendations.append("СРЕДНИЙ РИСК - мониторить ситуацию")
        elif '🟢' in risk_level:
            if price_change < -8:
                recommendations.append("НИЗКИЙ РИСК - возможность для покупки")
            else:
                recommendations.append("НИЗКИЙ РИСК - удерживать позиции")
        elif '⚪' in risk_level:
            recommendations.append("МИНИМАЛЬНЫЙ РИСК - стабильная ситуация")
        
        # Учитываем изменение цены
        if price_change < -10:
            recommendations.append("Сильное падение цены - осторожность")
        elif price_change > 10:
            recommendations.append("Сильный рост - фиксировать прибыль")
        
        return "; ".join(recommendations) if recommendations else "Требуется дополнительный анализ"
    
    def generate_general_recommendations(self, company_risk_levels):
        """Генерация общих рекомендаций"""
        recommendations = []
        
        # Анализ секторов
        sector_analysis = {}
        for ticker, data in company_risk_levels.items():
            sector = data['Отрасль']
            if sector not in sector_analysis:
                sector_analysis[sector] = []
            sector_analysis[sector].append(data['Среднее воздействие'])
        
        # Рекомендации по секторам
        for sector, impacts in sector_analysis.items():
            avg_impact = np.mean(impacts)
            if avg_impact > 60:
                recommendations.append(f"⚠️ Избегать инвестиций в сектор {sector} (средний риск: {avg_impact:.1f})")
            elif avg_impact > 40:
                recommendations.append(f"⚡ Осторожно в секторе {sector} (средний риск: {avg_impact:.1f})")
            elif avg_impact < 20:
                recommendations.append(f"✅ Рассмотреть возможности в секторе {sector} (низкий риск: {avg_impact:.1f})")
        
        # Рекомендации по наиболее рискованным компаниям
        high_risk_companies = [ticker for ticker, data in company_risk_levels.items() 
                              if data['Риск (число)'] >= 4]
        
        if high_risk_companies:
            recommendations.append(f"🚨 Наибольшему риску подвержены: {', '.join(high_risk_companies[:3])}")
        
        # Рекомендации по безопасным компаниям
        safe_companies = [ticker for ticker, data in company_risk_levels.items() 
                         if data['Риск (число)'] <= 2]
        
        if safe_companies:
            recommendations.append(f"🛡️ Наиболее защищены: {', '.join(safe_companies[:3])}")
        
        return recommendations[:10]  # Ограничиваем 10 рекомендациями
    
    def run_analysis(self, use_real_news=True):
        """Запуск полного анализа"""
        print("=" * 60)
        print("ГЕОПОЛИТИЧЕСКИЙ АНАЛИЗ ДЛЯ АКЦИЙ МОСБИРЖИ")
        print("=" * 60)
        
        # 1. Сбор новостей
        print("\n1. Сбор геополитических новостей...")
        if use_real_news:
            news_items = self.fetch_real_news(days_back=7)
        else:
            news_items = self.generate_synthetic_news(days_back=7)
        
        print(f"   Найдено новостей: {len(news_items)}")
        
        # 2. Анализ новостей
        print("2. Анализ тональности и категоризация...")
        analyzed_news = self.analyze_news_sentiment(news_items)
        
        # 3. Анализ подверженности компаний
        print("3. Оценка подверженности компаний рискам...")
        company_exposure = self.analyze_company_exposure(analyzed_news)
        
        # 4. Определение уровней риска
        print("4. Определение уровней риска...")
        company_risk_levels = self.calculate_risk_levels(company_exposure)
        
        # 5. Получение данных по ценам
        print("5. Получение данных по акциям...")
        tickers = list(self.moex_companies.keys())
        price_changes = self.get_stock_price_changes(tickers, days=5)
        
        print(f"   Получены данные для {len(price_changes)} акций")
        
        # 6. Генерация отчета
        print("6. Формирование отчета...")
        report = self.generate_report(analyzed_news, company_risk_levels, price_changes)
        
        print("\n✅ Анализ завершен!")
        return report, analyzed_news, company_risk_levels, price_changes

# Тестовое использование и проверка
def test_geopolitical_analyzer():
    """Тестирование функционала анализатора"""
    
    print("🚀 ЗАПУСК ТЕСТОВОГО АНАЛИЗА")
    print("-" * 40)
    
    # Инициализация анализатора
    analyzer = GeopoliticalNewsAnalyzer()
    
    # Запуск анализа
    report, analyzed_news, risk_levels, price_changes = analyzer.run_analysis(
        use_real_news=False  # Используем синтетические данные для теста
    )
    
    # Вывод результатов
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Проанализировано новостей: {report['total_news_analyzed']}")
    print(f"   Время анализа: {report['timestamp']}")
    
    print(f"\n📈 ТОП-5 ГЕОПОЛИТИЧЕСКИХ РИСКОВ:")
    for i, (risk, count) in enumerate(report['top_risks'], 1):
        print(f"   {i}. {risk}: {count} упоминаний")
    
    print(f"\n🏢 АНАЛИЗ КОМПАНИЙ (ТОП-10 ПО УРОВНЮ РИСКА):")
    
    # Сортируем компании по уровню риска
    sorted_companies = sorted(
        report['company_analysis'].items(),
        key=lambda x: x[1]['Риск (число)'],
        reverse=True
    )[:10]
    
    print(f"\n{'Тикер':<8} {'Название':<20} {'Риск':<15} {'Воздействие':<12} {'Изменение':<10} {'Рекомендация':<30}")
    print("-" * 95)
    
    for ticker, data in sorted_companies:
        price_data = data.get('price_data', {})
        change = price_data.get('change_percent', 0)
        change_str = f"{change:+.1f}%" if change else "N/A"
        
        # Сокращаем рекомендацию для отображения
        rec = data.get('recommendation', '')
        if len(rec) > 28:
            rec = rec[:25] + "..."
        
        print(f"{ticker:<8} {data['Название'][:18]:<20} {data['Уровень риска']:<15} "
              f"{data['Среднее воздействие']:<12.1f} {change_str:<10} {rec:<30}")
    
    print(f"\n🏭 АНАЛИЗ СЕКТОРОВ:")
    for sector, stats in report['sector_analysis'].items():
        print(f"   {sector}: {stats['avg_impact']:.1f} (компаний: {len(stats['companies'])}, "
              f"новостей: {stats['total_news']})")
    
    print(f"\n💡 ОСНОВНЫЕ РЕКОМЕНДАЦИИ:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"   {i}. {rec}")
    
    # Детальный анализ для компании с максимальным риском
    if sorted_companies:
        top_risk_ticker = sorted_companies[0][0]
        top_risk_data = sorted_companies[0][1]
        
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ {top_risk_ticker} ({top_risk_data['Название']}):")
        print(f"   Отрасль: {top_risk_data['Отрасль']}")
        print(f"   Уровень риска: {top_risk_data['Уровень риска']}")
        print(f"   Среднее воздействие: {top_risk_data['Среднее воздействие']}")
        print(f"   Максимальное воздействие: {top_risk_data['Макс воздействие']}")
        print(f"   Основные риски: {top_risk_data['Основные риски']}")
        print(f"   Затронутые регионы: {top_risk_data['Затронутые регионы']}")
        print(f"   Количество связанных новостей: {top_risk_data['Кол-во новостей']}")
        
        if top_risk_data.get('price_data'):
            price_info = top_risk_data['price_data']
            print(f"   Цена: {price_info.get('price', 'N/A')} ({price_info.get('change_percent', 0):+.2f}%)")
            print(f"   Источник данных: {price_info.get('source', 'N/A')}")
        
        print(f"   Рекомендация: {top_risk_data.get('recommendation', 'Нет данных')}")
        
        # Анализ связанных новостей
        company_exposure = analyzer.analyze_company_exposure(analyzed_news)
        related_news = company_exposure[top_risk_ticker]['related_news']
        
        if related_news:
            print(f"\n   📰 СВЯЗАННЫЕ НОВОСТИ:")
            for i, news in enumerate(related_news[:3], 1):
                print(f"   {i}. {news['title']}")
                print(f"      Воздействие: {news['impact']:.1f}, "
                      f"Дата: {news['date']}, "
                      f"Категории: {', '.join(news.get('categories', []))}")
    
    # Сохранение отчета в файл
    try:
        with open('geopolitical_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Отчет сохранен в файл: geopolitical_analysis_report.json")
        
        # Также сохраняем таблицу в CSV
        df_data = []
        for ticker, data in report['company_analysis'].items():
            row = {
                'Тикер': ticker,
                'Название': data['Название'],
                'Отрасль': data['Отрасль'],
                'Уровень риска': data['Уровень риска'],
                'Риск (число)': data['Риск (число)'],
                'Среднее воздействие': data['Среднее воздействие'],
                'Кол-во новостей': data['Кол-во новостей'],
                'Основные риски': data['Основные риски'],
                'Цена': data.get('price_data', {}).get('price', 'N/A'),
                'Изменение %': data.get('price_data', {}).get('change_percent', 'N/A'),
                'Рекомендация': data.get('recommendation', '')
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv('geopolitical_analysis_table.csv', index=False, encoding='utf-8-sig')
        print(f"📊 Таблица сохранена в файл: geopolitical_analysis_table.csv")
        
    except Exception as e:
        print(f"\n⚠️ Ошибка сохранения отчета: {e}")
    
    return report

# Демонстрация работы анализатора
def demonstrate_analyzer():
    """Демонстрация возможностей анализатора"""
    
    analyzer = GeopoliticalNewsAnalyzer()
    
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ГЕОПОЛИТИЧЕСКОГО АНАЛИЗАТОРА")
    print("=" * 60)
    
    # 1. Показать список компаний
    print("\n📋 АНАЛИЗИРУЕМЫЕ КОМПАНИИ:")
    for ticker, info in analyzer.moex_companies.items():
        print(f"  {ticker}: {info['name']} ({info['industry']})")
    
    # 2. Протестировать парсинг новостей
    print("\n📰 ТЕСТ ПАРСИНГА НОВОСТЕЙ:")
    news_items = analyzer.generate_synthetic_news(days_back=3)
    print(f"  Сгенерировано синтетических новостей: {len(news_items)}")
    
    # 3. Протестировать анализ новостей
    print("\n🧠 ТЕСТ АНАЛИЗА НОВОСТЕЙ:")
    analyzed = analyzer.analyze_news_sentiment(news_items[:3])
    for i, news in enumerate(analyzed[:2], 1):
        print(f"  Новость {i}: {news['title'][:50]}...")
        print(f"    Воздействие: {news['impact_score']:.1f}, "
              f"Тональность: {news['sentiment_compound']:.2f}")
        print(f"    Компании: {', '.join(news.get('affected_companies', []))}")
    
    # 4. Протестировать анализ компании
    print("\n🏢 ТЕСТ АНАЛИЗА КОМПАНИИ:")
    company_exposure = analyzer.analyze_company_exposure(analyzed)
    test_ticker = 'SBER'
    if test_ticker in company_exposure:
        data = company_exposure[test_ticker]
        print(f"  {test_ticker}: {data['name']}")
        print(f"    Среднее воздействие: {data['average_impact']:.1f}")
        print(f"    Новостей: {data['news_count']}")
        print(f"    Риски: {', '.join(data['risk_categories'])}")
    
    print("\n✅ Демонстрация завершена. Для полного анализа запустите test_geopolitical_analyzer()")

# Запуск теста
if __name__ == "__main__":
    print("Начало тестирования геополитического анализатора...\n")
    
    try:
        # Демонстрация возможностей
        demonstrate_analyzer()
        
        print("\n" + "="*60)
        
        # Полный тест
        report = test_geopolitical_analyzer()
        print("\n✅ Тестирование завершено успешно!")
        
        # Дополнительная проверка
        print("\n📋 ПРОВЕРКА РЕЗУЛЬТАТОВ:")
        print(f"1. Количество проанализированных компаний: {len(report['company_analysis'])}")
        print(f"2. Количество секторов: {len(report['sector_analysis'])}")
        
        impacts = [d['Среднее воздействие'] for d in report['company_analysis'].values()]
        print(f"3. Диапазон воздействия: {min(impacts):.1f} - {max(impacts):.1f}")
        
        risk_distribution = {}
        for data in report['company_analysis'].values():
            risk = data['Риск (число)']
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        print(f"4. Распределение рисков:")
        for risk_level in sorted(risk_distribution.keys()):
            count = risk_distribution[risk_level]
            print(f"   Уровень {risk_level}: {count} компаний")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()