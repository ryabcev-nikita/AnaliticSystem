import requests
from transformers import pipeline
import re
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class GeopoliticalNewsAnalyzer:
    """Анализ геополитических новостей и их влияния на компании"""
    
    def __init__(self):
        # Инициализация NLP моделей
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Геополитические категории
        self.geopolitical_categories = [
            "санкции", "торговые войны", "валютные риски", 
            "политическая нестабильность", "регуляторные изменения",
            "международные конфликты", "энергетическая безопасность",
            "кибербезопасность", "экологические риски", "транспортные ограничения"
        ]
        
        # Ключевые страны и регионы
        self.key_regions = {
            'Россия': ['росси', 'рф', 'москв', 'кремл'],
            'США': ['сша', 'америк', 'вашингтон'],
            'ЕС': ['евросоюз', 'европ', 'брюссел'],
            'Китай': ['кита', 'пекин'],
            'Ближний Восток': ['ближний восток', 'сауд', 'иран', 'сири'],
            'Украина': ['украин', 'киев'],
        }
        
        # Источники новостей (RSS или API)
        self.news_sources = {
            'РБК': 'https://rssexport.rbc.ru/rbcnews/news/30/full.rss',
            'Интерфакс': 'https://www.interfax.ru/rss.asp',
            'Ведомости': 'https://www.vedomosti.ru/rss/news',
            'Коммерсант': 'https://www.kommersant.ru/RSS/news.xml',
            'ТАСС': 'https://tass.ru/rss/v2.xml'
        }
    
    def fetch_geopolitical_news(self, days_back=7):
        """Сбор геополитических новостей за последние дни"""
        news_items = []
        
        # В реальном проекте здесь будет парсинг новостей
        # Для демонстрации создаем синтетические данные
        synthetic_news = [
            {
                'title': 'Новые санкции против российских компаний',
                'summary': 'Западные страны вводят дополнительные ограничения',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'РБК',
                'category': 'санкции'
            },
            {
                'title': 'Изменение цен на нефть на мировых рынках',
                'summary': 'Цены на нефть выросли на 5% после решения ОПЕК+',
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'source': 'Интерфакс',
                'category': 'энергетическая безопасность'
            },
            {
                'title': 'Китай увеличивает импорт российского газа',
                'summary': 'Подписаны новые долгосрочные контракты',
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'source': 'ТАСС',
                'category': 'торговые войны'
            }
        ]
        
        return synthetic_news
    
    def analyze_news_sentiment(self, news_items):
        """Анализ тональности и категоризация новостей"""
        analyzed_news = []
        
        for news in news_items:
            text = f"{news['title']} {news['summary']}"
            
            # Анализ тональности
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Определение категории с помощью zero-shot classification
            try:
                classification = self.classifier(
                    text, 
                    candidate_labels=self.geopolitical_categories,
                    multi_label=True
                )
                categories = classification['labels'][:3]  # Топ-3 категории
                scores = classification['scores'][:3]
            except:
                categories = ['неопределено']
                scores = [0.5]
            
            # Определение затронутых регионов
            affected_regions = []
            for region, keywords in self.key_regions.items():
                if any(keyword in text.lower() for keyword in keywords):
                    affected_regions.append(region)
            
            analyzed_news.append({
                **news,
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_compound': sentiment['compound'],
                'categories': categories,
                'category_scores': scores,
                'affected_regions': affected_regions,
                'impact_score': self.calculate_impact_score(sentiment['compound'], len(affected_regions))
            })
        
        return analyzed_news
    
    def calculate_impact_score(self, sentiment, regions_count):
        """Расчет оценки влияния новости"""
        # Более негативные новости имеют большее влияние
        impact = (1 - sentiment) * 50  # sentiment от -1 до 1
        
        # Учет количества затронутых регионов
        impact *= (1 + regions_count * 0.2)
        
        return min(100, max(0, impact))
    
    def map_companies_to_geopolitical_risks(self, df_companies, analyzed_news):
        """Сопоставление компаний с геополитическими рисками"""
        company_risk_scores = {}
        
        # Ключевые слова для определения уязвимости компаний
        risk_keywords = {
            'экспорт': ['экспорт', 'export', 'поставк', 'отгрузк'],
            'импорт': ['импорт', 'import', 'закупк', 'ввоз'],
            'санкции': ['санкци', 'ограничен', 'запрет'],
            'валютный': ['валют', 'курс', 'доллар', 'евро'],
            'энергетический': ['энерг', 'электр', 'топлив'],
            'транспортный': ['транспорт', 'логист', 'доставк'],
            'технологический': ['технолог', 'софт', 'оборудован']
        }
        
        for idx, row in df_companies.iterrows():
            company_name = str(row.get('Название', '')).lower()
            company_industry = self.detect_industry(company_name)
            
            # Анализ новостей, связанных с компанией
            company_related_news = []
            total_impact = 0
            
            for news in analyzed_news:
                news_text = f"{news['title']} {news['summary']}".lower()
                
                # Проверяем упоминание компании или отрасли
                if (company_name.split()[0].lower() in news_text or 
                    any(keyword in news_text for keyword in company_industry.split())):
                    
                    company_related_news.append(news)
                    total_impact += news['impact_score']
            
            # Определение уровня риска
            risk_level = 'Низкий'
            risk_score = 0
            
            if company_related_news:
                avg_impact = total_impact / len(company_related_news)
                risk_score = min(100, avg_impact)
                
                if risk_score > 70:
                    risk_level = 'Критический'
                elif risk_score > 50:
                    risk_level = 'Высокий'
                elif risk_score > 30:
                    risk_level = 'Средний'
                else:
                    risk_level = 'Низкий'
            
            company_risk_scores[idx] = {
                'geopolitical_risk_score': risk_score,
                'geopolitical_risk_level': risk_level,
                'related_news_count': len(company_related_news),
                'primary_risks': list(set([cat for news in company_related_news for cat in news['categories']])),
                'affected_regions': list(set([region for news in company_related_news for region in news['affected_regions']]))
            }
        
        return company_risk_scores
    
    def detect_industry(self, company_name):
        """Определение отрасли компании по названию"""
        industry_keywords = {
            'нефтегаз': ['нефт', 'газ', 'нефтегаз'],
            'металлургия': ['метал', 'сталь', 'алюмин', 'медь'],
            'финансы': ['банк', 'страх', 'финанс'],
            'энергетика': ['энерг', 'электр'],
            'транспорт': ['транспорт', 'аэро', 'жд'],
            'телеком': ['телеком', 'связь'],
            'ритейл': ['магнит', 'лента', 'x5'],
            'технологии': ['технолог', 'софт', 'ит']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in company_name for keyword in keywords):
                return industry
        
        return 'другое'