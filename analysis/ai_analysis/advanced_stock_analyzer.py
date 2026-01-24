class AdvancedStockAnalyzer:
    """Продвинутый анализатор акций с учетом ресурсов и геополитики"""
    
    def __init__(self):
        self.resource_analyzer = ResourcePriceAnalyzer()
        self.news_analyzer = GeopoliticalNewsAnalyzer()
        
    def analyze_with_external_factors(self, df_companies):
        """Полный анализ с учетом внешних факторов"""
        
        print("="*80)
        print("АНАЛИЗ ВНЕШНИХ ФАКТОРОВ")
        print("="*80)
        
        # 1. Анализ цен на ресурсы
        print("\n1. Анализ цен на ресурсы...")
        resource_prices = self.resource_analyzer.get_resource_prices()
        
        if not resource_prices.empty:
            print(f"Получены данные по {len(resource_prices)} ресурсам")
            print(resource_prices[['Цена', 'Изменение_1м', 'Волатильность']].head())
        
        # 2. Анализ геополитических новостей
        print("\n2. Анализ геополитических новостей...")
        news_items = self.news_analyzer.fetch_geopolitical_news()
        analyzed_news = self.news_analyzer.analyze_news_sentiment(news_items)
        
        print(f"Проанализировано {len(analyzed_news)} новостей")
        for news in analyzed_news[:3]:
            print(f"  - {news['title']} (Влияние: {news['impact_score']:.1f})")
        
        # 3. Расчет влияния на компании
        print("\n3. Расчет влияния на компании...")
        
        # Чувствительность к ресурсам
        resource_sensitivity = self.resource_analyzer.calculate_resource_sensitivity(
            df_companies, resource_prices
        )
        
        # Геополитические риски
        geopolitical_risks = self.news_analyzer.map_companies_to_geopolitical_risks(
            df_companies, analyzed_news
        )
        
        # 4. Интеграция результатов в DataFrame
        for idx in df_companies.index:
            if idx in resource_sensitivity:
                sens = resource_sensitivity[idx]
                df_companies.at[idx, 'resource_sensitivity'] = sens['resource_sensitivity_score']
                df_companies.at[idx, 'resource_exposure'] = sens['resource_exposure_count']
                df_companies.at[idx, 'primary_resource'] = sens['primary_resource']
                df_companies.at[idx, 'avg_resource_change'] = sens.get('avg_resource_change_1m', 0)
            
            if idx in geopolitical_risks:
                risk = geopolitical_risks[idx]
                df_companies.at[idx, 'geopolitical_risk'] = risk['geopolitical_risk_score']
                df_companies.at[idx, 'geopolitical_risk_level'] = risk['geopolitical_risk_level']
                df_companies.at[idx, 'related_news_count'] = risk['related_news_count']
                df_companies.at[idx, 'primary_risks'] = ', '.join(risk['primary_risks'][:3]) if risk['primary_risks'] else ''
        
        # 5. Расчет комплексного рейтинга с учетом внешних факторов
        df_companies = self.calculate_comprehensive_rating(df_companies)
        
        return df_companies, resource_prices, analyzed_news
    
    def calculate_comprehensive_rating(self, df):
        """Расчет комплексного рейтинга"""
        
        def calculate_row_rating(row):
            # Базовый рейтинг из фундаментального анализа (если есть)
            base_rating = row.get('Счет_качества', 0)
            
            # Влияние ресурсов
            resource_score = row.get('resource_sensitivity', 50)  # 50 - нейтральный
            resource_weight = 0.3 if row.get('resource_exposure', 0) > 0 else 0.1
            
            # Геополитические риски
            geo_risk = row.get('geopolitical_risk', 0)
            geo_weight = 0.4 if row.get('related_news_count', 0) > 0 else 0.2
            
            # Адаптация весов в зависимости от отрасли
            industry_factor = self.get_industry_factor(row.get('Название', ''))
            
            # Комплексный расчет
            comprehensive_score = (
                base_rating * 0.4 +
                resource_score * resource_weight * industry_factor +
                (100 - geo_risk) * geo_weight  # Инвертируем риск в возможность
            )
            
            # Нормализация
            comprehensive_score = min(100, max(0, comprehensive_score))
            
            # Определение категории
            if comprehensive_score >= 70:
                category = 'A: Отличные перспективы'
            elif comprehensive_score >= 55:
                category = 'B: Хорошие перспективы'
            elif comprehensive_score >= 40:
                category = 'C: Средние перспективы'
            elif comprehensive_score >= 25:
                category = 'D: Высокие риски'
            else:
                category = 'E: Критические риски'
            
            return pd.Series([comprehensive_score, category])
        
        df[['comprehensive_score', 'comprehensive_category']] = df.apply(calculate_row_rating, axis=1)
        
        return df
    
    def get_industry_factor(self, company_name):
        """Фактор чувствительности отрасли к внешним факторам"""
        name_lower = str(company_name).lower()
        
        # Отрасли с высокой чувствительностью
        high_sensitivity = ['нефт', 'газ', 'метал', 'банк', 'финанс']
        # Отрасли со средней чувствительностью
        medium_sensitivity = ['энерг', 'транспорт', 'телеком']
        # Отрасли с низкой чувствительностью
        low_sensitivity = ['ритейл', 'потребит', 'услуг', 'софт']
        
        if any(keyword in name_lower for keyword in high_sensitivity):
            return 1.5  # Высокая чувствительность
        elif any(keyword in name_lower for keyword in medium_sensitivity):
            return 1.2  # Средняя чувствительность
        elif any(keyword in name_lower for keyword in low_sensitivity):
            return 0.8  # Низкая чувствительность
        else:
            return 1.0  # Нейтральная
    
    def create_risk_heatmap(self, df):
        """Создание тепловой карты рисков"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 10))
        
        # Подготовка данных для визуализации
        risk_data = df[['Название', 'geopolitical_risk', 'resource_sensitivity', 
                       'comprehensive_score']].copy()
        risk_data = risk_data.sort_values('comprehensive_score', ascending=False).head(20)
        
        risk_data.set_index('Название', inplace=True)
        
        # Создание тепловой карты
        sns.heatmap(risk_data.T, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   center=50, linewidths=0.5, linecolor='gray')
        
        plt.title('Тепловая карта рисков и возможностей', fontsize=16)
        plt.xlabel('Компании', fontsize=12)
        plt.ylabel('Показатели', fontsize=12)
        plt.tight_layout()
        plt.savefig('risk_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return risk_data