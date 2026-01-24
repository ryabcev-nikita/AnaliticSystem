# В основной скрипт добавляем:

def main_analysis_with_external_factors():
    """Основной анализ с внешними факторами"""
    
    # 1. Загрузка и обработка данных компаний
    df = load_company_data_simple('companies_data.xlsx')
    
    # 2. Базовый фундаментальный анализ
    df = perform_fundamental_analysis(df)  # Ваш существующий анализ
    
    # 3. Анализ внешних факторов
    advanced_analyzer = AdvancedStockAnalyzer()
    df_with_external, resource_prices, news = advanced_analyzer.analyze_with_external_factors(df)
    
    # 4. Обучение нейросети с внешними факторами
    df_with_nn, nn_model = train_external_factors_nn(df_with_external)
    
    # 5. Создание рекомендаций с учетом всех факторов
    recommendations = generate_comprehensive_recommendations(df_with_nn)
    
    # 6. Визуализация результатов
    risk_heatmap = advanced_analyzer.create_risk_heatmap(df_with_nn)
    
    # 7. Сохранение результатов
    save_comprehensive_report(df_with_nn, recommendations, resource_prices, news)
    
    return df_with_nn, recommendations

def generate_comprehensive_recommendations(df):
    """Генерация рекомендаций с учетом всех факторов"""
    
    recommendations = []
    
    for idx, row in df.iterrows():
        score = row.get('comprehensive_score', 0)
        resource_exp = row.get('resource_exposure', 0)
        geo_risk = row.get('geopolitical_risk', 0)
        
        # Логика рекомендаций
        if score >= 70 and geo_risk < 30:
            recommendation = "Сильная покупка"
            reason = "Отличные фундаментальные показатели, низкие внешние риски"
        elif score >= 60 and geo_risk < 40:
            recommendation = "Покупка"
            reason = "Хорошие показатели, умеренные риски"
        elif score >= 50:
            recommendation = "Умеренная покупка"
            reason = "Средние показатели, требуют мониторинга рисков"
        elif score >= 40:
            recommendation = "Удержание"
            reason = "Нейтральные показатели, высокие риски"
        elif score >= 30:
            recommendation = "Продажа"
            reason = "Слабые показатели, высокие риски"
        else:
            recommendation = "Сильная продажа"
            reason = "Критические показатели и риски"
        
        # Учет специфики ресурсной зависимости
        if resource_exp > 3:
            reason += f". Высокая зависимость от ресурсов ({resource_exp} видов)"
        
        recommendations.append({
            'Название': row.get('Название', ''),
            'Тикер': row.get('Тикер', ''),
            'Рекомендация': recommendation,
            'Общий_скор': score,
            'Геополитический_риск': geo_risk,
            'Ресурсная_зависимость': resource_exp,
            'Обоснование': reason
        })
    
    return pd.DataFrame(recommendations)