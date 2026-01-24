from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

class InvestingResourceAnalyzer:
    """Получение данных с Investing.com"""
    
    def __init__(self):
        self.resource_urls = {
            'Нефть Brent': 'https://ru.investing.com/commodities/brent-oil',
            'Нефть WTI': 'https://ru.investing.com/commodities/crude-oil',
            'Золото': 'https://ru.investing.com/commodities/gold',
            'Серебро': 'https://ru.investing.com/commodities/silver',
            'Медь': 'https://ru.investing.com/commodities/copper',
            'Палладий': 'https://ru.investing.com/commodities/palladium',
            'Платина': 'https://ru.investing.com/commodities/platinum',
            'Газ природный': 'https://ru.investing.com/commodities/natural-gas',
            'Пшеница': 'https://ru.investing.com/commodities/wheat',
            'Кукуруза': 'https://ru.investing.com/commodities/corn',
            'Соя': 'https://ru.investing.com/commodities/us-soybeans',
            'Никель': 'https://ru.investing.com/commodities/nickel',
            'Алюминий': 'https://ru.investing.com/commodities/aluminum',
            'Сахар': 'https://ru.investing.com/commodities/us-sugar-no11',
            'Кофе': 'https://ru.investing.com/commodities/robusta-coffee'
        }
    
    def get_prices_from_investing(self):
        """Парсинг данных с Investing.com"""
        resource_data = {}
        
        # Настройка Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Без GUI
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        
        for resource_name, url in self.resource_urls.items():
            try:
                driver.get(url)
                
                # Ждем загрузки цены
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "text-2xl"))
                )
                
                # Получаем цену
                price_element = driver.find_element(By.CLASS_NAME, "text-2xl")
                print(price)
                price = float(price_element.text.replace(',', ''))
                
                # Получаем изменение
                change_element = driver.find_element(By.XPATH, 
                    "//div[contains(@class, 'instrument-price_change')]")
                change_text = change_element.text
                
                resource_data[resource_name] = {
                    'Цена': price,
                    'Изменение': change_text,
                    'URL': url
                }
                
                print(f"{resource_name}: {price}")
                
            except Exception as e:
                print(f"Ошибка для {resource_name}: {e}")
                continue
        
        driver.quit()
        return pd.DataFrame(resource_data).T

investing = InvestingResourceAnalyzer()
result = investing.get_prices_from_investing()
print(result)
result.to_excel('result.xlsx', index=False)
