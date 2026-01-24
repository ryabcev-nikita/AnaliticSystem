import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_fundamental_shares():
    # URL страницы с таблицей
    url = 'https://smart-lab.ru/q/shares_fundamental2'

    # Получение HTML-кода страницы
    response = requests.get(url)
    time.sleep(5)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Поиск таблицы
    table = soup.find('table', class_='simple-little-table little trades-table')

    print(table)
    # Инициализация списков для хранения данных
    data = []

    # Получение заголовков таблицы
    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text(strip=True))

    # Получение строк таблицы
    for row in table.find_all('tr'):  # Пропускаем заголовок
        cols = row.find_all('td')
        cols = [col.get_text(strip=True).replace('\xa0', '') for col in cols]  # Убираем лишние пробелы
        data.append(cols)

    # Создание DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Сохранение в Excel
    df.to_excel(f'companies_data.xlsx', index=False)

    print("Данные успешно сохранены в файл 'companies_data.xlsx'")