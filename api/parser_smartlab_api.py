import os
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
    # Ваш исходный вариант, но с добавлением папки data
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, 'data', 'companies_data.xlsx')

    # Убедимся, что папка существует
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df.to_excel(file_path, index=False)

if __name__ == "__main__":
    get_fundamental_shares()    