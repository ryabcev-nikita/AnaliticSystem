# file: data_loader.py
"""
Модуль для загрузки и предобработки данных
"""
import pandas as pd
import numpy as np
from typing import Optional
from ai_models.ai_config import AnalysisConfig


class RussianNumberParser:
    """Парсер русских числовых форматов"""
    
    @staticmethod
    def parse(value) -> Optional[float]:
        """
        Парсинг чисел в русском формате (млрд, млн, тыс)
        
        Args:
            value: значение для парсинга
            
        Returns:
            float или np.nan в случае ошибки
        """
        if pd.isna(value) or value == '' or value == 0:
            return np.nan
            
        if isinstance(value, (int, float)):
            return float(value)
            
        try:
            value_str = str(value).replace(' ', '').replace(',', '.')
            
            if 'млрд' in value_str:
                num = float(value_str.replace('млрд', '').strip())
                return num * 1e9
            elif 'млн' in value_str:
                num = float(value_str.replace('млн', '').strip())
                return num * 1e6
            elif 'тыс' in value_str:
                num = float(value_str.replace('тыс', '').strip())
                return num * 1e3
            else:
                return float(value_str)
                
        except (ValueError, TypeError):
            return np.nan


class DataLoader:
    """Загрузчик данных из Excel файла"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация загрузчика
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
        self.parser = RussianNumberParser()
        self.column_names = [
            'ticker', 'name', 'asset_uid', 'figi', 'currency', 'market_cap', 'ev', 'revenue',
            'net_income', 'ebitda', 'pe', 'pb', 'ps', 'pfcf', 'roe', 'roa', 'roic',
            'ev_ebitda', 'ev_s', 'fcf', 'cagr_sales', 'avg_dividend_yield',
            'avg_cagr_dividend', 'current_ratio', 'payout_ratio', 'npm', 'debt',
            'debt_capital', 'net_debt_ebitda', 'debt_ebitda', 'eps', 'dividend_yield',
            'beta', 'dividend_per_share'
        ]
    
    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из Excel файла
        
        Returns:
            DataFrame с загруженными данными
        """
        print(f"Загрузка данных из {self.config.INPUT_FILE}...")
        
        try:
            df = pd.read_excel(self.config.INPUT_FILE, sheet_name='Sheet1', header=None, skiprows=1)
            df.columns = self.column_names
            
            # Парсинг числовых колонок
            for col in self.config.NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].apply(self.parser.parse)
            
            print(f"Загружено {len(df)} компаний")
            return df
            
        except FileNotFoundError:
            print(f"Ошибка: файл {self.config.INPUT_FILE} не найден")
            raise
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise


class DataPreprocessor:
    """Препроцессор данных для анализа"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация препроцессора
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
    
    def calculate_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет темпа роста g = (1 - payout_ratio) * ROE
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленной колонкой g
        """
        df = df.copy()
        df['payout_ratio'] = df['payout_ratio'].fillna(0)
        df['roe'] = df['roe'].fillna(0) / 100  # перевод процентов в десятичные дроби
        df['g'] = (1 - df['payout_ratio']/100) * df['roe']
        return df
    
    def filter_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Фильтрация компаний по критериям
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Отфильтрованный DataFrame
        """
        filtered_df = df[
            (df['ev_ebitda'] > self.config.MIN_EV_EBITDA) &
             (df['ev_ebitda'] < self.config.MAX_EV_EBITDA) &
            (df['roe'].notna()) &
            (df['pb'].notna()) &
            (df['pb'] > self.config.MIN_PB) &
            (df['pb'] < self.config.MAX_PB) &
            (df['dividend_yield'].notna()) &
            (df['beta'].notna())
        ].copy()
        
        # Логарифмируем P/E для более нормального распределения
        filtered_df['log_ev_ebitda'] = np.log(filtered_df['ev_ebitda'])
        
        print(f"После фильтрации осталось {len(filtered_df)} компаний")
        return filtered_df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Подготовка признаков для нейронной сети
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Кортеж (X, y, feature_names)
        """
        X = df[self.config.FEATURE_COLUMNS].fillna(0)
        y = df['log_ev_ebitda']
        
        return X, y, self.config.FEATURE_COLUMNS