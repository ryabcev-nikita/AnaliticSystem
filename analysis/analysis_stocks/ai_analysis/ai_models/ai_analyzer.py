# file: neural_network.py
"""
Модуль для анализа недооцененности акций с помощью нейронной сети
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict
from ai_models.ai_config import AnalysisConfig


class UndervaluedDetector:
    """Детектор недооцененных акций на основе нейронной сети"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Инициализация детектора
        
        Args:
            config: конфигурация анализа
        """
        self.config = config
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=config.NN_HIDDEN_LAYERS,
            activation=config.NN_ACTIVATION,
            solver=config.NN_SOLVER,
            max_iter=config.NN_MAX_ITER,
            random_state=config.RANDOM_STATE,
            verbose=False
        )
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.feature_names = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Обучение нейронной сети
        
        Args:
            X: признаки
            y: целевая переменная (log EV/EBITDA)
            
        Returns:
            Словарь с историей обучения
        """
        self.feature_names = X.columns.tolist()
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        print("\nОбучение нейронной сети...")
        
        # Обучение с сохранением истории
        for i in range(self.config.NN_EPOCHS):
            self.model.partial_fit(X_train, y_train)
            
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_test)
            
            train_loss = mean_squared_error(y_train, train_pred)
            val_loss = mean_squared_error(y_test, val_pred)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{self.config.NN_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание P/E
        
        Args:
            X: признаки
            
        Returns:
            Массив предсказанных значений EV/EBITDA
        """
        X_scaled = self.scaler.transform(X)
        log_pe_pred = self.model.predict(X_scaled)
        return np.exp(log_pe_pred)
    
    def get_undervalued_stocks(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Определение недооцененных акций
        
        Args:
            df: DataFrame с данными
            top_n: количество лучших акций для отбора
            
        Returns:
            DataFrame с отобранными акциями
        """
        X = df[self.config.FEATURE_COLUMNS].fillna(0)
        
        # Предсказание
        predicted_pe = self.predict(X)
        
        # Расчет метрик недооцененности
        result_df = df.copy()
        result_df['predicted_pe'] = predicted_pe
        result_df['residual'] = result_df['pe'] - result_df['predicted_pe']
        result_df['undervalued_score'] = -result_df['residual'] / result_df['predicted_pe']
        
        # Отбор наиболее недооцененных
        undervalued_df = result_df.nsmallest(top_n, 'residual')
        
        # Расчет ожидаемой доходности
        undervalued_df['expected_return'] = undervalued_df['dividend_yield']/100 + undervalued_df['g']
        
        return undervalued_df
    
    def get_training_history(self) -> Dict:
        """Получение истории обучения"""
        return self.training_history
    
    def get_final_losses(self) -> Tuple[float, float]:
        """Получение финальных значений потерь"""
        return (
            self.training_history['train_loss'][-1],
            self.training_history['val_loss'][-1]
        )