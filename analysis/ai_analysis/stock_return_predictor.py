import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class StockReturnPredictor(nn.Module):
    """Нейросеть для прогнозирования будущей доходности"""
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 класса: высокая, средняя, низкая доходность
        )
    
    def forward(self, x):
        return self.network(x)

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def predict_future_returns_nn(df, future_periods=12):
    """Прогнозирование будущей доходности с помощью нейросети"""
    
    # Подготовка исторических данных (в реальности нужны временные ряды)
    features = []
    targets = []
    
    # Используем текущие мультипликаторы как признаки
    feature_cols = ['ДД_используемая', 'P_E', 'P_B', 'EV_EBITDA', 
                   'Рентабельность_EBITDA', 'Прибыльность', 'EV_Sales']
    
    # Создаем синтетические целевые переменные (в реальности нужны исторические данные о доходности)
    for _, row in df.iterrows():
        # Признаки
        feature_vector = []
        for col in feature_cols:
            val = row.get(col, 0)
            feature_vector.append(val if pd.notna(val) else 0)
        
        # Синтетическая цель: на основе текущих показателей прогнозируем будущую доходность
        score = 0
        if row.get('ДД_используемая', 0) > 0.05:
            score += 1
        if row.get('P_E', 100) < 15:
            score += 1
        if row.get('P_B', 10) < 2:
            score += 1
        if row.get('Рентабельность_EBITDA', 0) > 0.2:
            score += 1
        
        # Классификация: 0-низкая, 1-средняя, 2-высокая доходность
        target_class = min(2, score // 2)
        
        features.append(feature_vector)
        targets.append(target_class)
    
    # Нормализация признаков
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Разделение данных
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
    y_train, y_test = targets[:split_idx], targets[split_idx:]
    
    # Создание датасетов
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    # Обучение модели
    model = StockReturnPredictor(len(feature_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Тренировка
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Прогнозирование для всех компаний
    model.eval()
    with torch.no_grad():
        all_features = torch.FloatTensor(scaler.transform(features))
        predictions = model(all_features)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    # Добавляем прогнозы в DataFrame
    df['Прогноз_доходности_NN'] = predicted_classes.numpy()
    df['Вероятность_высокой_доходности'] = torch.softmax(predictions, dim=1)[:, 2].numpy()
    
    return df, model