import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ExternalFactorsNN(nn.Module):
    """Нейросеть для анализа с учетом внешних факторов"""
    
    def __init__(self, num_fundamental_features, num_external_features):
        super().__init__()
        
        # Ветвь для фундаментальных показателей
        self.fundamental_branch = nn.Sequential(
            nn.Linear(num_fundamental_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Ветвь для внешних факторов
        self.external_branch = nn.Sequential(
            nn.Linear(num_external_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Объединенный слой
        self.combined = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Прогноз доходности
            nn.Sigmoid()  # Приводим к диапазону 0-1
        )
        
        # Внимание к внешним факторам
        self.attention = nn.MultiheadAttention(
            embed_dim=num_external_features,
            num_heads=4,
            dropout=0.1
        )
    
    def forward(self, fundamental, external):
        # Обработка фундаментальных показателей
        fund_features = self.fundamental_branch(fundamental)
        
        # Механизм внимания для внешних факторов
        external = external.unsqueeze(0)  # Добавляем dimension для внимания
        attended_external, _ = self.attention(external, external, external)
        attended_external = attended_external.squeeze(0)
        
        # Обработка внешних факторов
        ext_features = self.external_branch(attended_external)
        
        # Объединение признаков
        combined = torch.cat([fund_features, ext_features], dim=1)
        
        # Прогноз
        output = self.combined(combined)
        
        return output

class ExternalFactorsDataset(Dataset):
    """Датасет с внешними факторами"""
    
    def __init__(self, df, fundamental_cols, external_cols):
        self.fundamental_data = []
        self.external_data = []
        self.targets = []
        
        for _, row in df.iterrows():
            # Фундаментальные показатели
            fund_features = [row[col] if pd.notna(row[col]) else 0 
                           for col in fundamental_cols]
            
            # Внешние факторы
            ext_features = [
                row.get('resource_sensitivity', 50) / 100,  # Нормализация
                row.get('geopolitical_risk', 50) / 100,
                row.get('resource_exposure', 0) / 10,  # Нормализация
                row.get('related_news_count', 0) / 10
            ]
            
            # Целевая переменная (например, категория риска)
            target = 1 if row.get('comprehensive_score', 50) > 60 else 0
            
            self.fundamental_data.append(fund_features)
            self.external_data.append(ext_features)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.fundamental_data)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.fundamental_data[idx]),
            torch.FloatTensor(self.external_data[idx]),
            torch.FloatTensor([self.targets[idx]])
        )

def train_external_factors_nn(df):
    """Обучение нейросети с внешними факторами"""
    
    # Определение признаков
    fundamental_cols = ['ДД_используемая', 'P_E', 'P_B', 'EV_EBITDA',
                       'Рентабельность_EBITDA', 'Прибыльность', 'EV_Sales']
    
    # Создание датасета
    dataset = ExternalFactorsDataset(df, fundamental_cols, [])
    
    # Разделение на train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Создание модели
    model = ExternalFactorsNN(
        num_fundamental_features=len(fundamental_cols),
        num_external_features=4  # resource_sensitivity, geopolitical_risk и т.д.
    )
    
    # Обучение
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for fund_batch, ext_batch, target_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(fund_batch, ext_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Прогнозирование
    model.eval()
    with torch.no_grad():
        predictions = []
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        for fund_batch, ext_batch, _ in test_loader:
            batch_preds = model(fund_batch, ext_batch)
            predictions.extend(batch_preds.numpy())
    
    # Добавление прогнозов в DataFrame
    df['nn_external_prediction'] = 0.0
    for i, (_, row_idx) in enumerate(test_dataset.indices):
        df.iloc[row_idx, df.columns.get_loc('nn_external_prediction')] = predictions[i]
    
    return df, model