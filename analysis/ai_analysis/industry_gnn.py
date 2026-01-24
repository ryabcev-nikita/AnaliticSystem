import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class IndustryGNN(torch.nn.Module):
    """Графовая нейросеть для анализа отраслевых связей"""
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.lin(x)
        return x

def create_industry_graph(df):
    """Создание графа отраслевых связей между компаниями"""
    
    # Группируем компании по отраслям (по первой части названия или ключевым словам)
    industry_keywords = {
        'Нефтегаз': ['нефт', 'газ', 'роснефт', 'лукойл', 'татнефт', 'сургут', 'башнефт'],
        'Финансы': ['банк', 'страх', 'биржа', 'инвест', 'финанс', 'кредит'],
        'Металлургия': ['метал', 'сталь', 'алюмин', 'никел', 'медь', 'золот', 'серебр'],
        'Энергетика': ['энерг', 'электр', 'тепло', 'гидро', 'атом', 'мощност'],
        'Телеком': ['телеком', 'связь', 'мтс', 'билайн', 'теле2', 'ростелеком'],
        'Ритейл': ['магнит', 'лента', 'x5', 'озон', 'яндекс', 'м.видео', 'эльдорадо'],
        'Транспорт': ['транспорт', 'аэро', 'железн', 'судо', 'авто', 'транснефт'],
        'Технологии': ['технолог', 'софт', 'ит', 'интернет', 'кибер', 'нейро'],
        'Химия': ['хим', 'азот', 'фос', 'агро', 'пласт', 'полимер'],
    }
    
    # Присваиваем компании отрасли
    company_features = []
    node_labels = []
    
    for idx, row in df.iterrows():
        name = str(row.get('Название', '')).lower()
        
        # Признаки компании
        feature_vector = [
            row.get('ДД_используемая', 0) if pd.notna(row.get('ДД_используемая')) else 0,
            row.get('P_E', 0) if pd.notna(row.get('P_E')) else 0,
            row.get('P_B', 0) if pd.notna(row.get('P_B')) else 0,
            row.get('Рентабельность_EBITDA', 0) if pd.notna(row.get('Рентабельность_EBITDA')) else 0,
            row.get('EV_EBITDA', 0) if pd.notna(row.get('EV_EBITDA')) else 0,
        ]
        
        company_features.append(feature_vector)
        
        # Определяем отрасль
        industry = 'Другое'
        for ind_name, keywords in industry_keywords.items():
            if any(keyword in name for keyword in keywords):
                industry = ind_name
                break
        
        node_labels.append(industry)
    
    # Создаем ребра между компаниями одной отрасли
    edge_index = []
    industry_to_indices = {}
    
    for i, industry in enumerate(node_labels):
        if industry not in industry_to_indices:
            industry_to_indices[industry] = []
        industry_to_indices[industry].append(i)
    
    # Добавляем ребра внутри каждой отрасли
    for industry, indices in industry_to_indices.items():
        if len(indices) > 1:
            # Полносвязный граф внутри отрасли
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edge_index.append([indices[i], indices[j]])
                    edge_index.append([indices[j], indices[i]])  # Неориентированное ребро
    
    # Добавляем ребра между крупными компаниями (топ-10 по капитализации)
    df_sorted = df.sort_values('Капитализация', ascending=False)
    top_indices = df_sorted.head(10).index.tolist()
    
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            idx_i = df.index.get_loc(top_indices[i])
            idx_j = df.index.get_loc(top_indices[j])
            edge_index.append([idx_i, idx_j])
            edge_index.append([idx_j, idx_i])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    
    # Создаем граф
    x = torch.tensor(company_features, dtype=torch.float)
    
    # Преобразуем метки отраслей в числовые
    unique_industries = list(set(node_labels))
    industry_to_label = {ind: i for i, ind in enumerate(unique_industries)}
    y = torch.tensor([industry_to_label[label] for label in node_labels], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data, node_labels, unique_industries

def analyze_with_gnn(df):
    """Анализ с помощью графовых нейросетей"""
    
    # Создаем граф
    data, node_labels, industries = create_industry_graph(df)
    
    print(f"Граф создан: {data.num_nodes} узлов, {data.num_edges} ребер")
    print(f"Отрасли: {industries}")
    
    # Создаем модель GNN
    model = IndustryGNN(
        num_features=data.num_features,
        hidden_channels=64,
        num_classes=len(industries)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Разделение на train/test
    train_mask = torch.rand(data.num_nodes) < 0.8
    test_mask = ~train_mask
    
    # Обучение
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Оценка
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
                test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Получаем эмбеддинги узлов
    model.eval()
    with torch.no_grad():
        # Используем выход второго слоя как эмбеддинги
        x = data.x
        x = model.conv1(x, data.edge_index).relu()
        x = model.conv2(x, data.edge_index).relu()
        embeddings = model.conv3(x, data.edge_index)
        
        # Сохраняем эмбеддинги в DataFrame
        df['GNN_Эмбеддинг'] = [str(emb.tolist()) for emb in embeddings]
        
        # Кластеризация эмбеддингов для выявления групп
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(embeddings.numpy())
        
        df['GNN_Кластер'] = clusters
    
    return df, model, data