class AttentionModel(nn.Module):
    """Модель с механизмом внимания для анализа важности мультипликаторов"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        self.feature_embedding = nn.Linear(input_size, hidden_size)
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 4)  # 4 категории риска
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, return_attention=False):
        # Преобразуем входные признаки
        batch_size = x.size(0)
        
        # Встраивание признаков
        embedded = self.feature_embedding(x).unsqueeze(0)  # (1, batch_size, hidden_size)
        
        # Self-attention
        attn_output, attn_weights = self.attention(embedded, embedded, embedded)
        
        # Извлекаем контекстный вектор
        context = attn_output.squeeze(0)
        
        # Классификация
        x = self.dropout(context)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        if return_attention:
            return x, attn_weights
        return x

def analyze_with_attention(df):
    """Анализ с механизмом внимания для понимания важности признаков"""
    
    # Подготовка данных
    feature_cols = ['ДД_используемая', 'P_E', 'P_B', 'EV_EBITDA',
                   'Рентабельность_EBITDA', 'Прибыльность', 'EV_Sales']
    
    X = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        feature_vector = []
        valid = True
        
        for col in feature_cols:
            val = row.get(col, None)
            if pd.isna(val):
                valid = False
                break
            feature_vector.append(float(val))
        
        if valid and len(feature_vector) == len(feature_cols):
            X.append(feature_vector)
            valid_indices.append(idx)
    
    if not X:
        print("Недостаточно данных")
        return df
    
    X = np.array(X)
    
    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Создание меток (используем существующие категории)
    category_map = {
        'A: Высокая доходность / Низкий риск': 0,
        'B: Средняя доходность / Средний риск': 1,
        'C: Низкая доходность / Высокий риск': 2,
        'D: Спекулятивная / Очень высокий риск': 3
    }
    
    y = []
    for idx in valid_indices:
        category = df.at[idx, 'Категория']
        y.append(category_map.get(category, 2))
    
    y = np.array(y)
    
    # Разделение данных
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Обучение модели
    model = AttentionModel(len(feature_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Обучение
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
    
    # Анализ весов внимания
    model.eval()
    with torch.no_grad():
        # Получаем веса внимания для всех компаний
        all_attention_weights = []
        
        for i in range(0, len(X_scaled), 16):
            batch = X_scaled[i:i+16]
            batch_tensor = torch.FloatTensor(batch)
            _, attn_weights = model(batch_tensor, return_attention=True)
            all_attention_weights.append(attn_weights)
        
        # Агрегируем веса внимания
        if all_attention_weights:
            avg_attention = torch.mean(torch.cat(all_attention_weights, dim=0), dim=0)
            avg_attention = avg_attention.squeeze().mean(dim=0)  # Усредняем по головам внимания
            
            print("\nСредняя важность признаков:")
            for i, col in enumerate(feature_cols):
                importance = avg_attention[i].item()
                print(f"{col}: {importance:.4f}")
            
            # Сохраняем веса внимания для каждой компании
            df['Attention_Weights'] = ""
            for i, idx in enumerate(valid_indices):
                if i < len(X_scaled):
                    sample_tensor = torch.FloatTensor(X_scaled[i:i+1])
                    _, attn_weights = model(sample_tensor, return_attention=True)
                    weights_str = ','.join([f"{w:.4f}" for w in attn_weights.squeeze().mean(dim=0).numpy()])
                    df.at[idx, 'Attention_Weights'] = weights_str
    
    return df, model, scaler, feature_cols