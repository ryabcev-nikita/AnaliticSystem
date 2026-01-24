class AnomalyDetectorAE(nn.Module):
    """Автоэнкодер для обнаружения аномалий в мультипликаторах"""
    def __init__(self, input_size):
        super().__init__()
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Скрытое представление
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def detect_anomalies_with_ae(df):
    """Обнаружение аномалий и недооцененных акций с помощью автоэнкодера"""
    
    # Подготовка данных
    feature_cols = ['ДД_используемая', 'P_E', 'P_B', 'EV_EBITDA',
                   'Рентабельность_EBITDA', 'Прибыльность', 'EV_Sales']
    
    features = []
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
            features.append(feature_vector)
            valid_indices.append(idx)
    
    if not features:
        print("Недостаточно данных для обучения автоэнкодера")
        return df
    
    X = np.array(features)
    
    # Нормализация
    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler = RobustScaler()  # RobustScaler более устойчив к выбросам
    X_scaled = scaler.fit_transform(X)
    
    # Обучение автоэнкодера
    input_size = len(feature_cols)
    model = AnomalyDetectorAE(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_scaled))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Обучение
    n_epochs = 100
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Reconstruction Loss: {total_loss/len(dataloader):.6f}")
    
    # Вычисление ошибок реконструкции
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        _, reconstructed = model(X_tensor)
        
        # Ошибка реконструкции
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        # Получаем скрытые представления
        encoded, _ = model(X_tensor)
        encoded_features = encoded.numpy()
    
    # Нормализуем ошибки
    errors_np = reconstruction_errors.numpy()
    error_mean = errors_np.mean()
    error_std = errors_np.std()
    
    # Определяем аномалии (выбросы)
    anomaly_threshold = error_mean + 2 * error_std
    is_anomaly = errors_np > anomaly_threshold
    
    # Находим недооцененные акции (низкая ошибка реконструкции + хорошие фундаментальные показатели)
    undervalued_scores = []
    for i, idx in enumerate(valid_indices):
        # Комбинированный скор: низкая ошибка + хорошие показатели
        error_score = 1.0 / (1.0 + errors_np[i])  # Чем меньше ошибка, тем выше скор
        
        fundamental_score = 0
        row = df.iloc[idx]
        if row.get('P_E', 100) < 15 and row.get('P_E', 100) > 0:
            fundamental_score += 1
        if row.get('P_B', 10) < 2 and row.get('P_B', 10) > 0:
            fundamental_score += 1
        if row.get('ДД_используемая', 0) > 0.05:
            fundamental_score += 1
        
        combined_score = error_score * (1 + fundamental_score * 0.2)
        undervalued_scores.append(combined_score)
    
    # Добавляем результаты в DataFrame
    df['AE_Ошибка_реконструкции'] = np.nan
    df['AE_Аномалия'] = False
    df['AE_Недооцененность'] = np.nan
    df['AE_Скрытые_признаки'] = None
    
    for i, idx in enumerate(valid_indices):
        df.at[idx, 'AE_Ошибка_реконструкции'] = errors_np[i]
        df.at[idx, 'AE_Аномалия'] = is_anomaly[i]
        df.at[idx, 'AE_Недооцененность'] = undervalued_scores[i]
        df.at[idx, 'AE_Скрытые_признаки'] = str(encoded_features[i])
    
    return df, model, scaler