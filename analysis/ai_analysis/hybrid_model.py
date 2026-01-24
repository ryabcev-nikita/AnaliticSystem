class HybridModel:
    """Гибридная модель: дерево решений для feature engineering + нейросеть"""
    
    def __init__(self):
        self.tree_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        
    def extract_tree_features(self, X, y):
        """Извлечение признаков с помощью дерева решений"""
        from sklearn.tree import DecisionTreeClassifier
        
        tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        tree.fit(X, y)
        
        # Получаем путь для каждого образца
        tree_paths = tree.decision_path(X)
        
        # Используем sparse матрицу путей как новые признаки
        return tree_paths.toarray(), tree
    
    def train(self, X, y):
        """Обучение гибридной модели"""
        
        # 1. Извлекаем признаки с помощью дерева
        tree_features, self.tree_model = self.extract_tree_features(X, y)
        
        # 2. Объединяем исходные признаки с tree features
        combined_features = np.hstack([X, tree_features])
        
        # 3. Нормализуем
        combined_scaled = self.scaler.fit_transform(combined_features)
        
        # 4. Создаем нейросеть
        self.nn_model = nn.Sequential(
            nn.Linear(combined_scaled.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # 4 категории
        )
        
        # 5. Обучение нейросети
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)
        
        dataset = StockDataset(combined_scaled, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, X):
        """Прогнозирование"""
        tree_features = self.tree_model.decision_path(X).toarray()
        combined = np.hstack([X, tree_features])
        combined_scaled = self.scaler.transform(combined)
        
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(combined_scaled)
            outputs = self.nn_model(X_tensor)
            predictions = torch.softmax(outputs, dim=1)
            
        return predictions.numpy()