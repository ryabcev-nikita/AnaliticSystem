import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

def create_risk_assessment_nn(input_shape):
    """Создание ансамбля нейросетей для оценки риска"""
    
    models = []
    
    # Разные архитектуры для ансамбля
    architectures = [
        # Простая сеть
        keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(4, activation='softmax')  # 4 категории риска
        ]),
        
        # Более глубокая сеть
        keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(4, activation='softmax')
        ]),
        
        # Широкая сеть
        keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
    ]
    
    return architectures

def train_ensemble_nn(df, n_folds=5):
    """Обучение ансамбля нейросетей с кросс-валидацией"""
    
    # Подготовка данных
    feature_cols = ['ДД_используемая', 'P_E', 'P_B', 'EV_EBITDA',
                   'Рентабельность_EBITDA', 'Прибыльность', 'EV_Sales']
    
    X = []
    y = []
    
    # Преобразуем категории в числовые метки
    category_map = {
        'A: Высокая доходность / Низкий риск': 0,
        'B: Средняя доходность / Средний риск': 1,
        'C: Низкая доходность / Высокий риск': 2,
        'D: Спекулятивная / Очень высокий риск': 3
    }
    
    for _, row in df.iterrows():
        feature_vector = []
        for col in feature_cols:
            val = row.get(col, 0)
            feature_vector.append(val if pd.notna(val) else 0)
        
        X.append(feature_vector)
        category = row.get('Категория', 'C: Низкая доходность / Высокий риск')
        y.append(category_map.get(category, 2))
    
    X = np.array(X)
    y = np.array(y)
    
    # Нормализация
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot кодирование целей
    y_categorical = keras.utils.to_categorical(y, num_classes=4)
    
    # Кросс-валидация
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    ensemble_predictions = []
    models = create_risk_assessment_nn(len(feature_cols))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\nTraining fold {fold + 1}/{n_folds}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
        
        fold_predictions = []
        
        for i, model in enumerate(models):
            print(f"  Training model {i+1}/{len(models)}")
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=16,
                verbose=0
            )
            
            # Прогнозирование на валидационной выборке
            val_pred = model.predict(X_val)
            fold_predictions.append(val_pred)
        
        # Усреднение предсказаний моделей в фолде
        avg_pred = np.mean(fold_predictions, axis=0)
        ensemble_predictions.append((val_idx, avg_pred))
    
    # Собираем все предсказания
    all_preds = np.zeros_like(y_categorical)
    for idx, pred in ensemble_predictions:
        all_preds[idx] = pred
    
    # Добавляем результаты в DataFrame
    df['NN_Категория'] = np.argmax(all_preds, axis=1)
    df['NN_Уверенность'] = np.max(all_preds, axis=1)
    
    # Создаем словарь обратного преобразования
    reverse_map = {v: k for k, v in category_map.items()}
    df['NN_Категория_текст'] = df['NN_Категория'].map(reverse_map)
    
    return df, models, scaler