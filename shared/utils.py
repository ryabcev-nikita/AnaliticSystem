import matplotlib
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

matplotlib.use('TkAgg')  # или 'Qt5Agg'

def draw_diagram_scattering(data_frame, y, label):
    # Построение диаграммы рассеяния для каждой пары факторов после удаления выбросов
    for column in data_frame.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data_frame, x=column, y=y, color='blue', label=f'{y.name}')

        # Подписываем каждую точку
        for i in range(data_frame.shape[0]):
            plt.annotate(label[i], (data_frame[column].iloc[i], y.iloc[i]),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center', fontsize=8)

        plt.title(f'Диаграмма рассеяния: {column} и {y.name}')
        plt.xlabel(column)
        plt.ylabel(y.name)
        plt.legend()
        plt.show()


def make_regression_with_one_parameter(X, y, isLogarifm, label):
    X_begin = X
    if isLogarifm:
        X = np.log(X)
        X = X.fillna(0)

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"Результаты регрессии для {y.name} от {', '.join(X_begin.columns[0])}:\n")
    print(model.summary())

    plt.scatter(X_begin, y, color='blue', label=f'{X_begin.columns[0]}')
    # Подписываем каждую точку
    for i in range(X_begin.shape[0]):
        plt.annotate(label[i], (X_begin.iloc[:,0].iloc[i], y.iloc[i]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center', fontsize=8)
    x_range = np.linspace(X_begin.min(), X_begin.max(), 100)

    x_value = x_range
    x_value_str = f"{X_begin.columns[0]}"
    if isLogarifm:
        x_value = np.log(x_range)
        x_value_str = f"ln({X_begin.columns[0]})"

    y_pred = model.params[0] + model.params[1] * x_value

    param_one = f" + {str(round(model.params[1], 2))}"
    if model.params[1] < 0:
        param_one = f" - {str(abs(round(model.params[1], 2)))}"

    plt.plot(x_range, y_pred, color='red', label=f'{str(round(model.params[0], 2)) + param_one + '*' + x_value_str}')
    plt.xlabel(f'{X_begin.columns[0]}')
    plt.ylabel(f'{y.name}')
    plt.title(f'Регрессия {y.name} от {X_begin.columns[0]}')
    plt.legend()
    plt.show()


def make_regression_with_many_parameter(X, y):
    X_begin = X
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"Результаты регрессии для {y.name} от {', '.join(X_begin.columns)}:\n")
    print(model.summary())

def calculate_log_return(value_prev, value_next):
    return np.log(value_next/value_prev)*100
def calculate_return(value_prev, value_next):
    return ((value_next - value_prev)/value_prev)*100

# Функция для удаления выбросов с использованием Z-оценки
def remove_outliers_zscore(data_frame, threshold=3):
    from scipy import stats
    z_scores = np.abs(stats.zscore(data_frame))
    data_frame = data_frame[(z_scores < threshold).all(axis=1)]
    print("\nДанные после удаления выбросов с использованием Z-оценки:")
    print(data_frame.describe())
    print(data_frame)
    return data_frame.reset_index(drop=True)

# Функция для удаления выбросов с использованием IQR
def remove_outliers_iqr(data_frame):
    for column in data_frame.columns:
        Q1 = data_frame[column].quantile(0.25)
        Q3 = data_frame[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_frame = data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]
    print("Данные после удаления выбросов с использованием IQR:")
    print(data_frame.describe())
    print(data_frame)
    return data_frame.reset_index(drop=True)


# Функция для удаления выбросов с помощью IsolationForest
def remove_outliers_isolation_forest(data_frame):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples=58, random_state=12)
    clf.fit(data_frame)

    data_frame['anomaly'] = clf.predict(data_frame)
    print("Датафрейм с аномалиями")
    print(data_frame)
    data_frame = data_frame[data_frame.anomaly == 1]
    data_frame = data_frame.drop(columns='anomaly').reset_index(drop=True)
    print("Очищенный датафрейм")
    print(data_frame)
    return data_frame

"""def draw_diagram_scattering(data_frame):
    # Построение диаграммы рассеяния для каждой пары факторов после удаления выбросов
    for column in data_frame.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data_frame, x=column, y='Asset_Return', color='blue', label='Данные')"""  