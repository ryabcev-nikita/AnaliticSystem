import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import seaborn as sns

import matplotlib

matplotlib.use('TkAgg')  # или 'Qt5Agg'

# Загрузка данных
data = pd.read_excel('data/data_surgut.xlsx')
dff = pd.DataFrame(data)
df = dff.iloc[:, 5:dff.shape[1]]
df = df.dropna()
df_begin = df

print("Предварительные загруженные данные")
print(df)

n = len(df)  # количество наблюдений
k = df.shape[1] # количество факторов
ddf = n - k - 1  # Степени свободы
alpha = 0.05 # уровень значимости

risk_free_rate = 14.98

# Рассчитаем избыточные доходности (сверх безрисковой)
df['R_stock'] = df['R_stock'] - risk_free_rate
df['R_imoex'] = df['R_imoex'] - risk_free_rate
df['R_usd'] = df['R_usd'] - risk_free_rate
df['R_oil'] = df['R_oil'] - risk_free_rate
df['Inflation'] = df['Inflation'] - risk_free_rate
df['IPP'] = df['IPP'] - risk_free_rate

y_var = df['R_stock']
X_var = df[['R_imoex', 'R_usd', 'R_oil', 'Inflation', 'IPP']]
X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

model = sm.OLS(y_var, X_var).fit()
print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
print(model.summary())
print("\n" + "=" * 80 + "\n")