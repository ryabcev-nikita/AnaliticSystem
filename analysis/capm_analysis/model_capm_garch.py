import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt

matplotlib.use('tkagg')
def calculate_return(value_prev, value_next):
    return np.log(value_next/value_prev)*100

data_asset = pd.read_csv('data/sber.csv', sep=';')
data_market = pd.read_csv('data/imoex.csv', sep=';')
data_asset = data_asset.dropna(axis=0, how='any')

print(data_asset.head())
print(data_market.head())

returns_asset = []
returns_market = []
for i in range(len(data_asset['<CLOSE>']) - 1):
    returns_asset.append(calculate_return(data_asset['<CLOSE>'][i], data_asset['<CLOSE>'][i+1]))

for i in range(len(data_market['<CLOSE>']) - 1):
    returns_market .append(calculate_return(data_market['<CLOSE>'][i], data_market['<CLOSE>'][i+1]))

returns_asset = pd.DataFrame(returns_asset)
returns_market = pd.DataFrame(returns_market)

print(returns_asset.head())
print(returns_market.head())

# Объединяем доходности в один DataFrame, выравниваем по датам
data = pd.concat([returns_asset, returns_market], axis=1).dropna()
data.columns = ['Asset_Return', 'Market_Return']

# Добавим безрисковую ставку (предположим 0 или достоверных данных нет)
risk_free_rate = 15.5

# Рассчитаем избыточные доходности (сверх безрисковой)
excess_asset_return = data['Asset_Return'] - risk_free_rate
excess_market_return = data['Market_Return'] - risk_free_rate

# CAPM регрессия: excess_asset_return = alpha + beta * excess_market_return + error
X = sm.add_constant(excess_market_return)
capm_model = sm.OLS(excess_asset_return, X).fit()
print("CAPM Model Summary:")
print(capm_model.summary())

# Получаем остатки (ошибки) от модели CAPM
residuals = capm_model.resid

# Подгоняем GARCH(1,1) модель к остаткам для моделирования условной дисперсии
garch_model = arch_model(residuals, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp='off')
print("\nGARCH(1,1) Model Summary:")
print(garch_fit.summary())

# Визуализация условной волатильности
plt.figure(figsize=(10,6))
plt.plot(garch_fit.conditional_volatility, color='blue', label='Условная волатильность (GARCH)')
plt.title('Условная волатильность доходностей акции по модели GARCH(1,1)')
plt.legend()
plt.show()

# Построение графика линейной регрессии с доверительным интервалом и предсказанием
# Предсказания с доверительным интервалом
pred = capm_model.get_prediction(X)
pred_summary = pred.summary_frame(alpha=0.05)  # 95% доверительный интервал
plt.figure(figsize=(10,6))
plt.scatter(excess_market_return, excess_asset_return, label='Фактические данные', color='gray', alpha=0.5)

# Линия предсказанной регрессии
plt.plot(excess_market_return, pred_summary['mean'], color='red', label='Линия регрессии')
# Доверительный интервал для среднего предсказания
plt.fill_between(excess_market_return,
                 pred_summary['mean_ci_lower'],
                 pred_summary['mean_ci_upper'],
                 color='red', alpha=0.2, label='Доверительный интервал среднего')

# Диапазон предсказания (prediction interval) для отдельных наблюдений
plt.fill_between(excess_market_return,
                 pred_summary['obs_ci_lower'],
                 pred_summary['obs_ci_upper'],
                 color='orange', alpha=0.2, label='Доверительный интервал предсказания')
plt.title('CAPM: регрессия доходности акции на доходность рынка')
plt.xlabel('Избыточная доходность рынка')
plt.ylabel('Избыточная доходность акции')
plt.legend()
plt.grid(True)
plt.show()

