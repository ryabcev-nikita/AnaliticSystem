import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from shared.utils import calculate_log_return, calculate_return, remove_outliers_zscore

def create_model_capm(free_rate, file_asset, file_market):
    matplotlib.use('tkagg')

    data_asset = pd.read_csv(file_asset, sep=';')
    data_market = pd.read_csv(file_market, sep=';')
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
    risk_free_rate = free_rate

    # Рассчитаем избыточные доходности (сверх безрисковой)
    excess_asset_return = data['Asset_Return'] - risk_free_rate
    excess_market_return = data['Market_Return'] - risk_free_rate

    # CAPM регрессия: excess_asset_return = alpha + beta * excess_market_return + error
    X = sm.add_constant(excess_market_return)
    capm_model = sm.OLS(excess_asset_return, X).fit()
    print("CAPM Model Summary:")
    print(capm_model.summary())

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
    plt.title('CAPM: регрессия доходности акции от доходность рынка')
    plt.xlabel('Избыточная доходность рынка')
    plt.ylabel('Избыточная доходность акции')
    plt.legend()
    plt.grid(True)
    plt.show()