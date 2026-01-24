import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import shared.utils

def create_model_capm_ff(free_rate, file_asset, file_market, file_big_cap, file_small_cap, file_stock_value, file_stock_growth):
    matplotlib.use('tkagg')

    data_asset = pd.read_csv(file_asset, sep=';')
    data_market = pd.read_csv(file_market, sep=';')
    data_asset = data_asset.dropna(axis=0, how='any')

    print(data_asset.head())
    print(data_market.head())

    # Рассчёт логарифмических доходностей
    returns_asset = []
    returns_market = []
    for i in range(len(data_asset['<CLOSE>']) - 1):
        returns_asset.append(utils.calculate_log_return(data_asset['<CLOSE>'][i], data_asset['<CLOSE>'][i+1]))

    for i in range(len(data_market['<CLOSE>']) - 1):
        returns_market.append(utils.calculate_log_return(data_market['<CLOSE>'][i], data_market['<CLOSE>'][i+1]))

    returns_asset = pd.DataFrame(returns_asset)
    returns_market = pd.DataFrame(returns_market)

    print(returns_asset.head())
    print(returns_market.head())

    data_big_cap = pd.read_csv(file_big_cap, sep=';')
    data_small_cap = pd.read_csv(file_small_cap, sep=';')
    data_stock_value = pd.read_csv(file_stock_value, sep=';')
    data_stock_growth = pd.read_csv(file_stock_growth, sep=';')

    data_big_cap = data_big_cap.dropna(axis=0, how='any')
    data_small_cap = data_small_cap.dropna(axis=0, how='any')
    data_stock_value = data_stock_value.dropna(axis=0, how='any')
    data_stock_growth = data_stock_growth.dropna(axis=0, how='any')

    returns_stock_growth = []
    returns_stock_value = []
    for i in range(len(data_stock_growth['<CLOSE>']) - 1):
        returns_stock_growth.append(utils.calculate_return(data_stock_growth['<CLOSE>'][i], data_stock_growth['<CLOSE>'][i+1]))

    for i in range(len(data_stock_value['<CLOSE>']) - 1):
        returns_stock_value.append(utils.calculate_return(data_stock_value['<CLOSE>'][i], data_stock_value['<CLOSE>'][i+1]))

    returns_hml = []
    for i in range(len(returns_stock_growth)):
        returns_hml.append(returns_stock_growth[i] - returns_stock_value[i])

    returns_hml = pd.DataFrame(returns_hml)

    returns_stock_small = []
    returns_stock_big = []
    for i in range(len(data_big_cap['<CLOSE>']) - 1):
        returns_stock_big.append(utils.calculate_return(data_big_cap['<CLOSE>'][i], data_big_cap['<CLOSE>'][i+1]))

    for i in range(len(data_small_cap['<CLOSE>']) - 1):
        returns_stock_small.append(utils.calculate_return(data_small_cap['<CLOSE>'][i], data_small_cap['<CLOSE>'][i+1]))

    returns_smb = []
    for i in range(len(returns_stock_big)):
        returns_smb.append(returns_stock_big[i] - returns_stock_small[i])

    returns_smb = pd.DataFrame(returns_smb)

    # Объединяем доходности в один DataFrame, выравниваем по датам
    data_begin = pd.concat([returns_asset, returns_market, returns_hml, returns_smb], axis=1).dropna()
    data_begin.columns = ['Asset_Return', 'Market_Return', 'Return_HML', 'Return_SMB']

    print(data_begin)

    # Удаление выбросов с использованием Z-оценки
    data_no_outliers = utils.remove_outliers_zscore(data_begin.copy())
    #data = data_no_outliers.copy()
    data = data_begin

    utils.draw_diagram_scattering(data)

    # Добавим безрисковую ставку
    risk_free_rate = free_rate

    # Рассчитаем избыточные доходности (сверх безрисковой)
    excess_asset_return = data['Asset_Return'] - risk_free_rate
    data['Market_Return'] = data['Market_Return'] - risk_free_rate

    # CAPM регрессия Fama-French: excess_asset_return = alpha + beta_market * excess_market_return + beta_hml * Return_HML + beta_smb * Return_SMB + error
    X = data[['Market_Return', 'Return_HML', 'Return_SMB']]
    X = sm.add_constant(X)  # Добавляем константу для alpha

    capm_model = sm.OLS(excess_asset_return, X).fit()
    print("CAPM Model Summary:")
    print(capm_model.summary())