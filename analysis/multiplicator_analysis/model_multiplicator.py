import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

def create_model_pe():
    data = pd.read_excel(file)
    df = pd.DataFrame(data)

    df['g'] = np.log(df['g'])
    df = df.fillna(0)
    print(df)


    y_var = df['P/E']
    X_var = df[['g']]
    X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

    # Построение модели OLS``
    model = sm.OLS(y_var, X_var).fit()
    print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
    print(model.summary())

def create_model_pbv(file):
    data = pd.read_excel(file)
    df = pd.DataFrame(data)
    #df = df[['P/BV', 'ROE', 'g']]
    #df['P/BV'] = np.log(df['P/BV'])
    #df['ROE'] = np.log(df['ROE'])
    #df['g'] = np.log(df['g'])
    df = df.fillna(0)
    print(df)

    y_var = df['P/BV']
    X_var = df[['ROE']]
    X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

    # Построение модели OLS
    model = sm.OLS(y_var, X_var).fit()
    print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
    print(model.summary())

def create_model_ps(file):
    data = pd.read_excel(file)
    df = pd.DataFrame(data)
    df = df.fillna(0)
    print(df)

    y_var = df['P/S']
    X_var = df[['NPM']]
    X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

    # Построение модели OLS
    model = sm.OLS(y_var, X_var).fit()
    print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
    print(model.summary())

def create_model_ev_ebitda(file):
    data = pd.read_excel(file)
    df = pd.DataFrame(data)

    #df['A/EBITDA'] = np.log(df['A/EBITDA'])
    df = df.fillna(0)

    y_var = df['EV/EBITDA']
    X_var = df[['ROIC', 't', 'A/EBITDA']]
    X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

    # Построение модели OLS
    model = sm.OLS(y_var, X_var).fit()
    print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
    print(model.summary())

def create_model_peg(file):
    data = pd.read_excel(file)
    df = pd.DataFrame(data)

    df['g'] = np.log(df['g'])
    df = df.fillna(0)

    y_var = df['PEG']
    X_var = df[['g']]
    X_var = sm.add_constant(X_var)  # Добавляем константу для alpha

    # Построение модели OLS
    model = sm.OLS(y_var, X_var).fit()
    print(f"Результаты регрессии для {y_var.name} от {', '.join(X_var.columns)}:\n")
    print(model.summary())