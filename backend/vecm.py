
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pathlib import Path
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank


mexico = "data//mexico.csv"
debt = "data//deuda.csv"
pib = "data//pib.csv"
interes = "data//interes.csv"
tipoc = "data//tipoc.csv"


def getData(country):

    #read data from csv files - all of them
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    debt_path = parent_dir / debt
    pib_path = parent_dir / pib
    interes_path = parent_dir / interes
    tipoc_path = parent_dir / tipoc
    dfDebt = pd.read_csv(debt_path)
    dfPib = pd.read_csv(pib_path)
    dfInteres = pd.read_csv(interes_path)
    dfTipoc = pd.read_csv(tipoc_path)

    #filter by country, filter between 1980 and 2024
    countryFilter = dfDebt[dfDebt['Country Code'] == country]
    selected_cols_debt = countryFilter.iloc[:, 23:68]

    countryFilter = dfPib[dfPib['Country Code'] == country]
    selected_cols_pib = countryFilter.iloc[:, 23:68]

    countryFilter = dfInteres[dfInteres['Country Code'] == country]
    selected_cols_interes = countryFilter.iloc[:, 23:68]

    countryFilter = dfTipoc[dfTipoc['Country Code'] == country]
    selected_cols_tipoc = countryFilter.iloc[:, 23:68]

    #concat dataframes into a single one containg debt, pib, interest rate and exchange rate
    df_all = pd.concat([selected_cols_debt, selected_cols_pib, selected_cols_interes, selected_cols_tipoc], ignore_index=True)

    #do tranpose to convert into 5 columns dataframe
    df_transposedDebt = df_all.T 
    df_transposedDebt.columns = ['deuda', 'pib', 'interes', 'tipoc']
    df_transposedDebt.insert(0, 'year', range(1980, len(df_transposedDebt) + 1980))
    
    #if Nan values found, replace them with the mean of each variable
    deuda_medio = df_transposedDebt['deuda'].mean()
    df_transposedDebt['deuda'] = df_transposedDebt['deuda'].fillna(deuda_medio) 
    pib_medio = df_transposedDebt['pib'].mean()
    df_transposedDebt['pib'] = df_transposedDebt['pib'].fillna(pib_medio)
    interes_medio = df_transposedDebt['interes'].mean()
    df_transposedDebt['interes'] = df_transposedDebt['interes'].fillna(interes_medio)
    tipoc_medio = df_transposedDebt['tipoc'].mean()
    df_transposedDebt['tipoc'] = df_transposedDebt['tipoc'].fillna(tipoc_medio)

    df_transposedDebt['interes'] = df_transposedDebt['interes'].clip(lower=0)
    return df_transposedDebt



def vecm():

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    debt_path = parent_dir / mexico

    #data = pd.read_csv(debt_path)
    df = getData('MEX')
    df = df.set_index("year")

    # Debt ratio
    df["debt_gdp"] = df["deuda"] / df["pib"]

    # Log transform exchange rate
    df["log_exr"] = np.log(df["tipoc"])

    data = df[["debt_gdp", "interes", "log_exr"]]

    print("----------------------------------- data head --------------------------------")
    print(data.head(45))
    

    
    lag_order = select_order(data, maxlags=5, deterministic="ci")
    lags = lag_order.selected_orders["aic"]

    coint_rank = select_coint_rank(
    data,
    det_order=0,
    k_ar_diff=lags,
    method="trace"
    )

    rank = coint_rank.rank

    vecm = VECM(
    data,
    k_ar_diff=lags,
    coint_rank=rank,
    deterministic="ci"
    )

    vecm_res = vecm.fit()

    steps = 2039 - data.index.max()

    forecast = vecm_res.predict(steps=steps)

    forecast_index = np.arange(data.index.max()+2, 2041)

    forecast_df = pd.DataFrame(
    forecast,
    index=forecast_index,
    columns=data.columns
    )

    #print(forecast_df.head(20))


    gdp_forecast = []

    last_gdp = df["pib"].iloc[-1]

    for i in range(steps):
        last_gdp = last_gdp * 1.025
        gdp_forecast.append(last_gdp)

    gdp_forecast = pd.Series(gdp_forecast, index=forecast_index)

    forecast_df["deuda"] = forecast_df["debt_gdp"] * gdp_forecast

    #print("Model stable:", vecm_res.is_stable())
    print("----------------------------------- vecm summary --------------------------------")
    #print(vecm_res.summary())



    print("----------------------------------- vecm forecast --------------------------------")
    print(forecast_df)


    


vecm();


