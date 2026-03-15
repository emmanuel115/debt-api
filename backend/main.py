from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
import time
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
from fastapi.responses import PlainTextResponse
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

debt = "data//deuda.csv"
pib = "data//pib.csv"
interes = "data//interes.csv"
tipoc = "data//tipoc.csv"

class Item:
    def __init__(self, year: str, value: str):
        self.year = year
        self.value = value

def sci_to_text(num):
    #Converts a number in scientific notation to a plain string without exponent."
    #Handles integers, floats, and strings.
    try:
        num_float = float(num)
        temp = format(num_float, 'f').rstrip('0').rstrip('.') if '.' in format(num_float, 'f') else format(num_float, 'f')
        temp = temp.split('.', 1)[0]
        num = int(temp)
        num = f"{num:,}"
        return num
    except (ValueError, TypeError):
        raise ValueError("Input must be a number or numeric string.")
    
def item_to_dict(obj):
    if isinstance(obj, Item):
        return {
            "year": obj.year,
            "value": obj.value
        }
    raise TypeError(f"Type {type(obj)} not serializable")

def getCountryData(country):

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
    selected_cols_pib = countryFilter.iloc[:, 24:69]

    countryFilter = dfInteres[dfInteres['Country Code'] == country]
    selected_cols_interes = countryFilter.iloc[:, 23:68]

    countryFilter = dfTipoc[dfTipoc['Country Code'] == country]
    selected_cols_tipoc = countryFilter.iloc[:, 24:69]

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

    return df_transposedDebt

    

def sarimax(df):   
    
    df.set_index("year", inplace=True)
    
    # target Variable
    y = df["deuda"]
    #  independant Variables - Exogeneus
    X = df[["pib", "interes", "tipoc"]]
    
    # Fit SARIMAX model
    model = SARIMAX(y, exog=X, order=(3,1,1), seasonal_order=(0,0,0,0))
    results = model.fit(disp=False)

    future_years = list(range(2027, 2041))
    future_interes = []
    for i in range (0, 14):
        future_interes.append(6.6)
    
    future_pib = []
    for i in range(0, 14):
        future_pib.append(1.89e12)
    
    future_tipoc = []
    for i in range(0, 14):
        future_tipoc.append(22.4)
    
    future_exog = pd.DataFrame({
        "pib": [2.0e12 + i*5e10 for i in range(len(future_years))],   # hypothetical GDP growth,
        "interes": [5 + 0.1*i for i in range(len(future_years))],     # hypothetical interest rates,
        "tipoc": [18 + 0.05*i for i in range(len(future_years))]      # hypothetical exchange rate,
    }, index=future_years)
    
    forecast = results.get_forecast(steps=len(future_years), exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_valor = forecast_mean.apply(sci_to_text)

    #predicciones = pd.DataFrame({
      #  "year": future_years,
     #   "value": forecast_valor,
    #}),

    items_list = []
    for i in range(len(forecast_valor)):
        item = Item(str(future_years[i]), forecast_valor.iloc[i])
        items_list.append(item)

    json_str = json.dumps(items_list, default=item_to_dict, indent=4)
    
    return json_str


def vecm(df):

    print(type(df))
    df = df.set_index("year")
    # Debt ratio
    df["debt_gdp"] = df["deuda"] / df["pib"]

    # Log transform exchange rate
    df["log_exr"] = np.log(df["tipoc"])
    data = df[["debt_gdp", "interes", "log_exr"]]
    
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

    gdp_forecast = []
    last_gdp = df["pib"].iloc[-1]

    for i in range(steps):
        last_gdp = last_gdp * 1.025
        gdp_forecast.append(last_gdp)

    gdp_forecast = pd.Series(gdp_forecast, index=forecast_index)
    forecast_df["deuda"] = forecast_df["debt_gdp"] * gdp_forecast

    items_list = []

    for idx, row in forecast_df.iterrows():
        obj = Item(
            year=idx,
            value=row['deuda']
        )
        items_list.append(obj)

    return json.dumps(items_list, default=item_to_dict, indent=4)


def getPlt(country):

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    debt_path = parent_dir / debt
    pib_path = parent_dir / pib
    interes_path = parent_dir / interes
    dfDebt = pd.read_csv(debt_path)
    dfPib = pd.read_csv(pib_path)
    dfInteres = pd.read_csv(interes_path)

    #filter by country, filter between 1980 and 2024
    countryFilter = dfDebt[dfDebt['Country Code'] == country]
    selected_cols_debt = countryFilter.iloc[:, 23:68]

    countryFilter = dfPib[dfPib['Country Code'] == country]
    selected_cols_pib = countryFilter.iloc[:, 24:69]

    countryFilter = dfInteres[dfInteres['Country Code'] == country]
    selected_cols_interes = countryFilter.iloc[:, 23:68]

    #concat dataframes into a single one containg debt, pib, interest rate and exchange rate
    df_all = pd.concat([selected_cols_debt, selected_cols_pib, selected_cols_interes], ignore_index=True)

    #do tranpose to convert into 5 columns dataframe
    df_transposedDebt = df_all.T 
    df_transposedDebt.columns = ['deuda', 'pib', 'interes']
    df_transposedDebt.insert(0, 'year', range(1980, len(df_transposedDebt) + 1980))

    # 4. Limpiar datos (convertir Año a número y quitar nulos)
    df_transposedDebt['year'] = pd.to_numeric(df_transposedDebt['year'], errors='coerce')
    df_transposedDebt = df_transposedDebt.dropna()

    print(df_transposedDebt.head(40))


    print("--------------------------- Matriz correlacion ---------------------------------")
    # 2. Calcular la matriz de correlación (Pearson)
    matriz_corr = df_transposedDebt.corr()

    # 3. Configurar y dibujar el Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_corr, 
            annot=True,       # Muestra los números dentro de los cuadros
            cmap='RdBu_r',    # Escala de Rojo (positivo) a Azul (negativo)
            center=0,         # El blanco representa correlación cero
            fmt=".2f",        # Dos decimales
            linewidths=2,     # Espacio entre celdas
            square=True)      # Celdas cuadradas

    plt.title('Mapa de Correlación: Deuda, PIB y tasas de Interes', fontsize=14)
    
    return plt

@app.get("/api/debt/predictions/sarimax/{country}", response_class=PlainTextResponse)
def get_SarimaxPredictions(country):

    #measure processing time
    start_time = time.perf_counter()

    #Call SARIMAX function - function returns JSON string
    sarimaxPredictions = sarimax(getCountryData(country));

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SARIMAX Elapsed time: {elapsed:.6f} seconds")

    return sarimaxPredictions




@app.get("/api/debt/predictions/vecm/{country}", response_class=PlainTextResponse)
def get_VecmPredictions(country):

    #measure processing time
    start_time = time.perf_counter()

    #Call SARIMAX function - function returns JSON string
    vecmPredictions = vecm(getCountryData(country));

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"VECM Elapsed time: {elapsed:.6f} seconds")

    return vecmPredictions


@app.get("/api/debt/corrm/{country}")
def get_plot_json(country):

    plt = getPlt(country);
    try:
        # Create a simple Matplotlib plot
        #fig = plt.subplots()

        # Save plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Encode image to Base64 string
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Return JSON with Base64 image
        return JSONResponse(content={"imageBase64": img_base64})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#vecm(getCountryData('MEX'));
    
    




