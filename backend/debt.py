import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

def sarimax(df):   
    
    #current_dir = Path(__file__).resolve().parent
    #parent_dir = current_dir.parent
    #file_path = parent_dir / "data//mexico.csv"
    #df = pd.read_csv(file_path)
    df.set_index("year", inplace=True)
    
    # Variable objetivo
    y = df["deuda"]
    # Variables independientes - Exogenas
    X = df[["pib", "interes", "tipoc"]]
    
    # Entrenar el modelo SARIMAX
    model = SARIMAX(y, exog=X, order=(3,1,1), seasonal_order=(0,0,0,0))
    results = model.fit(disp=False)
    
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print(results.summary())
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------------------------------------------")


    future_years = list(range(2027, 2041))
    #print("tamano")
    #print(len(future_years) )
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

    #print(future_exog)

    future_exog2 = pd.DataFrame({
        "pib": future_pib,   # hipotetico,
        "interes": future_interes,     # hipotetico,
        "tipoc": future_tipoc      # hipotetico,
    }, index=future_years)

    #print(future_exog2)
    
    forecast = results.get_forecast(steps=len(future_years), exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_valor = forecast_mean.apply(sci_to_text)
    predicciones = pd.DataFrame({
        "year": future_years,
        "value": forecast_valor,
    }),
    #print("Predicciones del modelo")
    #print(predicciones)
    print("-----------------------------------------------------------------")
    #for item in predicciones:
        #print(type(item.year))
        #print(" - ")
        #print(type(item.value))
    #print(type(predicciones))

    #columns = ["year", "value"]
    #result = pd.DataFrame(predicciones, columns=columns)
    return predicciones

#sarimax()