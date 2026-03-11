import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax():   
    
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    file_path = parent_dir / "data//mexico.csv"
    df = pd.read_csv(file_path)
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

sarimax()