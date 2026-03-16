import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pathlib import Path
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

def getPlt():

    debt = "data//deuda.csv"
    pib    = "data//pib.csv"
    interes = "data//interes.csv"

    country = 'MEX'

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

    plt.title('Mapa de Correlación: PIB, Deuda y Tasas en México', fontsize=14)
    
    return plt





@app.get("/plot-json")
def get_plot_json():

    plt = getPlt();
    try:
        # Create a simple Matplotlib plot
        fig = plt.subplots()

        # Save plot to a BytesIO buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)  # Free memory
        buf.seek(0)

        # Encode image to Base64 string
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Return JSON with Base64 image
        return JSONResponse(content={"image_base64": img_base64})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))