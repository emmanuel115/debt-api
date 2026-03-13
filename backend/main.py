from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
from fastapi.responses import JSONResponse
import time
from backend import debt


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

class DataItem(BaseModel):
    year: str
    value: str

class Data2(BaseModel):
    year: str
    deuda: str
    pib: str


@app.get("/api/debt/predictions/{country}", response_model=List[DataItem])
def get_data(country):

    start_time = time.perf_counter()

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

    countryFilter = dfDebt[dfDebt['Country Code'] == country]
    selected_cols_debt = countryFilter.iloc[:, 23:68]

    countryFilter = dfPib[dfPib['Country Code'] == country]
    selected_cols_pib = countryFilter.iloc[:, 23:68]

    countryFilter = dfInteres[dfInteres['Country Code'] == country]
    selected_cols_interes = countryFilter.iloc[:, 23:68]

    countryFilter = dfTipoc[dfTipoc['Country Code'] == country]
    selected_cols_tipoc = countryFilter.iloc[:, 23:68]


    #df_transposedDebt = selected_cols_debt.T  # or df.transpose()
    #df_transposedDebt.columns = ['year', 'debt']

    #print(df_transposedDebt.info())
    #result = pd.merge(selected_cols_debt, selected_cols_pib)

    #print(result.info())





    df_all = pd.concat([selected_cols_debt, selected_cols_pib, selected_cols_interes, selected_cols_tipoc], ignore_index=True)
    #print(df_all.head(2))

    df_transposedDebt = df_all.T 
    df_transposedDebt.columns = ['deuda', 'pib', 'interes', 'tipoc']
    df_transposedDebt.insert(0, 'year', range(1980, len(df_transposedDebt) + 1980))
    #print(df_transposedDebt.columns)
    
    

    

    deuda_medio = df_transposedDebt['deuda'].mean()
    df_transposedDebt['deuda'] = df_transposedDebt['deuda'].fillna(deuda_medio) 
    pib_medio = df_transposedDebt['pib'].mean()
    df_transposedDebt['pib'] = df_transposedDebt['pib'].fillna(pib_medio)
    interes_medio = df_transposedDebt['interes'].mean()
    df_transposedDebt['interes'] = df_transposedDebt['interes'].fillna(interes_medio)
    tipoc_medio = df_transposedDebt['tipoc'].mean()
    df_transposedDebt['tipoc'] = df_transposedDebt['tipoc'].fillna(tipoc_medio)


    #print(df_transposedDebt.head(30))

     #df_filled = df.fillna(0)   
    predicciones = sarimax(df_transposedDebt);


    debtItemsList = []

    #for col_name, col_data in result.items():
        #data = col_data.to_list()[0]
        #debtItemsList.append(DataItem(year=col_name, value=data))


    for prediccion in predicciones:
        debtItemsList.append(DataItem(year=prediccion.year.to_string(index=False), value=prediccion.value.to_string(index=False)))      

    #print(type(debtItemsList))

    #for item in debtItemsList:
        #print(item)
        #print(" - ")
        #print(item.value)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.6f} seconds")


#get_data('MEX');
    
    return debtItemsList