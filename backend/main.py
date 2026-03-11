from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np
from fastapi.responses import JSONResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebtItem(BaseModel):
    year: str
    value: str


@app.get("/api/debt/predictions/{country}", response_model=List[DebtItem])
def get_data(country):

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    file_path = parent_dir / "data//deuda.csv"
    df = pd.read_csv(file_path)

    countryFilter = df[df['Country Code'] == country]
    selected_cols = countryFilter.iloc[:, 23:68] 
    itemsList = []

    for col_name, col_data in selected_cols.items():
        data = col_data.to_list()[0]
        itemsList.append(DebtItem(year=col_name, value=data))

#get_data('MEX');

    
    return itemsList