

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/debt/predictions/{country}")
def get_data(country):

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    file_path = parent_dir / "data//deuda.csv"
    df = pd.read_csv(file_path)

    countryFilter = df[df['Country Code'] == country]
    selected_cols = countryFilter.iloc[:, 23:68] 

    #selected_cols.info()

    try:
        json_str = selected_cols.to_json(orient='records', indent=4, date_format='utf-8', default_handler=str)
    
        #print("JSON String Output:")
        #print(json_str)
    except Exception as e:
        print(f"Error converti ng DataFrame to JSON: {e}")


    #df.info()

#get_data('MEX');

    return json_str