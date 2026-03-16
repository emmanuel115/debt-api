import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def getCorrUniCountryData(country, indicator):

    debt = "data//deuda.csv"
    pobreza = "data//" + indicator + ".csv"


    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    debt_path = parent_dir / debt
    pib_path = parent_dir / pobreza
    
    
    dfDebt = pd.read_csv(debt_path)
    dfPobreza = pd.read_csv(pib_path)
    
    min = 0
    initiYear = 1980

    yearsD = any;
    yearsP = any;
    if(indicator == "tugurios" or
       indicator == "consumomedio"):
        yearsD = np.r_[1,43:66].copy()
        yearsP = np.r_[1,42:65].copy()
        min = 23
        initiYear = 2000
    else:
        yearsD = np.r_[1,23:68].copy()
        yearsP = np.r_[1,24:69].copy()
        initiYear = 1980
        min = 45


    #filter by country, filter between 1980 and 2024
    countryFilter = dfDebt[dfDebt['Country Code'] == country]
    selected_cols_debt = countryFilter.iloc[:, yearsD]

    countryFilter = dfPobreza[dfPobreza['Country Code'] == country]
    selected_cols_pobreza = countryFilter.iloc[:, yearsP]

    print(len(selected_cols_debt.columns))
    print(len(selected_cols_pobreza.columns))

    selected_cols_debt = selected_cols_debt.drop('Country Code', axis=1)
    selected_cols_pobreza = selected_cols_pobreza.drop('Country Code', axis=1)

    #dfT1 = selected_cols_debt.T
    #print(dfT1.head(50))

    #print(selected_cols_debt)
    #print(selected_cols_pobreza)

    if(len(selected_cols_debt.columns) != min or
        len(selected_cols_debt.columns) == 0 or 
       len(selected_cols_pobreza.columns) == 0 or 
       (len(selected_cols_pobreza.columns) != len(selected_cols_debt.columns))):
        print("no data")
        return "no_data_found";

    #concat dataframes into a single one containg debt, pib, interest rate and exchange rate
    df_all = pd.concat([selected_cols_debt, selected_cols_pobreza], ignore_index=True)

    #print(df_all)
    

    #do tranpose to convert into 5 columns dataframe
    df_transposedDebt = df_all.T 
    #print(df_transposedDebt)
    df_transposedDebt.columns = ['deuda', 'pobreza']
    
    #df_transposedDebt = df_transposedDebt.drop('Country Code', axis=1)
    #print(df_transposedDebt)
    df_transposedDebt.insert(0, 'year', range(initiYear, len(df_transposedDebt) + initiYear))

    #print(df_transposedDebt.head(50))

    

    null_percentage = df_transposedDebt.isnull().mean() * 100
    #print("Percentage of null values per column:")
    null_percentage = null_percentage.round(2)
    #print(type(null_percentage))
    #print(null_percentage)
    for value in null_percentage:
        if(value > 60):
            return "mo_data_low_per";
        
    
    
    #if Nan values found, replace them with the mean of each variable
    deuda_medio = df_transposedDebt['deuda'].mean()
    df_transposedDebt['deuda'] = df_transposedDebt['deuda'].fillna(deuda_medio) 
    pobreza_medio = df_transposedDebt['pobreza'].mean()
    df_transposedDebt['pobreza'] = df_transposedDebt['pobreza'].fillna(pobreza_medio)

    #print(df_transposedDebt.head(40))
    return df_transposedDebt











def seabornPlot(df):

    sns.regplot(x=df['deuda'], y=df['pobreza'])
    plt.title("Correlation between column1 and column2")
    plt.show()


def scatterPlot(df):
    plt.scatter(df['deuda'], df['pobreza'])
    plt.xlabel('deuda')
    plt.ylabel('pobreza')
    plt.title('Correlation between deuda and column2')
    plt.show()


def plotCorrelation(df):

    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation calculation.")
    
    # Calculate correlation matrix (Pearson by default)
    pearson_corr = numeric_df.corr(method='pearson')
    kendall_corr = numeric_df.corr(method='kendall')
    spearman_corr = numeric_df.corr(method='spearman')

    # Display results
    print("DataFrame:")
    print(df, "\n")

    print("Pearson Correlation:\n", pearson_corr, "\n")
    print("Kendall Correlation:\n", kendall_corr, "\n")
    print("Spearman Correlation:\n", spearman_corr)

    plt.scatter(
        df['deuda'],
        df['pobreza'],
        color='blue',
        alpha=0.6,
        s=40
    )
    plt.xlabel('Deuda')
    plt.ylabel('Pobreza')
    plt.title('Deuda vs Pobreza')
    plt.show()


def regPlot(df):
    sns.regplot(
        x='deuda',
        y='pobreza',
        data=df
    )

    plt.title('Deuda Internacional vs Pobreza')
    plt.xlabel('Deuda')
    plt.ylabel('Pobreza')
    plt.show()



country = 'ARG'
indicator = "gini"
    
result = getCorrUniCountryData(country, indicator);
if (isinstance(result, str)):
    print(result)
else:
    print(result.head(20))
    print("-------------------------- PLOT -----------------------------")
    #seabornPlot(result)
    #plotCorrelation(result)
    regPlot(result)
    


#df = pd.read_csv(debt_path)
#non_null_counts = df.count()

#print("Non-null counts per column:")
#print(non_null_counts)

#df = pd.read_csv(pobreza_path)
#non_null_counts = df.count()

#print("Non-null counts per column:")

#print(non_null_counts)

