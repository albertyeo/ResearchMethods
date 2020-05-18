import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from scipy.optimize import fsolve
import quandl
quandl.ApiConfig.api_key = "LVQspt3ywQv2x6hS_zMn"

# List of all CDSCOMP's files
CDS_list = os.listdir("CDSCOMP/")
CDS_list = [CDS_list[i] for i in range(len(CDS_list))]
CDS_list.sort()
CDS_list = CDS_list[1:]

# Extracting date from the filename
CDS_date = [CDS_list[i][-11:-4:1] for i in range(len(CDS_list))]

# Converting the date into datetime
CDS_date = [dt.datetime.strptime(i, '%d%b%y') for i in CDS_date]
CDS_date.sort()

# Converting the datetime into string of the desired format
CDS_dates = [i.strftime("%Y-%m-%d") for i in CDS_date]

# Importing the zero-rate from Quandl
Quandl_data = quandl.get("FED/SVENY")

# Filtering the required data for CDSCOMP's files
Quandl_df = Quandl_data[Quandl_data.index.isin(CDS_dates)]

# Replacing the column names from SVENYXX to XX.00
Quandl_df.columns = [format(i, '.2f') for i in np.arange(1, 31, 1)]

# Sorting the list of CDS based on chronological order
x = [i for i in CDS_date if i in Quandl_df.index]
CDS_list_sorted = ['V5 CDS Composites-'+str(i.strftime("%d%b%y"))+'.csv' for i in x]

# Creating a list of file names for Hazard Rates
Hazard_list = ["Hazard Rate-"+str(CDS_list_sorted[i][-11:-4:1])+".csv" for i in range(len(CDS_list_sorted))]

print(Quandl_df)

# Creating a DataFrame of zero-rates from timestep 0 to 30 years with a quarterly frequency
month_list = [format(i, '.2f') for i in np.arange(0.0, 30.25, 0.25)]
rate_df = pd.DataFrame(Quandl_df, index = Quandl_df.index, columns = month_list)

# Converting the columns into floats
rate_df.columns = np.arange(0.0, 30.25, 0.25)

# Interpolating the rates
rate_df[0.00] = 0
rate_df = rate_df / 100
rate_df = rate_df.interpolate(axis = 1)

print(rate_df)

# Creating a DataFrame of Discount Factors from the zero-rates
B_df = pd.DataFrame(rate_df, index = rate_df.index, columns = rate_df.columns)

for i in B_df.columns:
    if i <= 1.00:
        B_df[i] = 1 / (1 + B_df[i].values * i)
    elif i > 1.00:
        B_df[i] = 1 / ((1 + B_df[i].values) ** i)
        
print(B_df)

# List of columns to be imported from CDSCOMP's files
column_list = ["Ticker", "Ccy", "DocClause", "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", "Spread5y", 
               "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y", "Recovery"]

# List of tickers of most traded CDS
ticker_list = ["SVU", "DNY", "GT", "XEL-NRGInc", "THC", "CHK", "BBY", 
               "CVC", "DNSFDS", "AMD", "HRB", "JCP", "PBI", "AES", "OI"]

# Importing all CDSCOMP files into a list
CDSCOMP_df = [pd.read_csv("CDSCOMP/"+i, usecols = column_list, index_col = 0, header = 1) 
              for i in CDS_list_sorted]

# Creating a list of DataFrame for Discount Factors to match the date of CDS file
B = [B_df.iloc[x, :] for x in range(len(B_df.index))]

# Cleaning CDS data
for i in range(len(CDSCOMP_df)):
    # Filtering the data based on DocClause = XR14 & Currency = USD
    CDSCOMP_df[i] = CDSCOMP_df[i][CDSCOMP_df[i]["DocClause"] == "XR14"]
    CDSCOMP_df[i] = CDSCOMP_df[i][CDSCOMP_df[i]["Ccy"] == "USD"]
    
    # Creating a list of tickers available in one CDS file
    tickers = [ticker_list[j] for j in range(len(ticker_list)) if ticker_list[j] in CDSCOMP_df[i].index]

    # Taking the selected ticker data
    CDSCOMP_df[i] = CDSCOMP_df[i].loc[tickers, :]
    
    # Taking Spread and Recovery data
    CDSCOMP_df[i] = CDSCOMP_df[i].loc[:, "Spread6m":"Recovery"]
    
    # Changing the Spread column name from str to float
    CDSCOMP_df[i].columns = [0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00, 30.00, "Recovery"]
    
    # Dropping data if both Spread6m and Spread1y are missing
    CDSCOMP_df[i].dropna(axis = 0, how = "all", subset = [0.50, 1.00], inplace = True)
    
    # Dropping data if Recovery is missing
    CDSCOMP_df[i].dropna(axis = 0, how = "all", subset = ["Recovery"], inplace = True)
    
    # Converting the data into float
    CDSCOMP_df[i] = CDSCOMP_df[i].apply(lambda x: pd.to_numeric(x.str.strip(".%")), axis = 1) / 100
    
    # Creating a DataFrame of CDS from timestep 0.25 to 30 years with a quarterly frequency & Recovery
    recovery_value = CDSCOMP_df[i]["Recovery"].values
    CDSCOMP_df[i] = pd.DataFrame(CDSCOMP_df[i], index = CDSCOMP_df[i].index, columns = np.arange(0.25, 30.25, 0.25))
    CDSCOMP_df[i]["Recovery"] = recovery_value
    
# Creating a list of DataFrame for Hazard Rates
lambda_df = [pd.DataFrame(np.NaN, index = CDSCOMP_df[i].index, columns = np.arange(0.0, 30.25, 0.25))
             for i in range(len(CDSCOMP_df))]
             
# Defining a function to bootstrap the first Hazard Rates
def firstHazardRate(CDS, B, i, row, row_year, x):
    # CDS = List of DataFrame containing the CDS Spreads
    # B = List of Series of Discount Factors
    # i = ith CDS file
    # row = row index of one CDS file
    # row_year = Time period with available Spread data
    # y = column index to be solved
    # x = Hazard Rate
    
    premium_list = []
    protection_list = []
    year = row_year[0]
    
    count = 1
    for t in np.arange(0.00 + 0.25, row_year[0] + 0.25, 0.25):
        premium = np.exp(-0.25 * x * count) * B[i][t]
        protection = (np.exp(-0.25 * x * (count - 1)) - 
                      np.exp(-0.25 * x * count)) * B[i][t]
        premium_list.append(premium)
        protection_list.append(protection)
        count = count + 1

    PV_premium = CDS[i][year][row] * 0.25 * np.sum(premium_list)
    PV_protection = (1 - CDS[i]["Recovery"][row]) * np.sum(protection_list)
    # Equation to be solved at breakeven point
    equation = PV_premium - PV_protection
    return equation
    
# Defining a function to bootstrap the Hazard Rates
def HazardRate(CDS, B, HR, i, row, row_year, y, x):
    # CDS = List of DataFrame containing the CDS Spreads
    # B = List of Series of Discount Factors
    # HR = List of DataFrame of Hazard Rates
    # i = ith CDS file
    # row = row index of one CDS file
    # row_year = Time period with available Spread data
    # y = column index to be solved
    # x = Hazard Rate
    
    premium_list = []
    protection_list = []
    year = row_year[y]
    
    # Calculating the sum & cumsum of the Hazard Rates from time 0.25 to the first available Spread timestep
    lambda_sum = np.sum(HR[i].iloc[row, 1:int(row_year[y-1] * 4)+1].values)
    lambda_cumsum = np.cumsum(HR[i].iloc[row, 1:int(row_year[y-1] * 4)+1].values)
    
    # Calculating the premium & protection from time 0.25 to the first available Spread timestep
    premium_cum = np.dot(np.exp(-0.25 * lambda_cumsum), B[i][0.25:row_year[y-1]].values)
    protection_cum = np.dot((np.exp(-0.25 * np.insert(lambda_cumsum[0:-1], 0, 0)) -
                             np.exp(-0.25 * lambda_cumsum)), B[i][0.25:row_year[y-1]].values)

    premium_list.append(premium_cum)
    protection_list.append(protection_cum)

    count = 1
    for t in np.arange(row_year[y-1] + 0.25, row_year[y] + 0.25, 0.25):
        premium = np.exp(-0.25 * (x * count + lambda_sum)) * B[i][t]
        protection = (np.exp(-0.25 * (x * (count - 1) + lambda_sum)) - 
                      np.exp(-0.25 * (x * count + lambda_sum))) * B[i][t]
        premium_list.append(premium)
        protection_list.append(protection)
        count = count + 1 

    PV_premium = CDS[i][year][row] * 0.25 * np.sum(premium_list)
    PV_protection = (1 - CDS[i]["Recovery"][row]) * np.sum(protection_list)
    # Equation to be solved at breakeven point
    equation = PV_premium - PV_protection
    return equation
    
# Iterating through the CDS files
for i in range(len(CDSCOMP_df)):  
    # Iterating through the rows of one CDS file
    for row in range(len(CDSCOMP_df[i].index)):
        # row_year = List of time period with available Spread data of one CDS file
        row_year = []
        
        # Iterating through the columns of one CDS file
        for col in CDSCOMP_df[i].columns[0:-1:]:
            if np.isnan(CDSCOMP_df[i][col][row]):
                None
            else:
                row_year.append(col)
        
        lambda_df[i].iloc[row, int(row_year[0] * 4)] = fsolve(lambda x: firstHazardRate(CDSCOMP_df, B, i, row, row_year, x),0.5)[0]
        lambda_df[i].iloc[row, 0:int(row_year[0] * 4)] = lambda_df[i].iloc[row, int(row_year[0] * 4)]
        
        for y in range(1, len(row_year)):
            initial_guess = lambda_df[i].iloc[row, int(row_year[y-1] * 4)]
            lambda_df[i].iloc[row, int(row_year[y] * 4)] = fsolve(lambda x: HazardRate(CDSCOMP_df, B, lambda_df, i, row, row_year, y, x), initial_guess)[0]
            lambda_df[i].iloc[row, int(row_year[y-1] * 4)+1:int(row_year[y] * 4)] = lambda_df[i].iloc[row, int(row_year[y] * 4)]      
        
        lambda_df[i] = lambda_df[i].ffill(axis = 1)
        
        # Saving the hazard rate results into .csv format
        lambda_df[i].to_csv("Hazard Rates/"+Hazard_list[i], header=True)
