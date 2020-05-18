import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import datetime as dt
import numpy.linalg as lin
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.stats import t
from scipy.stats import chi2
from statsmodels.sandbox.regression import gmm
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

# Taking datetime from 2014-01-01 to 2014-12-31
CDS_dates1 = CDS_dates[-1261:-1000]

# Taking datetime from 2013-12-31 to 2016-02-24
CDS_dates2 = CDS_dates[-1262:-700]

# Importing the zero-rate from Quandl
Quandl_data = quandl.get("FED/SVENY")

# Filtering the required data for CDSCOMP's files
Quandl_df = Quandl_data[Quandl_data.index.isin(CDS_dates1)]

# Replacing the column names from SVENYXX to XX.00
Quandl_df.columns = [format(i, '.2f') for i in np.arange(1, 31, 1)]

# Filtering the required data for HRB's files
Quandl_df2 = Quandl_data[Quandl_data.index.isin(CDS_dates2)]

# Replacing the column names from SVENYXX to XX.00
Quandl_df2.columns = [format(i, '.2f') for i in np.arange(1, 31, 1)]

# Creating a DataFrame of zero-rates from timestep 0 to 30 years with a quarterly frequency
month_list = [format(i, '.2f') for i in np.arange(0.0, 30.25, 0.25)]
rate_df = pd.DataFrame(Quandl_df, index = Quandl_df.index, columns = month_list)

# Converting the columns into floats
rate_df.columns = np.arange(0.0, 30.25, 0.25)

# Interpolating the rates
rate_df[0.00] = 0
rate_df = rate_df / 100
rate_df = rate_df.interpolate(axis = 1)

# Creating a DataFrame of zero-rates from timestep 0 to 30 years with a quarterly frequency
rate_df2 = pd.DataFrame(Quandl_df2, index = Quandl_df2.index, columns = month_list)

# Converting the columns into floats
rate_df2.columns = np.arange(0.0, 30.25, 0.25)

# Interpolating the rates
rate_df2[0.00] = 0
rate_df2 = rate_df2 / 100
rate_df2 = rate_df2.interpolate(axis = 1)

# Creating a DataFrame of Discount Factors from the zero-rates
B_df = pd.DataFrame(rate_df.copy(), index = rate_df.index, columns = rate_df.columns)

for i in B_df.columns:
    if i <= 1.00:
        B_df[i] = 1 / (1 + B_df[i].values * i)
    elif i > 1.00:
        B_df[i] = 1 / ((1 + B_df[i].values) ** i)
        
# Filtering CDS_date based on Quandl_df
x = [i for i in CDS_date if i in Quandl_df.index]

# Sorting the list of CDS based on chronological order
CDS_list_sorted = ['V5 CDS Composites-'+str(i.strftime("%d%b%y"))+'.csv' for i in x]

# List of columns to be imported from CDSCOMP's files
column_list = ["Ticker", "Ccy", "DocClause", "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", "Spread5y", 
               "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y", "Recovery"]

ticker_list = ["HRB"]

# Importing all CDSCOMP files into a list
CDSCOMP_df = [pd.read_csv("CDSCOMP/"+i, usecols = column_list, index_col = 0, header = 1) 
              for i in CDS_list_sorted]
              
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
    
# Concatenating CDCSOMP_df into one DataFrame
CDSCOMP_df = pd.concat(CDSCOMP_df)

# Setting date as CDSCOMP_df index
CDSCOMP_df.index = Quandl_df.index

# Importing HRB stock data
HRB_df = pd.read_csv("HRB.csv", usecols = ["date", "days", "divd", "Close"], header = 0, index_col = 0)

# Taking the 365-day volatility
HRB_df = HRB_df.loc[HRB_df["days"] == 365, :]

# Converting the date into datetime
HRB_df.index = [dt.datetime.strptime(str(i), '%Y%m%d') for i in HRB_df.index]

# Filtering HRB_df based on Quandl_df
HRB_df = HRB_df[HRB_df.index.isin(Quandl_df2.index)]

# Calculating dividend yield
HRB_df["div_yield"] = HRB_df['divd'] * 4 / HRB_df['Close']

# Creating a rate column
HRB_df["rate"] = rate_df2[1.00]

# Creating price & price lag columns
HRB_df["price"] = HRB_df["Close"].values
HRB_df["price_lag"] = HRB_df["price"].shift(1)

# Calculating diff in ln S
HRB_df["price_diff"] = HRB_df["price"] - HRB_df["price_lag"]

# Creating date & date lag columns
HRB_df["date"] = HRB_df.index
HRB_df["date_lag"] = HRB_df["date"].shift(1)

# Calculating date difference (in days)
HRB_df["delta_t"] = HRB_df["date"] - HRB_df["date_lag"]

# Converting delta t into year (days/365)
HRB_df["delta_t"] = HRB_df["delta_t"].apply(lambda x: x.days/365)

# Creating parameters columns
HRB_df["chi_squared"] = np.NaN
HRB_df["p_value"] = np.NaN
HRB_df["b"] = np.NaN
HRB_df["c"] = np.NaN
HRB_df["Beta"] = np.NaN

# Taking data from 2014-01-02 onwards and the relevant columns
HRB_df = HRB_df.loc[HRB_df.index[1:], ["price_diff", "price", "price_lag", "div_yield", "rate", "delta_t", 
                                       "chi_squared", "p_value", "b", "c", "Beta"]]
                                       
# Creating a DataFrame for parameter a
a_df = pd.DataFrame(np.NaN, index = CDSCOMP_df.index, columns = np.arange(0.0, 30.25, 0.25))

# Creating a DataFrame for Hazard Rates
lambda_df = pd.DataFrame(np.NaN, index = CDSCOMP_df.index, columns = np.arange(0.0, 30.25, 0.25))

# Creating a DataFrame for t-stats of the parameters
t_df = pd.DataFrame(np.NaN, index = HRB_df.index, columns = ["a", "b", "c", "Beta"])

# Creating a DataFrame for p-value of the parameters
p_df = pd.DataFrame(np.NaN, index = HRB_df.index, columns = ["a", "b", "c", "Beta"])

# Defining a function to bootstrap the first Hazard Rates
def firstHazardRate(CDS, B, stock, row, row_year, a, b, c, Beta):
    # CDS = DataFrame containing the CDS Spreads
    # B = DataFrame of Discount Factors
    # stock = DataFrame of Stock Data
    # row = row index of one CDS file
    # row_year = Time period with available Spread data
    # y = column index to be solved
    # x = Hazard Rate
    
    premium_list = []
    protection_list = []
    year = row_year[0]
    
    count = 1
    for t in np.arange(0.00 + 0.25, row_year[0] + 0.25, 0.25):
        lambda_est = b + (c * a**2 * np.power(stock.loc[stock.index[row], "price"], 2 * Beta))
        premium = np.exp(-0.25 * lambda_est * count) * B.loc[B.index[row], t]
        protection = (np.exp(-0.25 * lambda_est * (count - 1)) - 
                      np.exp(-0.25 * lambda_est * count)) * B.loc[B.index[row], t]
        premium_list.append(premium)
        protection_list.append(protection)
        count = count + 1

    PV_premium = CDS.loc[CDS.index[row], year] * 0.25 * np.sum(premium_list)
    PV_protection = (1 - CDS.loc[CDS.index[row], "Recovery"]) * np.sum(protection_list)
    # Equation to be solved at breakeven point
    equation = PV_premium - PV_protection
    return equation
    
# Defining a function to bootstrap the Hazard Rates
def HazardRate(CDS, B, stock, HR, row, row_year, y, a):
    # CDS = DataFrame containing the CDS Spreads
    # B = DataFrame of Discount Factors
    # stock = DataFrame of Stock Data
    # HR = DataFrame of Hazard Rates
    # row = row index of one CDS file
    # row_year = Time period with available Spread data
    # y = column index to be solved
    # x = Hazard Rate
    
    premium_list = []
    protection_list = []
    year = row_year[y]
    
    # Calculating the sum & cumsum of the Hazard Rates from time 0.25 to the first available Spread timestep
    lambda_sum = np.sum(HR.iloc[row, 1:int(row_year[y-1] * 4)+1].values)
    lambda_cumsum = np.cumsum(HR.iloc[row, 1:int(row_year[y-1] * 4)+1].values)
    
    # Calculating the premium & protection from time 0.25 to the first available Spread timestep
    premium_cum = np.dot(np.exp(-0.25 * lambda_cumsum), B.loc[B.index[row], 0.25:row_year[y-1]].values)
    protection_cum = np.dot((np.exp(-0.25 * np.insert(lambda_cumsum[0:-1], 0, 0)) -
                             np.exp(-0.25 * lambda_cumsum)), B.loc[B.index[row], 0.25:row_year[y-1]].values)

    premium_list.append(premium_cum)
    protection_list.append(protection_cum)

    count = 1
    for t in np.arange(row_year[y-1] + 0.25, row_year[y] + 0.25, 0.25):
        lambda_est = stock.loc[stock.index[row], "b"] + (stock.loc[stock.index[row], "c"] * a**2 * 
                     np.power(stock.loc[stock.index[row], "price"], 2 * stock.loc[stock.index[row], "Beta"]))
        premium = np.exp(-0.25 * (lambda_est * count + lambda_sum)) * B.loc[B.index[row], t]
        protection = (np.exp(-0.25 * (lambda_est * (count - 1) + lambda_sum)) - 
                      np.exp(-0.25 * (lambda_est * count + lambda_sum))) * B.loc[B.index[row], t]
        premium_list.append(premium)
        protection_list.append(protection)
        count = count + 1 

    PV_premium = CDS.loc[CDS.index[row], year] * 0.25 * np.sum(premium_list)
    PV_protection = (1 - CDS.loc[CDS.index[row], "Recovery"]) * np.sum(protection_list)
    # Equation to be solved at breakeven point
    equation = PV_premium - PV_protection
    return equation
    
def err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta):
    alpha = rate - div_yield + b
    vol =  a**2 * np.power(price, 2 * (Beta + 1))
    model = price * delta_t * (alpha + c * a**2 * np.power(price, 2*Beta))
    mom1 = np.mean(price_diff - model)
    mom2 = np.mean((price_diff - model) * price)
    mom3 = np.mean((price_diff - model) * delta_t)
    mom4 = np.mean((price_diff - model) ** 2 - vol)
    mom5 = np.mean(((price_diff - model) ** 2 - vol) * price)
    
    err_vec = np.array([mom1, mom2, mom3, mom4, mom5])
    return err_vec
    
def criterion(params, *args):
    a, b, c, Beta = params
    price, price_diff, div_yield, rate, delta_t, W = args
    
    err = err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta)
    crit_val = err.T @ W @ err 
    
    return crit_val
    
def get_err_mat(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta):
    R = 5
    N = 251
    err_matrix = np.zeros((R, N))
    
    alpha = rate - div_yield + b
    vol =  a**2 * np.power(price, 2 * (Beta + 1))
    model = price * delta_t * (alpha + c * a**2 * np.power(price, 2*Beta))
    
    err_matrix[0, :] = price_diff - model
    err_matrix[1, :] = (price_diff - model) * price
    err_matrix[2, :] = (price_diff - model) * delta_t
    err_matrix[3, :] = (price_diff - model) ** 2 - vol
    err_matrix[4, :] = ((price_diff - model) ** 2 - vol) * price
    
    return err_matrix
    
def jac_err(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta):
    jac_err = np.zeros((5, 4))
    h_a = 1e-8 * a
    h_b = 1e-8 * b
    h_c = 1e-8 * c
    h_Beta = 1e-8 * Beta

    jac_err[:, 0] = \
        ((err_vec(price, price_diff, div_yield, rate, delta_t, a + h_a, b, c, Beta) -
          err_vec(price, price_diff, div_yield, rate, delta_t, a - h_a, b, c, Beta)) / (2 * h_a)).flatten()    
    jac_err[:, 1] = \
        ((err_vec(price, price_diff, div_yield, rate, delta_t, a, b + h_b, c, Beta) -
          err_vec(price, price_diff, div_yield, rate, delta_t, a, b - h_b, c, Beta)) / (2 * h_b)).flatten()    
    jac_err[:, 2] = \
        ((err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c + h_c, Beta) -
          err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c - h_c, Beta)) / (2 * h_c)).flatten()
    jac_err[:, 3] = \
        ((err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta + h_Beta) -
          err_vec(price, price_diff, div_yield, rate, delta_t, a, b, c, Beta - h_Beta)) / (2 * h_Beta)).flatten()
    
    return jac_err
    
for row in range(len(CDSCOMP_df.index)):
    # row_year = List of time period with available Spread data of one CDS file
    row_year = []
        
    # Iterating through the columns of one CDS file
    for col in CDSCOMP_df.columns[0:-1:]:
        if np.isnan(CDSCOMP_df[col][row]):
            None
        else:
            row_year.append(col)
    
    # Initializing GMM
    i = 0
    a_init = 0.50
    b_init = 0.50
    c_init = 0.25
    Beta_init = -0.25
    params_init = np.array([a_init, b_init, c_init, Beta_init])
    
    # First-step
    w_hat = np.eye(5)
    gmm_args = (HRB_df["price"][i:i+251], HRB_df["price_diff"][i:i+251], HRB_df["div_yield"][i:i+251], 
                HRB_df["rate"][i:i+251], HRB_df["delta_t"][i:i+251], w_hat)

    bnds = ((-np.Inf, np.Inf), (-np.Inf, np.Inf), (0, 0.5), (-0.5, 0))
    cons = ({'type': 'eq', 'fun': lambda x: firstHazardRate(CDSCOMP_df, B_df, HRB_df, row, 
                                                            row_year, x[0], x[1], x[2], x[3])})
    results = minimize(criterion, params_init, args = (gmm_args),
                       bounds = bnds, constraints = cons)
    
    a_GMM, b_GMM, c_GMM, Beta_GMM = results.x
   
    # Second-step
    err_mat = get_err_mat(HRB_df["price"][i:i+251], HRB_df["price_diff"][i:i+251], HRB_df["div_yield"][i:i+251], 
                          HRB_df["rate"][i:i+251], HRB_df["delta_t"][i:i+251],
                          a_GMM, b_GMM, c_GMM, Beta_GMM)
    
    VCV2 = (1 / 251) * (err_mat @ err_mat.T)
    w_hat2 = lin.pinv(VCV2)
    gmm_args2 = (HRB_df["price"][i:i+251], HRB_df["price_diff"][i:i+251], HRB_df["div_yield"][i:i+251], 
                 HRB_df["rate"][i:i+251], HRB_df["delta_t"][i:i+251], w_hat2)
    
    results2 = minimize(criterion, params_init, args = (gmm_args2), 
                        bounds = bnds, constraints = cons)
    
    a_GMM2, b_GMM2, c_GMM2, Beta_GMM2 = results2.x
    
    # Calculating chi-squared
    HRB_df.loc[HRB_df.index[row], "chi_squared"] = 251 * criterion(results2.x, HRB_df["price"][i:i+251], HRB_df["price_diff"][i:i+251], HRB_df["div_yield"][i:i+251], 
                                                                   HRB_df["rate"][i:i+251], HRB_df["delta_t"][i:i+251], w_hat2)
    
    # Calculating p_value from the chi_squared
    HRB_df.loc[HRB_df.index[row], "p_value"] = chi2.sf(HRB_df.loc[HRB_df.index[row], "chi_squared"], 1)
    
    # Calculating standard errors of the parameters
    d_err = jac_err(HRB_df["price"][i:i+251], HRB_df["price_diff"][i:i+251], HRB_df["div_yield"][i:i+251], 
                    HRB_df["rate"][i:i+251], HRB_df["delta_t"][i:i+251], a_GMM2, b_GMM2, c_GMM2, Beta_GMM2)

    sig_hat = (1 / 251) * lin.inv(d_err.T @ w_hat2 @ d_err)
    
    # Calculating t-stats of the parameters
    t_df.loc[t_df.index[row], ["a", "b", "c", "Beta"]] = [a_GMM2 / np.sqrt(sig_hat[0, 0]), b_GMM2 / np.sqrt(sig_hat[1, 1]),
                                                          c_GMM2 / np.sqrt(sig_hat[2, 2]), Beta_GMM2 / np.sqrt(sig_hat[3, 3])]
    
    # Calculating p-values of the parameters
    p_df.loc[p_df.index[row], ["a", "b", "c", "Beta"]] = [t.sf(np.abs(t_df.loc[t_df.index[row], "a"]), 250),
                                                          t.sf(np.abs(t_df.loc[t_df.index[row], "b"]), 250),
                                                          t.sf(np.abs(t_df.loc[t_df.index[row], "c"]), 250),
                                                          t.sf(np.abs(t_df.loc[t_df.index[row], "Beta"]), 250)]
    
    # Storing the parameters
    a_df.iloc[row, int(row_year[0] * 4)] = results2.x[0]
    a_df.iloc[row, 0:int(row_year[0] * 4)] = a_df.iloc[row, int(row_year[0] * 4)]
    
    HRB_df.iloc[row, -3] = results2.x[1]
    HRB_df.iloc[row, -2] = results2.x[2]
    HRB_df.iloc[row, -1] = results2.x[3]
    
    # Calculating lambda from the obtained parameters
    lambda_est = results2.x[1] + (results2.x[2] * results2.x[0] **2 *
                                  np.power(HRB_df.loc[HRB_df.index[row], "price"], 2 * results2.x[3]))
    
    lambda_df.iloc[row, int(row_year[0] * 4)] = lambda_est
    lambda_df.iloc[row, 0:int(row_year[0] * 4)] = lambda_df.iloc[row, int(row_year[0] * 4)]
    
for row in range(len(CDSCOMP_df.index)):
    # row_year = List of time period with available Spread data of one CDS file
    row_year = []
        
    # Iterating through the columns of one CDS file
    for col in CDSCOMP_df.columns[0:-1:]:
        if np.isnan(CDSCOMP_df[col][row]):
            None
        else:
            row_year.append(col)
    
    # Iterating through different CDS maturities
    for y in range(1, len(row_year)):
        x0 = [a_df.iloc[row, int(row_year[y-1] * 4)]]
                                   
        results3 = least_squares(lambda x: HazardRate(CDSCOMP_df, B_df, HRB_df, lambda_df, row, row_year, y, x[0]), x0)
        a_df.iloc[row, int(row_year[y] * 4)] = results3.x[0]
        a_df.iloc[row, int(row_year[y-1] * 4)+1:int(row_year[y] * 4)] = a_df.iloc[row, int(row_year[y] * 4)] 
                                   
        lambda_est2 = HRB_df.loc[HRB_df.index[row], "b"] + (HRB_df.loc[HRB_df.index[row], "c"] * results3.x[0] **2 *
                      np.power(HRB_df.loc[HRB_df.index[row], "price"], 2 * HRB_df.loc[HRB_df.index[row], "Beta"]))
                                   
        lambda_df.iloc[row, int(row_year[y] * 4)] = lambda_est2
        lambda_df.iloc[row, int(row_year[y-1] * 4)+1:int(row_year[y] * 4)] = lambda_df.iloc[row, int(row_year[y] * 4)]      
        
    lambda_df = lambda_df.ffill(axis = 1)
    
# Index number of the first data of each month
index_list = [0, 21, 40, 61, 82, 103, 124, 146, 167, 189, 211, 228]

for i in index_list:
    fig, ax = plt.subplots()
    font = {'fontname':'Helvetica', 'size':'16'}
    ax.bar(lambda_df.columns, lambda_df.iloc[i, :], width= 2.0, color = "#007acc")
    myLocator = mticker.MultipleLocator(20)
    ax.xaxis.set_major_locator(myLocator)
    ax.set_title("Intensity Curve "+str(lambda_df.index[i])[0:10], **font)
    ax.set_ylabel("\u03BB")
    ax.set_xlabel("Time (Years)", **font)
    plt.show()
    fig.savefig("Part B/Plots/Plot "+str(lambda_df.index[i])[0:10]+'.png')
    
paraChange_df = pd.DataFrame(np.NaN, index = HRB_df.index[index_list], columns = ["a", "b", "c", "Beta"])
paraChange_df["a"] = 100 * (a_df.iloc[index_list, 0].values - a_df.iloc[index_list, 0].shift(1).values) / a_df.iloc[index_list, 0].values
paraChange_df["b"] = 100 * (HRB_df.loc[HRB_df.index[index_list], "b"].values - 
                            HRB_df.loc[HRB_df.index[index_list], "b"].shift(1).values) / HRB_df.loc[HRB_df.index[index_list], "b"].values
paraChange_df["c"] = 100 * (HRB_df.loc[HRB_df.index[index_list], "c"].values - 
                            HRB_df.loc[HRB_df.index[index_list], "c"].shift(1).values) / HRB_df.loc[HRB_df.index[index_list], "c"].values
paraChange_df["Beta"] = 100 * (HRB_df.loc[HRB_df.index[index_list], "Beta"].values - 
                               HRB_df.loc[HRB_df.index[index_list], "Beta"].shift(1).values) / HRB_df.loc[HRB_df.index[index_list], "Beta"].values
                               
fig, ax = plt.subplots()
ax.plot(paraChange_df.index, paraChange_df["a"], label = "a")
ax.plot(paraChange_df.index, paraChange_df["b"], label = "b")
ax.plot(paraChange_df.index, paraChange_df["c"], label = "c")
ax.plot(paraChange_df.index, paraChange_df["Beta"], label = "\u03B2")
ax.legend()
myLocator = mticker.MultipleLocator(60)
ax.xaxis.set_major_locator(myLocator)
ax.set_ylabel("% Change")
ax.set_xlabel("Date", **font)
