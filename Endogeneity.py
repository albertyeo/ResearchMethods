import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

# Importing JC1.csv file as a DataFrame
# JC1.csv contains data of ore_price, growth, and VIX from December 2013 to May 2019
orePrice_df = pd.read_csv("JC1.csv", header = 0)

# Importing BDI.csv file as a DataFrame
# BDI.csv contains data of BDI from from December 2013 to May 2019
BDI = pd.read_csv("BDI.csv")

# Adding BDI data into orePrice DataFrame
orePrice_df["BDI"] = BDI.iloc[::-1, 1].values

# Importing #amended_JC3.csv file as a DataFrame
# JC3.csv contains data of port, ore_price, growth, logd, logf, avefreight, and aveqty from December 2013 to May 2019
freightCost_df = pd.read_csv("#amended_JC3.csv", header = 0)

# Adding BDI data into freightCost DataFrame (same data for every port)
freightCost_df["BDI"] = np.tile(BDI.iloc[::-1, 1].values, 4)

# Creating two port dummy variables
freightCost_df["port_dummy1"] = 0
freightCost_df["port_dummy2"] = 0

# Assigning the dummy variables to Dampier & Port Hedland ports
for i in range(len(freightCost_df.index)):
    if freightCost_df.iloc[i, 2] == "DAMPIER":
        freightCost_df.iloc[i, -2] = 1
    if freightCost_df.iloc[i, 2] == "PORT HEDLAND":
        freightCost_df.iloc[i, -1] = 1
        
# Defining descStats function to compute descriptive statistics of variables
def descStats(series):
    mean = np.mean(series)
    std = np.std(series)
    q1 = np.percentile(series, 25)
    q2 = np.median(series)
    q3 = np.percentile(series, 75)
    return np.array([mean, std, q1, q2, q3])
    
# Creating a table1 DataFrame which contains descStats results of the variables
table1 = pd.DataFrame([], columns = ["Mean", "Std Dev", "25%", "Median", "75%"])
table1.loc["Iron Ore Price $", :] = descStats(orePrice_df["ore_price"])
table1.loc["Growth Rate", :] = descStats(orePrice_df["growth"])
table1.loc["VIX Index", :] = descStats(orePrice_df["VIX"])
table1.loc["ln (Distance)", :] = descStats(freightCost_df["logd"])
table1.loc["ln (Fuel Price)", :] = descStats(freightCost_df["logf"])
table1.loc["Freight Rate $", :] = descStats(freightCost_df["avefreight"])
table1.loc["Cargo Volume", :] = descStats(freightCost_df["aveqty"])
table1.loc["BDI $", :] = descStats(orePrice_df["BDI"])

# Defining ADFStats function to compute results of Augmented Dickey-Fuller (ADF) test
def ADFStats(series):
    result = adfuller(series)
    ADF = result[0]
    p_value = result[1]
    one_percent = result[4]["1%"]
    five_percent = result[4]["5%"]
    ten_percent = result[4]["10%"]
    return np.array([ADF, p_value, one_percent, five_percent, ten_percent])
    
# Creating a table2 DataFrame which contains ADFStats results of the variables
table2 = pd.DataFrame([], columns = ["Iron Ore Prices $", "Growth Rates", "VIX Index", "BDI"],
                     index = ["ADF-Statistics", "p-Value", "1%", "5%", "10%"])
table2["Iron Ore Prices $"] = ADFStats(orePrice_df["ore_price"])
table2["Growth Rates"] = ADFStats(orePrice_df["growth"])
table2["VIX Index"] = ADFStats(orePrice_df["VIX"])
table2["BDI"] = ADFStats(orePrice_df["BDI"])

# Running an OLS regression
orePrice_BDI_formula = "ore_price ~ growth + VIX + BDI"
orePrice_BDI_results = smf.ols(orePrice_BDI_formula, orePrice_df).fit(cov_type = "HC1")

# Running an ADF unit root test on the estimated residuals of the regression
orePrice_df["ore_price_hat_BDI"] = np.dot(orePrice_BDI_results.params, np.array([1, orePrice_df["growth"], 
                                                                                 orePrice_df["VIX"],
                                                                                 orePrice_df["BDI"]]))
orePrice_df["residual_error_BDI"] = orePrice_df["ore_price"].values - orePrice_df["ore_price_hat_BDI"].values
orePriceRes_BDI_ADF = ADFStats(orePrice_df["residual_error_BDI"])

print('-----------------------------------')
print("ADF-Statistics:", orePriceRes_BDI_ADF[0])
print("p-Value       :", orePriceRes_BDI_ADF[1])
print("1%            :", orePriceRes_BDI_ADF[2])
print("5%            :", orePriceRes_BDI_ADF[3])
print("10%           :", orePriceRes_BDI_ADF[4])

# Setting up the DataFrame for PanelOLS and cluster effect by port
freightCost_panel = freightCost_df.set_index(["port", "date"])

# Defining the Explanatory Variables
freightCost_vars = ["growth", "logd", "logf", "ore_price", "port_dummy1", "port_dummy2"]
freightCost_reg = sm.add_constant(freightCost_panel[freightCost_vars])

# Running a panel regression
freightCost_results = PanelOLS(freightCost_panel["avefreight"], freightCost_reg, 
                               entity_effects = False).fit(cov_type="clustered", cluster_entity=True)
                               
# Setting up the DataFrame for PanelOLS and cluster effect by port
freightCost_BDI_panel = freightCost_df.set_index(["port", "date"])

# Defining the Explanatory Variables
freightCost_BDI_vars = ["growth", "logd", "logf", "ore_price", "BDI", "port_dummy1", "port_dummy2"]
freightCost_BDI_reg = sm.add_constant(freightCost_BDI_panel[freightCost_BDI_vars])

# Running a panel regression
freightCost_BDI_results = PanelOLS(freightCost_BDI_panel["avefreight"], freightCost_BDI_reg, 
                                   entity_effects = False).fit(cov_type="clustered", cluster_entity=True)
                                   
# Calculating of fitted value of ore-price
freightCost_df["ore_price_hat_BDI"] = np.dot(orePrice_BDI_results.params, np.array([1, freightCost_df["growth"], 
                                                                                    freightCost_df["VIX"],
                                                                                    freightCost_df["BDI"]]))
                                                                                    
# Setting up the DataFrame for PanelOLS and cluster effect by port
twoSLS_BDI_panel = freightCost_df.set_index(["port", "date"])

# Defining the Explanatory Variables
twoSLS_BDI_vars = ["growth", "logd", "logf", "ore_price_hat_BDI", "BDI", "port_dummy1", "port_dummy2"]
twoSLS_BDI_reg = sm.add_constant(twoSLS_BDI_panel[twoSLS_BDI_vars])

# Running a panel regression
twoSLS_BDI_results = PanelOLS(twoSLS_BDI_panel["avefreight"], twoSLS_BDI_reg, 
                              entity_effects = False).fit(cov_type="clustered", cluster_entity=True)
