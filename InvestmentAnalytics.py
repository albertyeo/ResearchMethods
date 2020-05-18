import pandas as pd
import numpy as np

crsp_q_result = pd.read_csv("crsq_q_result_python.csv")
crsp_q_result.head()
crsp_q_result['date'] = pd.to_datetime(crsp_q_result['date'])

# Get the number of missing values of each column
crsp_q_result.isnull().sum()

# Group by permno, count the number of non-null entries
s = crsp_q_result[['mktcap', 'return','permno']].groupby(['permno']).agg(
        # Get number of non-null observations of the return column for each group
        return_cnt = ('return',"count"),
        # Get number of non-null observations of the mktcap column for each group
        mktcap_cnt = ('mktcap',"count")).reset_index()
s.columns

# Sort the dataframe by count in descending order
s.sort_values(by=['mktcap_cnt'],ascending=False,inplace=True)
s.head(3)

# Filter the companies whose number of records = 362, which is the largest count
rs = s[s.mktcap_cnt == 362]
# Make the permno of filtered companies to a list 
uniqueVals = rs["permno"].tolist()
len(uniqueVals)

crsp_sample = crsp_q_result[crsp_q_result['permno'].isin(uniqueVals)]

beta_ratio = pd.read_csv("beta_ratio.csv")
beta_ratio.head(3)

# Sort beta_ratio by column permno, and public_date
beta_ratio = beta_ratio.sort_values(['permno', 'public_date'])
beta_ratio.head(2)

# Format column public_date into the format "yyyy-mm-dd"
beta_ratio['date'] = pd.to_datetime(beta_ratio['public_date'].astype(str), format='%Y%m%d').dt.strftime("%Y-%m-%d")

# Make column public_date as datetime object
beta_ratio['date'] = pd.to_datetime(beta_ratio['date'])

# Remove companies with too many null values of book_market_ratio
null_count = beta_ratio.groupby('permno').agg({'bm': lambda x: x.isnull().sum()})
null_count = null_count.sort_values(by=['bm'],ascending=False)
null_count = null_count.reset_index()

# Set the threshold to 30, 
# since we can still do reasonable imputation suppose there's no more than two missing recods per year
permno_list = null_count[null_count['bm'] <= 30].permno.values

# Filter the table beta_ratio using `permno_list`
beta_cleaned = beta_ratio[beta_ratio['permno'].isin(permno_list)]

# Check how many permno are there in common in two dataframes?
common_permno = list(set(beta_cleaned.permno.unique()) & set (crsp_sample.permno.unique()))

# Extract data of 432 companies 
crsp_sample = crsp_sample[crsp_sample['permno'].isin(common_permno)]

# Merge CRSP and beta table on two columns
merged = pd.merge(crsp_sample, beta_cleaned, on=['permno','date'], how='left')

# Compute 'ep' from 'pe_exi'
merged['ep'] = 1 / merged['pe_exi']

# Compute 'ab' from 'de_ratio' and 'debt_assets'
merged['ab'] = merged['de_ratio'] / merged['debt_assets']

# Compute 'am' from 'ab' and 'bm'
merged['am'] = merged['ab'] * merged['bm']

# Create a sub_dataframe merged_2 from merged with the columns we need
merged_2 = merged[['permno','date','price','return','vol','roe','mktcap','bm','ep','ab','am']]
merged_2.head(3)

import warnings
warnings.filterwarnings("ignore")

cols = ['price','return','vol','roe','mktcap','bm','ep','ab','am']
merged_2.update(merged_2.groupby('permno')[cols].ffill())

# groupby permno and count the number of null values in column book_market_ratio
df_null_count = merged_2.groupby('permno').agg({'bm': lambda x: x.isnull().sum()})
df_null_count = df_null_count.sort_values(by=['bm'])
df_null_count = df_null_count.reset_index()

df_null_count.groupby('bm').count().sort_values(by=['permno'],ascending=False).head(3)

final_list = df_null_count[df_null_count['bm']== 0.0].permno.values

merged_3 = merged_2[merged_2['permno'].isin(final_list)]

merged_3 = merged_3[~merged_3['permno'].isin(merged_3[merged_3['ep'].isnull()].permno.unique())]
merged_3 = merged_3[~merged_3['permno'].isin(merged_3[merged_3['ab'].isnull()].permno.unique())]
merged_3 = merged_3[~merged_3['permno'].isin(merged_3[merged_3['am'].isnull()].permno.unique())]

merged_3['ep_dummy'] = 0
merged_3.loc[merged_3['ep'] < 0, 'ep_dummy'] = 1
merged_3.loc[merged_3['ep'] < 0, 'ep'] = 0

from math import sqrt

merged_3['size_lag']= merged_3.groupby(['permno'])['mktcap'].shift(1)
merged_3['bm_lag']= merged_3.groupby(['permno'])['bm'].shift(1)
merged_3['ep_lag']= merged_3.groupby(['permno'])['ep'].shift(1)
merged_3['ab_lag']= merged_3.groupby(['permno'])['ab'].shift(1)
merged_3['am_lag']= merged_3.groupby(['permno'])['am'].shift(1)
merged_3['ep_dummy_lag']= merged_3.groupby(['permno'])['ep_dummy'].shift(1)
merged_3['vol_lag']= merged_3.groupby(['permno'])['vol'].shift(1)
merged_3['roe_lag']= merged_3.groupby(['permno'])['roe'].shift(1)
merged_3['price_lag'] = merged_3.groupby(['permno'])['price'].shift(1)

merged_3['log_size_lag'] = np.log(merged_3['size_lag'])
merged_3['log_bm_lag'] = np.log(merged_3['bm_lag'])
merged_3['log_ab_lag'] = np.log(merged_3['ab_lag'])
merged_3['log_am_lag'] = np.log(merged_3['am_lag'])

merged_4 = merged_3[['permno','date','price','return','price_lag','vol_lag','roe_lag','log_size_lag','log_bm_lag','log_ab_lag','log_am_lag',
                     'ep_lag','ep_dummy_lag']]
                     
# Create a column of date (end of month) for further use (to merge with ff_factors)
merged_4['EndOfMonth'] = merged_4['date'] + pd.offsets.MonthEnd(0)
merged_4[['date','EndOfMonth']]

import getFamaFrenchFactors as gff

# Get the Fama French 3 factor model (monthly data)
ff4 = gff.carhart4Factor(frequency='m') 
ff4 = ff4[~ff4['MOM'].isnull()].reset_index(drop=True)

ff4.rename(columns={'date_ff_factors': 'EndOfMonth'}, inplace=True)
ff4['EndOfMonth'] = pd.to_datetime(ff4['EndOfMonth'])
ff4.head(2)

ff4['MKT_RET'] = ff4['Mkt-RF'] + ff4['RF']

df = pd.merge(merged_4, ff4, on=['EndOfMonth'], how='left')
df['excess_ret']=df['return']-df['RF']

start_date = '1993-12-31'
end_date = '2018-12-31'
#greater than the start date and smaller than the end date
mask = (df['date'] > start_date) & (df['date'] <= end_date)
# reassign sub-dataframe to df
df = df.loc[mask]

df = df[~df['permno'].isin(df[df['log_ab_lag'].isnull()].permno.unique())]
df = df[~df['permno'].isin(df[df['log_am_lag'].isnull()].permno.unique())]

import seaborn as sns
from matplotlib import pyplot as pl
%matplotlib inline 
# This line is necessary for the plot to appear in a Jupyter notebook
corr = df[['log_size_lag','log_bm_lag','log_ab_lag','log_am_lag',
           'ep_lag','ep_dummy_lag', 'roe_lag']].corr()

f, ax = pl.subplots(figsize=(8, 6))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
           
corr.style.background_gradient(cmap='coolwarm')

df = df[~df['permno'].isin(df[np.logical_or(df['log_bm_lag']>10, df['log_bm_lag']<-10)].permno.unique())]
df.to_csv("data.csv",index=False)
