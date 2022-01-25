The aim for our project is to simplify cryptocurrency investing through a portfolio analyzer. We’ll examine five different cryptocurrencies past and futures through analyzing various models and ratios dictating each individual cryptos metrics, measuring future performance, along with comparing them amongst each other to dictate the best investment options. The cryptos analyzed in this project include the following: Bitcoin, Ethereum, Cardano, Binance, and Litecoin.

Table of contents
1. Libraries and Dependencies
2. Connect, Extract, and Prepare Data from API
3. Daily Return
4. Cumulative Return
5. Standard Deviation
6. Annualized Standard Deviation
7. Sharpe Ratio
8. Variance
9. Beta
10. Probability Distribution
11. Density
12. Monte Carlo



# 1. Libraries and Dependencies

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import hvplot.pandas
from MCForecastTools import MCSimulation
%matplotlib inline



# 2. Connect, Extract, and Prepare Data from API

# Load env environment variables
load_dotenv()

# Set crypto_api_key
crypto_api_key = os.getenv('crypto_api_key')

# Set the data endpoints
btc_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1116&api_key='
eth_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&limit=1116&api_key='
ada_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=ADA&tsym=USD&limit=1116&api_key='
bnb_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BNB&tsym=USD&limit=1116&api_key='
ltc_url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=LTC&tsym=USD&limit=1116&api_key='

# Concatenate endpoints and key
btc_request_url = btc_url + crypto_api_key
eth_request_url = eth_url + crypto_api_key
ada_request_url = ada_url + crypto_api_key
bnb_request_url = bnb_url + crypto_api_key
ltc_request_url = ltc_url + crypto_api_key

# Make request to the API
response_btc = requests.get(btc_request_url).json()

response_eth = requests.get(eth_request_url).json()

response_ada = requests.get(ada_request_url).json()

response_bnb = requests.get(bnb_request_url).json()

response_ltc = requests.get(ltc_request_url).json()

# Extract data into list
btc_high_price = []
for data_point in response_btc['Data']['Data']:
    btc_high_value = data_point.get('high')
    btc_high_price.append(btc_high_value)
    
eth_high_price = []
for data_point in response_eth['Data']['Data']:
    eth_high_value = data_point.get('high')
    eth_high_price.append(eth_high_value)

ada_high_price = []
for data_point in response_ada['Data']['Data']:
    ada_high_value = data_point.get('high')
    ada_high_price.append(ada_high_value)
    
bnb_high_price = []
for data_point in response_bnb['Data']['Data']:
    bnb_high_value = data_point.get('high')
    bnb_high_price.append(bnb_high_value)
    
ltc_high_price = []
for data_point in response_ltc['Data']['Data']:
    ltc_high_value = data_point.get('high')
    ltc_high_price.append(ltc_high_value)
    
crypto_data = {
    'Bitcoin': btc_high_price,
    'Ethereum': eth_high_price,
    'Cardano': ada_high_price,
    'Binance': bnb_high_price,
    'Litecoin': ltc_high_price
}

# Creating a dataframe for the data
crypto_data_df = pd.DataFrame(crypto_data)

# Replacing the index column and adding dates with one data row representing the high price for one day over the past two years
crypto_data_df['Date'] = pd.date_range(start='2019-01-01', periods=len(crypto_data_df), freq='D')
crypto_df = crypto_data_df.set_index('Date', drop=True)

# Subsetting data to individual crypto datframes
btc_series = crypto_df['Bitcoin']
btc_df = btc_series.to_frame()

eth_series = crypto_df['Ethereum']
eth_df = eth_series.to_frame()

ada_series = crypto_df['Cardano']
ada_df = ada_series.to_frame()

bnb_series = crypto_df['Binance']
bnb_df = bnb_series.to_frame()

ltc_series = crypto_df['Litecoin']
ltc_df = ltc_series.to_frame()



# 3. Daily Return

# Calculating daily return data for the whole two year period
crypto_daily_returns = crypto_df.pct_change()

# Plotting daily return data 
crypto_daily_returns.hvplot(figsize=(15,10), title='Daily Returns of BTC,ETH,ADA,BNB, and LTC 2019-Present')



# 4. Cumulative Return

# Calculating cumulative return data for the whole two year period
crypto_cumulative_returns = (1 + crypto_daily_returns).cumprod()


# Plotting cumulative return data
crypto_cumulative_returns.hvplot(figsize=(15,10), title='Cumulative Returns of BTC,ETH,ADA,BNB, and LTC 2019-Present')



# 5. Standard Deviation

# Calculating the std from 2019 to present
crypto_std = crypto_daily_returns.std()

# Displaying the sorted std from 2019 to present
display(crypto_std.sort_values())

# Plotting the std data
crypto_std.hvplot.bar(figsize=(15,10), title='Standard Deviation of BTC,ETH,ADA,BNB, and LTC 2019-Present', rot=0)

# 6. Annualized Standard Deviation

# The number of trading days in one crypto calendar year
trading_days=365

# Calculating annualized std for the whole two year period 
crypto_annualized_std = crypto_std * np.sqrt(trading_days)

# Displaying the annualized std 
display(crypto_annualized_std.sort_values())

# Plotting the annualized std data
crypto_annualized_std.hvplot.bar(figsize=(15,10), title='Annualized Standard Deviation of BTC,ETH,ADA,BNB, and LTC 2019-Present', rot=0)



# 7. Annualized Average Standard Deviation

# Calculating the annual average daily returns for the whole two year period
crypto_annualized_returns = crypto_daily_returns.mean() * trading_days

# Displaying the annualized average daily returns
display(crypto_annualized_returns.sort_values())

# Plotting the annualized daily return data
crypto_annualized_returns.hvplot.bar(figsize=(15,10), title='Annualized Average Daily Return of BTC,ETH,ADA,BNB, and LTC 2019-Present', rot=0)



# 8. Sharpe Ratios

# Calculating the Sharpe Ratios 
crypto_sharpe_ratios = crypto_annualized_returns / crypto_annualized_std

# Displaying sorted Sharpe Ratios
display(crypto_sharpe_ratios.sort_values())

# Plotting the Sharpe Ratios
crypto_sharpe_ratios.hvplot.bar(figsize=(15,10), title='Sharpe Ratios of BTC,ETH,ADA,BNB, and LTC 2019-Present')



# 9. Variance

# Calculating the variance
crypto_variance = crypto_daily_returns.var()

# Displaying the sorted variance
display(crypto_variance.sort_values())

# Plotting the variance
crypto_variance.hvplot.bar(figsize=(15,10), title='Variance of BTC,ETH,ADA,BNB, and LTC 2019-Present')

# Calculating bitcoin's variance for beta calculations
btc_variance = crypto_daily_returns['Bitcoin'].var()



# 10. Covariance

# Calculating the four crypto's covariance to bitcoin
eth_covariance = crypto_daily_returns['Ethereum'].cov(crypto_daily_returns['Bitcoin'])
ada_covariance = crypto_daily_returns['Cardano'].cov(crypto_daily_returns['Bitcoin'])
bnb_covariance = crypto_daily_returns['Binance'].cov(crypto_daily_returns['Bitcoin'])
ltc_covariance = crypto_daily_returns['Litecoin'].cov(crypto_daily_returns['Bitcoin'])



# 11. Beta

# Calculating the four crypto's beta
eth_beta = eth_covariance / btc_variance
ada_beta = ada_covariance / btc_variance
bnb_beta = bnb_covariance / btc_variance
ltc_beta = ltc_covariance / btc_variance



12. Probability Distribution

# Plotting the probability distributions  
crypto_daily_returns["Bitcoin"].hvplot(kind='hist', figsize=(15,7), title= 'Crypto Probability Distribution')
crypto_daily_returns["Ethereum"].hvplot(kind='hist', figsize=(15,7), title= 'Ethereum Probability Distribution')
crypto_daily_returns["Binance"].hvplot(kind='hist',figsize=(15,7), title='Binance Probability Distribution')
crypto_daily_returns["Litecoin"].hvplot(kind='hist',figsize=(15,7), title= "Litecoin Probability Distribution")
crypto_daily_returns["Cardano"].hvplot(kind='hist', figsize=(15,7), title= "Cardano Probability Distribution")



13. Density

# Plotting the density graph
crypto_daily_returns.hvplot.density(figsize=(20,10), title= 'Density Plot for Cryptocurrencies')



14. Monte Carlo

# Creating a new dataframe prepped for the monte carlo simulation for the next two years
mc_twoyear = MCSimulation(
    portfolio_data=crypto_df,
    weights=[.20,.20,.20,.20,.20],
    num_simulation=500,
    num_trading_days=365*2
)

# Running five hundred monte carlo simulations of cumulative return trajectories over the next 730 days
mc_twoyear.calc_cumulative_return()

# Plotting the monte carlo simulations cumulative return trajectories
mc_sim_line_plot = mc_twoyear.plot_simulation()

# Plotting the distribution across all 500 simulations
mc_sim_dist_plot = mc_twoyear.plot_distribution()

# Extracting the summary statistics 
mc_summary_statistics = mc_twoyear.summarize_cumulative_return()

# Using the lower and upper 95% confidence intervals from the summary statistics to calculate the range of probable cumulative returns for a $10,000 investment
ci_95_lower_cumulative_return = mc_summary_statistics[8] * 10000
ci_95_upper_cumulative_return = mc_summary_statistics[9] * 10000

print(f'There is a 95% chance that an initial investment of $10,000 in the portfolio'
     f' over the next two years will end within the range of'
     f' ${ci_95_lower_cumulative_return: .2f} and ${ci_95_upper_cumulative_return: .2f}.')

