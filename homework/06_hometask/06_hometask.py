#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:06:46 2023

@author: isa
"""

import pandas as pd
import matplotlib.pyplot as plt
#pip install statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import VECM
import os
# Set working directory 
os.chdir("/Users/isa/Downloads/pythondataMLU220213600/220213600/homework/06_hometask")

# Load data
prices = pd.read_csv('02_python_data.csv', index_col=0, parse_dates=True)

"""
First group of tasks
"""
dax_index = prices['.GDAXI']
# Set frequency to daily
dax_index = dax_index.asfreq('D')
# Fit an ARMA(1,1) model to the DAX index
model_arma_11 = ARIMA(dax_index, order=(1, 0, 1))
model_arma_11_fit = model_arma_11.fit()
# Generate a 30-day forecast using the ARMA(1,1) model
forecast_arma_11 = model_arma_11_fit.forecast(steps=30)
# Fit an ARMA(5,5) model to the DAX index
model_arma_55 = ARIMA(dax_index, order=(5, 0, 5))
model_arma_55_fit = model_arma_55.fit()
# Generate a 30-day forecast using the ARMA(5,5) model
forecast_arma_55 = model_arma_55_fit.forecast(steps=30)
# Plot the forecasts and compare the results
plt.figure(figsize=(12, 6))
plt.plot(dax_index, label='DAX Index')
plt.plot(forecast_arma_11, label='ARMA(1,1) Forecast')
plt.plot(forecast_arma_55, label='ARMA(5,5) Forecast')
plt.title('DAX Index Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
"""
Second group of tasks
"""
#Read solution script of professor 
from nbconvert import PythonExporter
from nbformat import read
notebook_path = '/Users/isa/Downloads/pythondataMLU220213600/220213600/homework/06_hometask/05_optimization_solutions.ipynb'
with open(notebook_path) as f:
    notebook_content = read(f, as_version=4)

exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(notebook_content)
exec(python_code)
# Step 2: Call the variables for the portfolios and calcualte return for the DAX index
dax_returns = returns['.GDAXI']
# Calculate the portfolio returns
mvp_returns = returns.iloc[:, 1:] @ weights_MVP
replication_returns = returns.iloc[:, 1:] @ weights_linreg
smart_beta_returns = returns[best_model.keys()] @ best_model

# Perform cointegration tests
result_mvp, pvalue_mvp, _ = coint(mvp_returns, dax_returns)
result_replication, pvalue_replication, _ = coint(replication_returns, dax_returns)
result_smart_beta, pvalue_smart_beta, _ = coint(smart_beta_returns, dax_returns)

# Analyze the cointegration results
if pvalue_mvp < 0.05:
    print("MVP Portfolio is cointegrated with the DAX index.")
else:
    print("MVP Portfolio is not cointegrated with the DAX index.")

if pvalue_replication < 0.05:
    print("Replication Portfolio is cointegrated with the DAX index.")
else:
    print("Replication Portfolio is not cointegrated with the DAX index.")

if pvalue_smart_beta < 0.05:
    print("Smart-Beta Portfolio is cointegrated with the DAX index.")
else:
    print("Smart-Beta Portfolio is not cointegrated with the DAX index.")
    
"""
Third group of tasks
"""
# Calculate returns for all assets
returns = prices.pct_change().iloc[1:, :]
# Perform cointegration tests for each asset pair
num_assets = returns.shape[1]
asset_names = returns.columns
cointegration_pairs = []
for i in range(num_assets):
    for j in range(i+1, num_assets):
        asset1_returns = returns.iloc[:, i]
        asset2_returns = returns.iloc[:, j]

        result, pvalue, _ = coint(asset1_returns, asset2_returns)

        if pvalue < 0.05:
            cointegration_pairs.append((asset_names[i], asset_names[j]))

# Analyze the cointegration results
if len(cointegration_pairs) > 0:
    print("Cointegration relationships found:")
    for pair in cointegration_pairs:
        print(f"{pair[0]} and {pair[1]}")
else:
    print("No cointegration relationships found among the assets.")

# Thus paired assets are cointegrated, as p-value is less than 0.05 indicating significant cointegration relationship
"""
Fourth group of tasks
"""
# Set the frequency of the date index
returns.index.freq = returns.index.inferred_freq
for i in range(num_assets):
    for j in range(i+1, num_assets):
        asset1_returns = returns.iloc[:, i]
        asset2_returns = returns.iloc[:, j]

        result, pvalue, _ = coint(asset1_returns, asset2_returns)

        if pvalue < 0.05:
            cointegration_pairs.append((asset_names[i], asset_names[j]))

# Analyze the cointegration results
if len(cointegration_pairs) > 0:
    print("Cointegration relationships found:")
    for pair in cointegration_pairs:
        print(f"{pair[0]} and {pair[1]}")
        
    # Select the first cointegration pair for analysis (change if desired)
    asset1, asset2 = cointegration_pairs[0]

    # Create the error correction model (ECM)
    cointegration_data = pd.concat([returns[asset1], returns[asset2]], axis=1)
    model = VECM(cointegration_data, k_ar_diff=1, coint_rank=1)
    fitted_model = model.fit()

    # Display the ECM summary
    print(fitted_model.summary())

else:
    print("No cointegration relationships found among the assets.")




