#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:09:09 2023
@author: isa
"""
# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
#conda install scikit-learn
from sklearn.linear_model import LinearRegression
# Set working directory 
os.chdir("/Users/isa/Downloads/pythondataMLU220213600/220213600/homework/05_hometask")
#Load the data
data = pd.read_csv('02_python_data.csv', index_col='Date', parse_dates=True)
# Extract the closing prices of the constituents
prices = data.iloc[:, 1:].values
# Extract GDAXI prices separately
gdaxi_prices = prices[:, 0]
# Exclude GDAXI from the constituents
"""
First Group of tasks
"""
constituent_prices = prices[:, 1:]
# Calculate the returns for GDAXI
gdaxi_returns = np.log(gdaxi_prices[1:] / gdaxi_prices[:-1])
# Calculate the returns for the constituents
constituent_returns = np.log(constituent_prices[1:] / constituent_prices[:-1])
# Calculate the covariance matrix for the constituents
constituent_covariance_matrix = np.cov(constituent_returns.T)
# Calculate the inverse of the covariance matrix for the constituents
inv_constituent_cov_matrix = np.linalg.inv(constituent_covariance_matrix)
# Create an array of ones with the same length as the number of assets
ones = np.ones(len(constituent_covariance_matrix))
# Calculate the MVP weights for the constituents
mvp_weights_constituents = inv_constituent_cov_matrix.dot(ones) / (ones.T.dot(inv_constituent_cov_matrix).dot(ones))
sum_mvp_weights = np.sum(mvp_weights_constituents)
# Normalize the weights to ensure their sum is 1
mvp_weights_constituents /= np.sum(mvp_weights_constituents)
# Calculate the risk (standard deviation) of the MVP portfolio for the constituents
mvp_risk_constituents = np.sqrt(mvp_weights_constituents.T.dot(constituent_covariance_matrix).dot(mvp_weights_constituents))
# Calculate the returns of the MVP portfolio for the constituents
mvp_returns_constituents = np.dot(constituent_returns, mvp_weights_constituents)
# Calculate the cumulative log returns for GDAXI
cumulative_returns_gdaxi = np.cumsum(gdaxi_returns)
# Calculate the cumulative log returns for the MVP portfolio of the constituents
cumulative_returns_mvp_constituents = np.cumsum(mvp_returns_constituents)
# Print the risk and returns for the constituents
print("Sum of vector of weights", sum_mvp_weights)
print("Risk (Standard Deviation) - MVP (Constituents):", mvp_risk_constituents)
print("Returns - MVP (Constituents):", np.sum(mvp_returns_constituents))
# Plot the cumulative log returns for GDAXI and MVP (Constituents)
plt.plot(cumulative_returns_gdaxi, label="GDAXI")
plt.plot(cumulative_returns_mvp_constituents, label="MVP (Constituents)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cumulative Log Returns")
plt.title("Cumulative Log Returns of GDAXI and MVP (Constituents)")
plt.show()
"""
Second Group of tasks
"""
# Fit a linear regression model
model = LinearRegression()
model.fit(constituent_returns, gdaxi_returns)
# Extract the coefficients (weights) of the regression model
weights = model.coef_
# Normalize the weights to ensure their sum is 1
weights /= np.sum(weights)
# Print the actual weights
print("Actual weights:", weights)
# Calculate the returns of the replicating portfolio
portfolio_returns = np.dot(constituent_returns, weights)
# Calculate the cumulative log-returns for the DAX index and the replicating portfolio
cumulative_returns_dax = np.cumsum(gdaxi_returns)
cumulative_returns_portfolio = np.cumsum(portfolio_returns)
# Calculate the tracking error as the difference between the index and the replicating portfolio
tracking_error = gdaxi_returns - portfolio_returns
# Calculate the risk (standard deviation) of the replicating portfolio
portfolio_risk = np.std(portfolio_returns)
# Calculate the return of the replicating portfolio
portfolio_return = np.sum(portfolio_returns)
# Print the risk and return of the replicating portfolio
print("Risk (Standard Deviation) - Replicating Portfolio:", portfolio_risk)
print("Return - Replicating Portfolio:", portfolio_return)
# Plot the cumulative log-returns of the DAX index and the replicating portfolio
plt.plot(cumulative_returns_dax, label="DAX Index")
plt.plot(cumulative_returns_portfolio, label="Replicating Portfolio")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cumulative Log Returns")
plt.title("Cumulative Log Returns - DAX Index vs. Replicating Portfolio")
plt.show()
# Plot the tracking error
plt.figure()
plt.plot(tracking_error)
plt.xlabel("Time")
plt.ylabel("Tracking Error")
plt.title("Tracking Error - DAX Index vs. Replicating Portfolio vs Tracking error")
# Show the plots
plt.show()
#Plotting all three lines in single graph as required. I did them separately in previous lines to make it clean
plt.plot(cumulative_returns_dax, label='DAX Index')
plt.plot(cumulative_returns_portfolio, label='Replicating Portfolio')
plt.plot(tracking_error, label='Tracking Error')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Cumulative Log-Returns')
plt.title('DAX Index vs. Replicating Portfolio')
plt.show()
"""
Third group of tasks
"""
# Select a subset of constituents for the smart-beta strategy (e.g., first 3 constituents)
selected_constituent_returns = constituent_returns[:, :3]
# Fit a linear regression model using the selected constituents
model = LinearRegression()
model.fit(selected_constituent_returns, gdaxi_returns)
# Extract the coefficients (weights) of the regression model
weights = model.coef_
# Normalize the weights to ensure their sum is 1
weights /= np.sum(weights)
# Calculate the returns of the replicating portfolio
selected_portfolio_returns = np.dot(selected_constituent_returns, weights)
# Calculate the cumulative log-returns for the DAX index and the replicating portfolio
selected_cumulative_returns_portfolio = np.cumsum(selected_portfolio_returns)
# Calculate the tracking error as the difference between the index and the replicating portfolio
selected_tracking_error = gdaxi_returns - selected_portfolio_returns
# Calculate the risk (standard deviation) of the replicating portfolio
selected_portfolio_risk = np.std(portfolio_returns)
# Calculate the return of the replicating portfolio
selected_portfolio_return = np.sum(selected_portfolio_returns)
# Print the risk and return of the replicating portfolio
print("Risk (Standard Deviation) - Replicating Portfolio:", selected_portfolio_risk)
print("Return - Replicating Portfolio:", selected_portfolio_return)
# Plot the cumulative log-returns of the DAX index and the selected replicating portfolio
plt.plot(cumulative_returns_dax, label="DAX Index")
plt.plot(selected_cumulative_returns_portfolio, label="Selected Replicating Portfolio")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Cumulative Log Returns")
plt.title("Cumulative Log Returns - DAX Index vs. Selected Replicating Portfolio")
# Plot the tracking error
plt.figure()
plt.plot(selected_tracking_error)
plt.xlabel("Time")
plt.ylabel("Tracking Error")
plt.title("Tracking Error - DAX Index vs. Selected Replicating Portfolio")
# Show the plots
plt.show()
#Plotting all three in single graph
plt.plot(cumulative_returns_dax, label='DAX Index')
plt.plot(selected_cumulative_returns_portfolio, label='Subset-Based Portfolio')
plt.plot(selected_tracking_error, label='Subset Tracking Error')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Cumulative Log-Returns')
plt.title('DAX Index vs. Subset-Based Portfolio')
plt.show()