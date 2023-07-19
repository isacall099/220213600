#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:11:29 2023

@author: isa
"""
conda activate case-env

current_dir = os.getcwd()
print("Current Working Directory:", current_dir)
os.chdir('/Users/isa/Downloads/pythondataMLU220213600/220213600/case_study')
import requests 
import os
import os
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sqlalchemy
from sqlalchemy import create_engine
import statsmodels.api as sm
#!pip install deribit_api
import deribit_api


# Step 1: Obtain historical market data from Test Deribit or your preferred source
# Download data
url = 'https://test.deribit.com/api/v2/'

msg = {
        "jsonrpc" : "2.0",
        "id" : 833,
        "method" : "public/get_tradingview_chart_data",
        "params" : {
        "instrument_name" : "BTC-PERPETUAL",
        "end_timestamp": int(time.time() * 1000),  # In Epoch milliseconds
        "start_timestamp": int((time.time() - 1e6) * 1000), # In Epoch milliseconds
        "resolution" : "1"  # Minute data
        }
    }
response = requests.post(url, json=msg)
response = response.json()

data = pd.DataFrame(response['result'])
data['timestamp'] = pd.to_datetime(data['ticks'], unit='ms')
data['instrument_name'] = "BTC-PERPETUAL"
data['resolution'] = 1
# Stuff we don't need
data.drop(columns=['ticks', 'status'], inplace=True)
# Display values
display(data)
# Step 2: Calculate the moving average
window_size = 20  # Adjust the window size as desired
data['MA'] = data['close'].rolling(window_size).mean()

# Step 3: Generate buy and sell signals
data['Signal'] = np.where(data['close'] > data['MA'], 1, -1)
data['Position'] = data['Signal'].diff()

# Step 4: Plot the buy and sell signals
plt.figure(figsize=(12, 6))
plt.plot(data['close'], label='Price')
plt.plot(data['MA'], label=f'{window_size}-day Moving Average')
plt.scatter(data.index[data['Position'] == 1], data['close'][data['Position'] == 1], marker='^', color='g', label='Buy Signal')
plt.scatter(data.index[data['Position'] == -1], data['close'][data['Position'] == -1], marker='v', color='r', label='Sell Signal')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Buy and Sell Signals based on Moving Average')
plt.legend()
plt.grid(True)
plt.show()


# Step 5: Backtesting
capital = 100000  # Initial capital
position = 0  # Current position (0: no position, 1: long position, -1: short position)
entry_price = 0  # Price at which the position is entered
exit_price = 0  # Price at which the position is exited
positions = []  # List to store position details
cumulative_returns = []  # List to store cumulative returns

# Step 5.1: Iterate over each row in the data DataFrame
for i in range(len(data)):
    price = data['close'].iloc[i]
    signal = data['Position'].iloc[i]
    
    # Step 5.2: Buy signal
    if signal == 1 and position == 0:
        position = 1  # Enter long position
        entry_price = price
        positions.append(('Buy', entry_price))
        
    # Step 5.3: Sell signal
    elif signal == -1 and position == 1:
        position = 0  # Exit long position
        exit_price = price
        positions.append(('Sell', exit_price))
        
        # Calculate profit/loss
        profit_loss = exit_price - entry_price
        capital += profit_loss
        cumulative_returns.append(capital)
        
# Step 5.4: Calculate the final capital and profit/loss percentage
final_capital = capital
profit_loss_percentage = ((final_capital - 100000) / 100000) * 100

print("Final Capital:", final_capital)
print("Profit/Loss Percentage:", profit_loss_percentage)

# Step 5.5: Calculate cumulative returns for the last trade
final_capital = capital
cumulative_returns.append(final_capital / 100000 - 1)  # Calculate and store cumulative returns
print(cumulative_returns)

### End of the Code
conda deactivate

conda env export > requirements_sma_2ndStrategy.yml
