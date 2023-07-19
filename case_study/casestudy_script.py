#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:23:30 2023

@author: isa
"""

#conda env create -f requirements.yml
conda activate case-env

### Beginning of my code
"""
Importing required packages and setting working directory
"""
import requests 
import os
import os
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import sqlalchemy
from sqlalchemy import create_engine
import statsmodels.api as sm
#!pip install deribit_api
import deribit_api
import datetime
from datetime import datetime
current_dir = os.getcwd()
print("Current Working Directory:", current_dir)
os.chdir('/Users/isa/Downloads/pythondataMLU220213600/220213600/case_study')
"""
Download data
"""
# Calculate the start and end timestamps
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
# Display values
display(data)

data = pd.DataFrame(response['result'])
data['timestamp'] = pd.to_datetime(data['ticks'], unit='ms')
data['instrument_name'] = "BTC-PERPETUAL"
data['resolution'] = 1
# Print the DataFrame with the moving averages
print(data)
"""
Connection to Test Deribit API
"""
# Connect to deribit

ClientID = "IHH9YBRC"
ClientSecret = "50SGOGECZrhekxkwqVTyybEobIJropFm2gWIsvQxBtI"

response = requests.get("https://test.deribit.com/api/v2/public/auth?client_id="+ClientID+"&client_secret="+
                        ClientSecret+"&grant_type=client_credentials")

response_json = response.json()

response_json["result"]
"""
Setting Moving Averages windows
"""
# There is timestamp on the data and its clean so I dont need to do those steps. 
# Moving averages

window_size_1 = 10  # First moving average window size
window_size_2 = 20  # Second moving average window size
window_size_3 = 50  # Third moving average window size

# Calculate the moving averages using the rolling function
data['sma_10'] = data['close'].rolling(window=window_size_1).mean() #10 minute average
data['sma_20'] = data['close'].rolling(window=window_size_2).mean() #20 minute average
data['sma_50'] = data['close'].rolling(window=window_size_3).mean() #50 minute average

"""
Getting current market price
"""
# function to obtain the current market price 

def get_market(instrument):
    
    # Define the URL of the endpoint
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"

    # Define the parameters
    params = {
        "instrument_name": instrument,  # Cryptocurrency to fetch data for
    }

    # Send the GET request; as public request, we don't have to authenticate
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse the response to JSON
        print('Download market data was successful')  # Print the data
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    market_data = pd.DataFrame(data["result"])

    '''
    extract the current market price and round it to the closest ten; 
    this is required as the contract size of Bitcoin Perpetuals are 10 USD 
    (see https://static.deribit.com/files/USDCContractSpecsandmargins.pdf)
    '''
    market_price = int(market_data['mark_price'].values[0].round(-1)) 
    
    print(f"Current market price: {market_price}")
    
    return market_price
"""
Defyning Sell function
"""
def sell():
    # Define the URL of the endpoint
    url = "https://test.deribit.com/api/v2/private/sell"
    instrument_name = "BTC-PERPETUAL"
    # Fetch instrument details to get the contract size
    instrument_details_url = f"https://test.deribit.com/api/v2/public/get_instrument?instrument_name={instrument_name}"
    instrument_details_response = requests.get(instrument_details_url)
    instrument_details = instrument_details_response.json()

    if 'result' in instrument_details:
        contract_size = instrument_details['result']['contract_size']
        print(f"Contract size for {instrument_name}: {contract_size}")

        # Calculate the correct amount as a multiple of the contract size
        amount_multiple = 5  # Adjust this value as needed
        amount = amount_multiple * contract_size

        sell_params = {
            "amount": amount,
            "instrument_name": instrument_name,
            "type": "market"
        }

        # Send the GET request
        response = requests.get(url, params=sell_params, auth=(ClientID, ClientSecret))
        response_json = response.json()

        if 'result' in response_json:
            print("Sell order successfully executed")
        else:
            print(response_json["message"])
    else:
        print("Failed to fetch instrument details")
        
instrument_name = "BTC-PERPETUAL"
# Fetch instrument details to get the contract size
instrument_details_url = f"https://test.deribit.com/api/v2/public/get_instrument?instrument_name={instrument_name}"
instrument_details_response = requests.get(instrument_details_url)
instrument_details = instrument_details_response.json()
contract_size = instrument_details['result']['contract_size']
print(f"Contract size for {instrument_name}: {contract_size}")

 # Calculate the correct amount as a multiple of the contract size= Setting different amountst to buy and sell, and stating numeric value broke the code
amount_multiple = 5  # Adjust this value as needed
amount = amount_multiple * contract_size

sell_params = {
     "amount": amount,
     "instrument_name": instrument_name,
     "type": "market"
 }
"""
Defyning Buy function
"""
def buy(buy_params):
    url = "https://test.deribit.com/api/v2/private/buy"

    # Send the GET request
    response = requests.get(url, params=buy_params, auth=(ClientID, ClientSecret))

    response_json = response.json()

    if 'result' in response_json:
        print("Buy order successfully executed")
    else:
        print("Failed to execute buy order")
# define the parameters 
buy_params = {
    "amount": "50",
    "instrument_name": "BTC-PERPETUAL",
    "label": "test01",
    "type": "market"
    }
"""
Function for Live Trading
"""
# Code to be iterated for a chosen time
start_time = time.time()
end_time = start_time + 60  # 60 seconds = 1 minute, for it to be recorded I run it for 1 minute, so you can see that code is working, when I first applied the code I didnt put end loop, so my last live trading was 28800 which I will later work on 
signals_df = pd.DataFrame(columns=['Timestamp', 'Readable Time', 'Signal', 'Price', 'Amount'])

while True:
    # Check if the current time exceeds the end time
    if time.time() >= end_time:
        break

    # Define buy and sell signals based on moving averages and HP filter
    data['buy_signal'] = np.where((data['close'] > data['sma_10']) & (data['sma_10'] > data['sma_20']) & (data['sma_20'] > data['sma_50']), 1, 0)
    data['sell_signal'] = np.where((data['close'] < data['sma_10']) & (data['sma_10'] < data['sma_20']) & (data['sma_20'] < data['sma_50']), 1, 0)

    capital = 10  # Initial capital
    position = 0  # Current position (0: no position, 1: long position, -1: short position)
    buy_price = 0  # Price at which the position is bought
    sell_price = 0  # Price at which the position is sold
    positions = []  # List to store position details

    for i in range(len(data)):
        # Check if the current time exceeds the end time
        if time.time() >= end_time:
            break

        close_price = data['close'].iloc[i]
        buy_signal = data['buy_signal'].iloc[i]
        sell_signal = data['sell_signal'].iloc[i]

        # Step 5.3: Buy signal
        if buy_signal == 1 and position == 0:
            position = 1  # Enter long position
            buy_price = close_price
            positions.append(('Buy', buy_price))
            buy_amount = buy_params['amount']  # Retrieve the buy amount from the buy_params
            buy(buy_params)
            timestamp = time.time()
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            signals_df = pd.concat([signals_df, pd.DataFrame({'Timestamp': [timestamp], 'Readable Time': [readable_time], 'Signal': ['Buy'], 'Price': [buy_price], 'Amount': [buy_amount]})], ignore_index=True)

        # Step 5.4: Sell signal
        elif sell_signal == 1 and position == 1:
            position = 0  # Exit long position
            sell_price = close_price
            positions.append(('Sell', sell_price))
            sell_amount = sell_params['amount']  # Retrieve the sell amount from the sell_params
            sell()
            timestamp = time.time()
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            signals_df = pd.concat([signals_df, pd.DataFrame({'Timestamp': [timestamp], 'Readable Time': [readable_time], 'Signal': ['Sell'], 'Price': [sell_price], 'Amount': [sell_amount]})], ignore_index=True)

            # Calculate profit/loss
            profit_loss = sell_price - buy_price
            capital += profit_loss
            time.sleep(30)

# Export the signals DataFrame to a CSV file
signals_df.to_csv('signals.csv', mode='a', header=False, index=False)
print("Signals exported to signals.csv file.")
"""
Closing Open positions
"""
# When I can ran the code for certain time frame, I had limit issues, so closing open positions helped me to continue operate
def close_position(instrument_name):
    # Define the URL of the endpoint 
    url = "https://test.deribit.com/api/v2/private/close_position"

    # Define the parameters
    params = {
        "instrument_name": instrument_name,
    }

    # Send the GET request with authentication
    response = requests.get(url, params=params, auth=(ClientID, ClientSecret))
    response_json = response.json()

    if 'result' in response_json:
        print("Position successfully closed")
    else:
        print("Failed to close position")

# Call the close_position function with the instrument_name parameter
instrument_name = "BTC-PERPETUAL"
close_position(instrument_name)  #It succeded to close positions when I first run it, though

"""
Backtesting
"""
# Step 1: Initialize variables
capital = 100000  # Initial capital
position = 0  # Current position (0: no position, 1: long position, -1: short position)
entry_price = 0  # Price at which the position is entered
exit_price = 0  # Price at which the position is exited
positions = []  # List to store position details

# Step 2: Read signals from signals.csv
tradinghistory_df = pd.read_csv('livetradedata.csv')
# Step 3: Iterate over each row in the signals DataFrame
for i in range(len(tradinghistory_df)):
    signal = tradinghistory_df['Side'].iloc[i]
    price = tradinghistory_df['Price'].iloc[i]
    
    # Step 4: Buy signal
    if signal == 'buy' and position == 0:
        position = 1  # Enter long position
        entry_price = price
        positions.append(('buy', entry_price))
        
    # Step 5: Sell signal
    elif signal == 'sell' and position == 1:
        position = 0  # Exit long position
        exit_price = price
        positions.append(('sell', exit_price))
        
        # Calculate profit/loss
        profit_loss = exit_price - entry_price
        capital += profit_loss
        
        # Reset entry_price for next position
        entry_price = 0
    
# Step 6: Calculate the final capital and profit/loss percentage
final_capital = capital
profit_loss_percentage = ((final_capital - 100000) / 100000) * 100
#Step 7: Calculate the Sharpe ratio
risk_free_rate = 0.01  
benchmark_return = 0.05  
trading_days_per_year = 252  

# Calculate the total return of the strategy
total_return = (final_capital - 100000) / 100000

# Calculate the benchmark's total return (annualized)
benchmark_total_return = (1 + benchmark_return) ** (trading_days_per_year / len(tradinghistory_df)) - 1

# Calculate the annualized return and volatility of the strategy
annualized_return = (1 + total_return) ** (trading_days_per_year / len(tradinghistory_df)) - 1
annualized_volatility = tradinghistory_df['Price'].pct_change().std() * (trading_days_per_year ** 0.5)

# Calculate the Sharpe ratio
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

print("Final Capital:", final_capital)
print("Profit/Loss Percentage:", profit_loss_percentage)
print("Sharpe Ratio:", sharpe_ratio)
"""
Equity Curve
"""
# Create a DataFrame for trade history with date and time
trade_history = pd.DataFrame(positions, columns=['Side', 'Price'])
trade_history['Timestamp'] = data['timestamp'].iloc[:len(trade_history)]
trade_history.set_index('Timestamp', inplace=True)

# Calculate equity curve
trade_history['Position'] = trade_history['Side'].apply(lambda x: 1 if x == 'Buy' else -1)
trade_history['Equity Curve'] = trade_history['Position'].cumsum() * trade_history['Price']
trade_history['Equity Curve'] += 100000  # Add initial capital to the equity curve

# Plot equity curve
plt.plot(trade_history.index, trade_history['Equity Curve'])
plt.xlabel('Trade Date')
plt.ylabel('Equity Curve')
plt.title('Equity Curve with Trade Date and Time')
plt.grid(True)
plt.show()

"""
Calculating Cumulative returns
"""
# Calculate cumulative returns for each trade
cumulative_returns = []
trade_returns = []
timestamps = []
for i in range(1, len(positions), 2):
    buy_price = positions[i-1][1]
    sell_price = positions[i][1]
    trade_return = (sell_price - buy_price) / buy_price
    trade_returns.append(trade_return)
    cumulative_return = (1 + sum(trade_returns)) if len(cumulative_returns) == 0 else cumulative_returns[-1] * (1 + trade_return)
    cumulative_returns.append(cumulative_return)
    timestamps.append(data['timestamp'].iloc[i])

# Convert timestamps to human-readable format
readable_timestamps = [timestamp.strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

# Create a DataFrame to store cumulative returns and timestamps
returns_df = pd.DataFrame({'Timestamp': readable_timestamps, 'Cumulative Return': cumulative_returns})

# Print the DataFrame
print(returns_df)

"""
Calcularting Risk metrics
"""
 # Calculate the cumulative return
cumulative_return = returns_df['Cumulative Return'].iloc[-1]

# Calculate the total trading days (considered as 1 trading day for less than a day period)
total_trading_days = 1

# Calculate the annualized return
annualized_return = (cumulative_return + 1) ** (365 / total_trading_days) - 1

# Calculate the daily returns
returns_df['Date'] = pd.to_datetime(returns_df['Timestamp'])
returns_df['Daily Return'] = returns_df['Cumulative Return'].pct_change()

# Calculate the daily volatility
daily_volatility = returns_df['Daily Return'].std()

# Calculate the annualized volatility
annualized_volatility = daily_volatility * np.sqrt(365)

# Print the results
print("Annualized Return:", annualized_return)
print("Annualized Volatility:", annualized_volatility)

# Step 5.8: Plot date and cumulative returns
plt.plot(returns_df['Timestamp'], returns_df['Cumulative Return'])
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns Over Time')
plt.show()

"""
Calculating Log Returns
"""
# Read the live trade data from 'livetradedata.csv'
tradinghistory_df = pd.read_csv('livetradedata.csv')

# Calculate log returns for each trade
log_returns = []
timestamps = []
for i in range(1, len(tradinghistory_df), 2):
    buy_price = tradinghistory_df['Price'].iloc[i - 1]
    sell_price = tradinghistory_df['Price'].iloc[i]
    trade_prices = tradinghistory_df['Price'].iloc[i - 1 : i + 1].reset_index(drop=True)
    trade_log_returns = np.log(trade_prices / trade_prices.shift(1))
    log_returns.extend(trade_log_returns)
    timestamps.extend(tradinghistory_df['Date'].iloc[i - 1 : i + 1].reset_index(drop=True))

# Convert timestamps to human-readable format
readable_timestamps = [timestamp for timestamp in timestamps]

# Create a DataFrame to store log returns and timestamps
logreturns_df = pd.DataFrame({'Timestamp': readable_timestamps, 'Log Return': log_returns})

# Print the DataFrame
print(logreturns_df)

def calculate_volatility(log_returns):
    log_returns = np.nan_to_num(log_returns)  # Replace NaN values with zeros
    volatility = np.std(log_returns)  # Calculate standard deviation of log returns
    return volatility
log_returns_asset = logreturns_df['Log Return'].values
log_returns_market = logreturns_df['Log Return'].values
volatility = calculate_volatility(log_returns_asset)
print("Volatility:", volatility)

logreturns_df.dropna(inplace=True)
plt.plot(logreturns_df['Timestamp'], logreturns_df['Log Return'], label='Log Return')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.title('Log Returns Over Time')
# Display the plot
plt.show()
"""
Plotting buy and sell signal results over narket price based on live trading data 
"""
buy_signals = tradinghistory_df[tradinghistory_df['Side'] == 'buy']
sell_signals = tradinghistory_df[tradinghistory_df['Side'] == 'sell']


# Plot buy and sell signals
plt.plot(tradinghistory_df['Date'], tradinghistory_df['Price'], label='Market Price')
plt.scatter(buy_signals['Date'], buy_signals['Price'], color='g', marker='^', label='Buy Signal')
plt.scatter(sell_signals['Date'], sell_signals['Price'], color='r', marker='v', label='Sell Signal')

# Set labels and title
plt.xlabel('Timestamp')
plt.ylabel('Signal')
plt.title('Buy and Sell Signals')

# Add a legend
plt.legend()

# Display the plot
plt.show()

###
### End of the Code
conda deactivate
conda env export > requirements.yml
