#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:58:14 2023

@author: isa
"""
import requests 
import pandas as pd
import time
import json
from datetime import datetime


ClientID = "IHH9YBRC"
ClientSecret = "wQvGodAP34oKfCpXZFwYIzWakR88tSkJZ_sl-7L1paM"

# Retrieve access token
response = requests.post("https://test.deribit.com/api/v2/public/auth", data={"grant_type": "client_credentials", "client_id": ClientID, "client_secret": ClientSecret})
response_json = response.json()
access_token = response_json["result"]["access_token"]

# Define the URL of the endpoint
url = "https://test.deribit.com/api/v2/public/get_last_trades_by_instrument"

# Define the parameters
params = {
    "instrument_name": "BTC-PERPETUAL",
    "count": 1
}

# Send the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Parse the response to JSON
    print(data)  # Print the data
else:
    print(f"Failed to retrieve data: {response.status_code}")

# Retrieve the last traded price from the response
last_trade_price = data["result"][0]["price"]

print(last_trade_price)

# Define the URL of the endpoint 
url = "https://test.deribit.com/api/v2/private/sell"

# Define the headers with the access token
headers = {
    "Authorization": f"Bearer {access_token}"
}

# Define the payload
payload = {
    "amount": "1",  # Number of contracts to sell
    "instrument_name": "BTC-PERPETUAL",
    "label": "test01",
    "type": "market"
}

# Send the POST request with the payload and headers
response = requests.post(url, json=payload, headers=headers)
response_json = response.json()

print(response_json['result'])
