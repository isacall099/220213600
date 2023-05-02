#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:07:43 2023
@author: isa
"""
"""
Importing packages
"""
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import norm
from scipy.integrate import quad
"""
Task 1
"""
# Define the parameters
initial_value = 2.0
drift = 0.1
variance = 0.16

# Calculate the expected value and standard deviation
t1 = 1  # time period in months
t6 = 6
t12 = 12

expected_value_1m = initial_value + drift * t1
std_dev_1m = math.sqrt(variance * t1)

expected_value_6m = initial_value + drift * t6
std_dev_6m = math.sqrt(variance * t6)

expected_value_12m = initial_value + drift * t12
std_dev_12m = math.sqrt(variance * t12)
# Calculate the probability of a negative cash position in 6 months
x6 = (0 - expected_value_6m) / std_dev_6m
prob_neg_6m = norm.cdf(x6)
# Calculate the probability of a negative cash position in 12 months
x12 = (0 - expected_value_12m) / std_dev_12m
prob_neg_12m = norm.cdf(x12)
# Print results
print("Expected value in 1 month:", expected_value_1m)
print("Standard deviation in 1 month:", std_dev_1m)
print("Expected value in 6 months:", expected_value_6m)
print("Standard deviation in 6 months:", std_dev_6m)
print("Expected value in 12 months:", expected_value_12m)
print("Standard deviation in 12 months:", std_dev_12m)
print("Probability of a negative cash position in 6 months:", prob_neg_6m)
print("Probability of a negative cash position in 12 months:", prob_neg_12m)

#Commets- using "0-expected value" is more general approach, however initial_value-expected value
# is a specific approach where we are interested in the probability of being below the initial cash position
"""
Task 2
"""

# Define parameters
S_0 = 220 # initial price of underlying asset
K = 220 # strike price
sigma = 0.98 # volatility of underlying asset
r = 0.1 # risk-free interest rate (continuous)
T = 1 # time to maturity
N = 1000 # number of time steps for GBM simulation
M = 1000 # number of simulations for Monte Carlo

# Function to simulate stock price path using Geometric Brownian Motion
def GBM(S_0, r, sigma, T, N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    W = np.random.standard_normal(size=N+1)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (r-0.5*sigma**2)*t + sigma*W
    S = S_0*np.exp(X)
    return S

# Function for the payoff of a call option
def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0)

# Black-Scholes-Merton formula for d1 and d2
d1 = (np.log(S_0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# Function for the Black-Scholes-Merton call price
def black_scholes_call(S_0, K, r, sigma, T):
    d1 = (np.log(S_0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = S_0*N_d1 - K*np.exp(-r*T)*N_d2
    return call_price

# Calculate the Black-Scholes-Merton call option price
call_price_BS = black_scholes_call(S_0, K, r, sigma, T)

# Calculate the call option price using numerical integration
S_T = GBM(S_0, r, sigma, T, N)
call_price_integration = call_payoff(S_T[-1], K) * np.exp(-r * T)

# Calculate the call option price using Monte Carlo simulation
payoffs = []
for i in range(M):
    S_T = GBM(S_0, r, sigma, T, N)
    payoffs.append(call_payoff(S_T[-1], K))
call_price_MC = np.mean(payoffs) * np.exp(-r * T)

# Print the results
print("Black-Scholes-Merton call option price: ", call_price_BS)
print("Call option price using numerical integration: ", call_price_integration)
print("Call option price using Monte Carlo simulation: ", call_price_MC)

#Comment- While calculating price for call option, we use Monte Carlo simulation
# over numerical integration when if asset follows complex stochastic process such as 
# such Geometric-Brownian function. When asset follows simple stochastic process like
# constant volatility we may prefer numerical integration