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
"""
Task 1
"""
# Parameters
mu = 0.1  # drift per month
sigma = math.sqrt(0.16)  # volatility per month
S_0 = 2.0  # initial cash position in million
# Expected value and standard deviation after one month
E_S_1 = S_0 + mu * 1
SD_S_1 = sigma * math.sqrt(1)
# Expected value and standard deviation after six months
E_S_6 = S_0 + mu * 6
SD_S_6 = sigma * math.sqrt(6)
# Expected value and standard deviation after twelve months
E_S_12 = S_0 + mu * 12
SD_S_12 = sigma * math.sqrt(12)
# Probability of negative cash position after six months
P_neg_S_6 = stats.norm.cdf(-S_0 / (sigma * math.sqrt(6)))
# Proability of negative cash position after twelve months
P_neg_S_12 = stats.norm.cdf(-S_0 / (sigma * math.sqrt(12)))
# Print results
print(f"Expected value after one month: {E_S_1:.2f} million")
print(f"Standard deviation after one month: {SD_S_1:.2f} million")
print(f"Expected value after six months: {E_S_6:.2f} million")
print(f"Standard deviation after six months: {SD_S_6:.2f} million")
print(f"Expected value after twelve months: {E_S_12:.2f} million")
print(f"Standard deviation after twelve months: {SD_S_12:.2f} million")
print(f"Probability of negative cash position after six months: {P_neg_S_6:.4f}")
print(f"Probability of negative cash position after twelve months: {P_neg_S_12:.4f}")
"""
Task 2
"""
#Defining input parameters
S = 220 # today's price of the underlying
K = 220 # strike price
r = 0.1 # risk-free rate (continuous)
sigma = 0.98 # volatility
T = 1 # maturity in years
#Defining Geometric-Brownian motion function
def GBM(S,r,sigma,T,N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    W = np.random.standard_normal(size=N+1)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (r-0.5*sigma**2)*t + sigma*W
    S_T = S*np.exp(X[-1])
    return S_T
#Defining Black-Scholes formula for call option
def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = S*N_d1 - K*np.exp(-r*T)*N_d2
    return call_price
#Calculating price of option using numerical integration
N = 1000 # number of steps
S_T = GBM(S, r, sigma, T, N)
call_price_int = max(S_T - K, 0) * np.exp(-r*T)
#Calculation price of option using Monte Carlo simulation
M = 1000 # number of simulations
payoffs = []
for i in range(M):
    S_T = GBM(S, r, sigma, T, N)
    payoffs.append(max(S_T - K, 0))
call_price_MC = np.mean(payoffs) * np.exp(-r*T)
print("Price of the European call option using numerical integration:", call_price_int)
print("Price of the European call option using Monte Carlo simulation:", call_price_MC)
#Comment- While calculating price for call option, we use Monte Carlo simulation
# over numerical integration when if asset follows complex stochastic process such as 
# such Geometric-Brownian function. When asset follows simple stochastic process like
# constant volatility we may prefer numerical integration
