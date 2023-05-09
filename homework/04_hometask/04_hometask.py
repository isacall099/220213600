"""
Importing relevant packages
"""
#pip install pandas
#pip install scipy
#pip install matplotlib
import pandas as pd
import numpy as np
# !pip install --upgrade numpy # for erfinv attribute there is need to upgrade numpy
# conda update numpy #I had to run it as erfinv i was still missing
#from scipy.special import erfinv
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
#import seaborn as sns
import os


"""
First group of tasks
"""
# Set working directory 
os.chdir("/Users/isa/Downloads/pythondataMLU220213600/220213600/homework/04_hometask")
# Import data
prices = pd.read_csv('02_python_data.csv', index_col='Date', parse_dates=True)
# Select the columns containing the stock prices
portfolio = prices.drop(columns=".GDAXI")
# Convert the data type of the selected columns to float
#portfolio = stock_prices.astype(float) #if type is not float, change it
# Calculate the log-returns of the ".GDAXI" column using numpy and pandas
gdaxi_returns = np.log(prices[".GDAXI"] / prices[".GDAXI"].shift(1))
gdaxi_returns = gdaxi_returns.dropna()
# Calculate the cumulative log-returns of the ".GDAXI" column using numpy and pandas
gdaxi_cum_returns = np.cumsum(gdaxi_returns)
# Calculate the equally weighted portfolio of the constituents (excluding ".GDAXI") using numpy and pandas
portfolio_weight = np.full(portfolio.shape[1], 1 / portfolio.shape[1])
portfolio_weighted_returns = np.sum(portfolio.pct_change() * portfolio_weight, axis=1)
portfolio_weighted_returns = portfolio_weighted_returns.dropna()
# Calculate the log-returns and cumulative log-returns of the equally weighted portfolio using numpy and pandas
portfolio_weighted_log_returns = np.log(1 + portfolio_weighted_returns)
portfolio_weighted_cum_returns = np.cumsum(portfolio_weighted_log_returns)
# Plot the cumulative log-returns of both ".GDAXI" and the equally weighted portfolio using matplotlib
plt.plot(gdaxi_cum_returns, label=".GDAXI")
plt.plot(portfolio_weighted_cum_returns, label="Equally Weighted Portfolio")
plt.title("Cumulative Log-Returns of .GDAXI and Equally Weighted Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Log-Returns")
plt.legend()
plt.show()

"""
Second group of tasks
"""
# Define the confidence level and time horizon for VaR calculations
confidence_level = 0.95
time_horizon = 1  # in days

# Calculate VaR using historical simulation for both portfolios
VaR_1day_95_gdaxi = np.percentile(gdaxi_cum_returns, 100 * (1 - confidence_level))
VaR_1day_95_portfolio = np.percentile(portfolio_weighted_cum_returns, 100 * (1 - confidence_level))
# Calculate VaR using the parametric method for both portfolios
mean_gdaxi = np.mean(gdaxi_cum_returns)
std_gdaxi = np.std(gdaxi_cum_returns)
VaR_1day_95_gdaxi_parametric = -mean_gdaxi - stats.norm.ppf(confidence_level) * std_gdaxi

mean_portfolio = np.mean(portfolio_weighted_cum_returns)
std_portfolio = np.std(portfolio_weighted_cum_returns)
VaR_1day_95_portfolio_parametric = -mean_portfolio - stats.norm.ppf(confidence_level) * std_portfolio

# Calculate VaR using Monte Carlo simulation for both portfolios
N_sim = 10000
simulated_gdaxi_returns = np.random.choice(gdaxi_cum_returns, size=N_sim)
simulated_portfolio_returns = np.random.choice(portfolio_weighted_cum_returns, size=N_sim)

VaR_1day_95_gdaxi_simulated = np.percentile(simulated_gdaxi_returns, 100 * (1 - confidence_level))
VaR_1day_95_portfolio_simulated = np.percentile(simulated_portfolio_returns, 100 * (1 - confidence_level))

# Print the results
print("1-day VaR at 95% confidence level (Historical Simulation)")
print(".GDAXI Portfolio:", VaR_1day_95_gdaxi)
print("Equally Weighted Portfolio:", VaR_1day_95_portfolio)
print("\n1-day VaR at 95% confidence level (Parametric)")
print(".GDAXI Portfolio:", VaR_1day_95_gdaxi_parametric)
print("Equally Weighted Portfolio:", VaR_1day_95_portfolio_parametric)
print("\n1-day VaR at 95% confidence level (Monte Carlo)")
print(".GDAXI Portfolio:", VaR_1day_95_gdaxi_simulated)
print("Equally Weighted Portfolio:", VaR_1day_95_portfolio_simulated)

"""
Third group of tasks
"""
#
confidence_level = 0.95

# Generate random daily returns for a two stock portfolio
np.random.seed(42)
num_days = 252*10
returns_A = np.random.normal(loc=0.001, scale=0.02, size=num_days)
returns_B = np.random.normal(loc=0.0005, scale=0.015, size=num_days)
# Create a DataFrame with the returns
returns = pd.DataFrame({'A': returns_A, 'B': returns_B})
# Calculate portfolio weights (50% each)
weights = np.array([0.5, 0.5])
# Calculate portfolio returns
portfolio_returns = returns.dot(weights)

# Calculate the 1-day VaR at a 95% confidence level (historical simulation)
confidence_level = 0.95
VaR_1day_95 = np.percentile(portfolio_returns, 100 * (1 - confidence_level))

# Calculate the 1-day VaR at a 95% confidence level (parametric)
mean_returns = returns.mean()
cov_matrix = returns.cov()
portfolio_mean = np.dot(weights, mean_returns)
portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
alpha = stats.norm.ppf(confidence_level)
VaR_1day_95_parametric = -portfolio_mean - alpha * portfolio_std

# Calculate the 1-day VaR at a 95% confidence level (Monte Carlo)
Nsim = 10000
# Simulate daily returns
simulated_returns = np.random.normal(portfolio_mean, portfolio_std, Nsim)
# Note how this is similar to historical simulation
VaR_1day_95_simulated = np.percentile(simulated_returns, 100 * (1 - confidence_level))

# Plot the portfolio returns and the VaR
plt.hist(portfolio_returns, bins=50, density=True, color='grey', alpha=0.75, label='Portfolio Returns')
plt.axvline(x=VaR_1day_95, color='red', linestyle='--', label=f'Historical: {VaR_1day_95:.4f}')
plt.axvline(x=VaR_1day_95_parametric, color='green', linestyle='--', label=f'Parametric: {VaR_1day_95_parametric:.4f}')
plt.axvline(x=VaR_1day_95_simulated, color='blue', linestyle='--', label=f'Simulated: {VaR_1day_95_simulated:.4f}')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Portfolio Returns and 1-day VaR at 95% Confidence Level')
plt.legend()
plt.show()