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
# Calculate the mean and standard deviation of log-returns
mu = gdaxi_returns.mean()
sigma = gdaxi_returns.std()

# Set the significance level and time horizon
alpha = 0.95
t = 1

# Calculate the 1-day VaR using the parametric method
var_parametric = norm.ppf(1 - alpha) * sigma * np.sqrt(t)

# Calculate the 1-day VaR using the historical method
var_historical = -np.percentile(gdaxi_returns, alpha * 100)

# Calculate the 1-day VaR using Monte Carlo simulation
num_simulations = 10000
simulated_returns = np.random.normal(mu, sigma, size=(t, num_simulations))
var_monte_carlo = -np.percentile(np.sum(simulated_returns, axis=0), alpha * 100)

# Plot a histogram of the returns
plt.hist(gdaxi_returns, bins=30, density=True, alpha=0.6)

# Add a line plot of the normal distribution
x = np.linspace(gdaxi_returns.min(), gdaxi_returns.max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')

# Add indicators for the 3 VaRs
plt.axvline(-var_parametric, color='b', linestyle='--', label='Parametric VaR')
plt.axvline(-var_historical, color='g', linestyle='--', label='Historical VaR')
plt.axvline(-var_monte_carlo, color='purple', linestyle='--', label='Monte Carlo VaR')

# Set the title and axis labels
plt.title('Distribution of .GDAXI Returns')
plt.xlabel('Returns')
plt.ylabel('Density')

# Add a legend and show the plot
plt.legend()
plt.show()
