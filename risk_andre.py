import pandas as pd
from scipy.stats import qmc, norm
import numpy as np

# Load the historical data for SOFR curve and the share prices of AAPL, MSFT, F, BAC
sofr_data = pd.read_excel('sofr.xlsx')
aapl_data = pd.read_excel('aapl.xlsx')
msft_data = pd.read_excel('msft.xlsx')
f_data    = pd.read_excel('f.xlsx')
bac_data  = pd.read_excel('bac.xlsx')

# check the first few rows of each dataset to verify the structure
print(sofr_data.head(), aapl_data.head(), msft_data.head(), f_data.head(), bac_data.head())

# compute daily returns for SOFR rates and each of the stocks
# for SOFR, we will use the 1Y-10Y tenor rates to represent the swap's floating rate leg
tenors = [str(i)+'Y' for i in range(1, 11)]
sofr_rates = sofr_data[sofr_data['Tenor'].isin(tenors)].iloc[:, 2:].T
sofr_rates.columns = tenors

print("sofr_rates: ")
print( sofr_rates )

# compute daily returns for the stocks
aapl_returns = aapl_data['Adj Close']
msft_returns = msft_data['Adj Close']
f_returns  	 = f_data['Adj Close']
bac_returns  = bac_data['Adj Close']

# prep a DataFrame for portfolio daily returns
portfolio = pd.DataFrame({
    'SOFR': sofr_rates.iloc[:, 0],
    'AAPL': aapl_returns.values,
    'MSFT': msft_returns.values,
    'F': f_returns.values,
    'BAC': bac_returns.values
})

# Convert the tenor labels from strings to integers for calculation
tenors = [int(col.replace('Y', '')) for col in sofr_rates.columns]

# Calculate discount rates using e^-rT for each tenor and rate
discount_rates = sofr_rates.copy()

for tenor, rate in zip(tenors, sofr_rates.columns):
    discount_rates[rate] = np.exp(-sofr_rates[rate] * tenor)

print("discount_rates: ")
print(discount_rates)

swap_notional = 100000000 # 100 million
fixed_leg_ir = 0.042 # 4.2%

def PV_payer_swap0(discount_rate_today, fixed_leg_ir, swap_notional):
    float_leg =  (1 - discount_rate_today[-1])
    fix_leg = fixed_leg_ir * sum(discount_rate_today)
    payer_swap_PV  = swap_notional * (float_leg - fix_leg)

    return payer_swap_PV

def PV_payer_swap(sofr_rates, discount_rates, fixed_leg_ir, swap_notional, start_date):
    """
    Calculate the Present Value (PV) of a payer swap using SOFR rates for floating payments
    and a fixed interest rate for fixed payments, both discounted to present value.
    
    Parameters:
    - sofr_rates: DataFrame with SOFR rates.
    - discount_rates: DataFrame with discount rates.
    - fixed_leg_ir: Fixed interest rate of the swap.
    - swap_notional: Notional amount of the swap.
    - start_date: The date from which to start the calculation (e.g., '2023-10-30').
    
    Returns:
    - The net present value (PV) of the payer swap.
    """
    # Initialize PVs of both legs to zero
    pv_fixed_leg = 0
    pv_floating_leg = 0
    
    # Use the first available date in sofr_rates to start calculations
    for year, tenor in enumerate(['1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y'], start=1):
        fixed_payment = fixed_leg_ir * swap_notional
        floating_payment = sofr_rates.loc[start_date, tenor] * swap_notional
        
        # Discount both fixed and floating payments to present value
        discount_factor = discount_rates.loc[start_date, tenor]
        pv_fixed_leg += fixed_payment * discount_factor
        pv_floating_leg += floating_payment * discount_factor
    
    # Calculate the net PV of the swap from the payer's perspective
    payer_swap_pv = pv_floating_leg - pv_fixed_leg
    
    return payer_swap_pv



# Convert the index of the sofr_rates DataFrame to datetime
sofr_rates.index = pd.to_datetime(sofr_rates.index)
discount_rates.index = pd.to_datetime(discount_rates.index)
today_date = '2023-10-30'  # Starting date for calculation
print("111", sofr_rates.loc[today_date, '1Y'])

# compute current value of swap
discount_rate_today = discount_rates.loc[today_date].to_numpy()
print("discount_rates000:" ,discount_rates)
print("discount_rate_today111: ", discount_rate_today)
# payer_swap_today = PV_payer_swap(sofr_rates, discount_rates, fixed_leg_ir, swap_notional, start_date)

swap_value_today = PV_payer_swap0(discount_rate_today, fixed_leg_ir, swap_notional)
print("swap_value_today: ", swap_value_today)



stock_prices = portfolio.drop(columns=['SOFR'])
stock_prices.index = pd.to_datetime(stock_prices.index)
print("stock_prices:" , stock_prices)
stock_prices_today = stock_prices.loc[today_date]
print("stock_prices_today: ", stock_prices_today)

# number of shares held today per stock with 1M USD
stock_values_today = np.array([1000000] * 4)
num_shares_held_today = stock_values_today / stock_prices_today
print("num_shares_held_today: ", num_shares_held_today)
portfolio_value_today = swap_value_today + sum(stock_values_today)

print(f'Stock Value Today (30 Oct 2023): ${round(sum(stock_values_today), 2):,}')
print(f'Swap Value Today (30 Oct 2023): ${round(swap_value_today, 2):,}')
print(f'Portfolio Value Today (30 Oct 2023): ${round(portfolio_value_today, 2):,}')


stock_prices_array = stock_prices.to_numpy()
print("stock_prices_array: ", stock_prices_array)

# compute daily returns
for stock_price in stock_prices.columns:
    stock_prices[f'{stock_price}_daily_returns'] = stock_prices[stock_price].pct_change()

stock_prices = stock_prices.dropna()

print("check stock_prices daily returns")
print(stock_prices)

# Splitting the DataFrame
prices_columns  = stock_prices.columns[~stock_prices.columns.str.contains('_daily_returns')]
returns_columns = stock_prices.columns[stock_prices.columns.str.contains('_daily_returns')]

# Creating NumPy arrays
prices_array  = stock_prices[prices_columns].to_numpy()
returns_array = stock_prices[returns_columns].to_numpy()

# print("prices_array: ", prices_array)
# print("returns_array: ", returns_array)

returns_mean = returns_array.mean(axis = 0)
returns_std = returns_array.std(axis = 0)
returns_cov = np.cov(returns_array, rowvar=False)
print("returns_mean: ", returns_mean)
print("returns_std: ", returns_std)
print("returns_cov: ", returns_cov)

# parametric stock statistics
parametric_stock_mean = (returns_mean * stock_values_today).sum()
parametric_stock_std = np.sqrt(np.matmul(np.matmul(stock_values_today, returns_cov), stock_values_today.T))
print("parametric stock mean: ", parametric_stock_mean)
print("parametric_stock_std: ", parametric_stock_std )

basis_point_change = 0.0001

swap_value_change = []

sofr_rates_change = sofr_rates.diff().dropna()

# filter for rates on that day
sofr_today = sofr_rates.loc[today_date].to_numpy()
print("sofr_today: ", sofr_today)

for tenor in range(len(sofr_rates.columns)):
    pv01_sofr_values = []
    for j in range(len(sofr_rates.columns)):
        # For each tenor, the SOFR rate for that tenor increases by 1 bp
        # keeping other rates constant
        adjust_sofr_rate_tenor_add_1bp = basis_point_change if j == tenor else 0
        pv01_sofr_tenor = sofr_today[j] + adjust_sofr_rate_tenor_add_1bp
        pv01_sofr_values.append( pv01_sofr_tenor )

    # compute new discount factors based on the adjusted SOFR rates 
    # to reflect the impact of the rate changes
    discount_factor_adj_sofr_rates = []

    
    for i in range(len(pv01_sofr_values)):
        discount_factor_adj_sofr_tenor = np.exp(- pv01_sofr_values[i] * (i + 1))
        discount_factor_adj_sofr_rates.append( discount_factor_adj_sofr_tenor )
    # print("discount_factor_adj_sofr_rates111: ", discount_factor_adj_sofr_rates)
    # compute how the value of the payer swap changes in response to the 1 bp rate increase for each tenor.
    swap_value_change_tenor = PV_payer_swap0(discount_factor_adj_sofr_rates, fixed_leg_ir, swap_notional) - swap_value_today
    swap_value_change.append( swap_value_change_tenor )

# convert bp to rate %
print("swap_value_change: ", swap_value_change)
print("basis_point_change: ", basis_point_change)
swap_value_change = np.array(swap_value_change)
pv01 = swap_value_change / basis_point_change
print("pv01: ", pv01)
# mean change in rates
sofr_mean = sofr_rates_change.mean(axis = 0).to_numpy()

# std change in rates 
sofr_std = sofr_rates_change.std(axis = 0).to_numpy()

# covariance matrixs of rates
sofr_cov = (sofr_rates_change).cov(numeric_only = False).to_numpy()

# Parametric swap statistics
parametric_swap_mean = (pv01 * sofr_mean).sum()
parametric_swap_std = np.sqrt(np.matmul(np.matmul(pv01, sofr_cov), pv01.T))
print("parametric_swap_mean: ", parametric_swap_mean)
print("parametric_swap_std: ", parametric_swap_std)

# Calculate 1-day 95% VaR for the portfolio
# Z-Score for 95% confidence interval
z_score_95 = norm.ppf(0.95)

# Parametric VaR
parametric_var_95_swap = -(parametric_swap_mean - z_score_95 * parametric_swap_std)
parametric_var_95_stock = -(parametric_stock_mean - z_score_95 * parametric_stock_std)
print("parametric_var_95 swap: ", parametric_var_95_swap)
print("parametric_var_95_stock: ", parametric_var_95_stock)


def simulate_sobol_returns(dimension, power, mean, std):
    """
    Generate Sobol sequences and transform to normal distribution.
    
    Parameters:
    - dimension: Number of dimensions for the Sobol sequence.
    - power: Determines the length of the sequence, 2^m.
    - mean: Mean values for the normal distribution.
    - std: Standard deviation values for the normal distribution.
    
    Returns:
    - Simulated returns following the specified normal distribution.
    """
    sampler = qmc.Sobol(d=dimension, scramble=False)
    samples = sampler.random_base2(m=power)
    samples = np.delete(samples, 0, axis=0)  # Remove the first all-zero sample
    
    # Transform uniform to normal
    normal_samples = norm.ppf(samples, loc=mean, scale=std)
    
    return normal_samples

def apply_cholesky(simulated_returns, correlation_matrix):
    """
    Apply Cholesky decomposition to simulate correlated returns.
    
    Parameters:
    - simulated_returns: Array of simulated returns to correlate.
    - correlation_matrix: Correlation matrix of the returns.
    
    Returns:
    - Correlated simulated returns.
    """
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    correlated_returns = np.dot(simulated_returns, cholesky_matrix.T)
    
    return correlated_returns

stocks_dimension = 4  # For 4 stocks
sofr_dimension = len(sofr_rates_change.columns)
power = 20
# Simulate stock returns
stock_simulation = simulate_sobol_returns(stocks_dimension, power, returns_mean, returns_std)
# print("returns_array: ", returns_array)
stock_corr_matrix = np.corrcoef(returns_array, rowvar=False)
stock_simulation = apply_cholesky(stock_simulation, stock_corr_matrix)

print("stock_simulation: ", stock_simulation)

# Simulate SOFR changes
sofr_simulation = simulate_sobol_returns(sofr_dimension, power, sofr_mean, sofr_std)
sofr_corr_matrix = sofr_rates_change.corr(numeric_only=False).to_numpy()
sofr_simulation = apply_cholesky(sofr_simulation, sofr_corr_matrix)

print("sofr_simulation: ", sofr_simulation)

confidence_level = 95

# generate new SOFR rates under different simulation scenarios
print("sofr_simulation")
print(sofr_simulation)
print(sofr_simulation.shape)
print("sofr_today: ", sofr_today)
sofr_simulation_applied_today = pd.DataFrame(sofr_today + sofr_simulation)

sofr_simulation_applied_today_discount_factor = pd.DataFrame()

# Iterate over the columns in the DataFrame
for i, column in enumerate(sofr_simulation_applied_today.columns, start=1):
    # Calculate the discount factor for the current time period
    discount_factors = np.exp(-sofr_simulation_applied_today[column].astype(float) * i)
    
    # Add the calculated discount factors as a new column to the discount factors DataFrame
    sofr_simulation_applied_today_discount_factor[column] = discount_factors

sofr_simulation_applied_today_discount_factor = np.array( sofr_simulation_applied_today_discount_factor )
print("sofr_simulation_applied_today_discount_factor: ", sofr_simulation_applied_today_discount_factor.shape)

monte_carlo_full_reval_payer = []
# compute payer swap value for each simulation
for simulation_discount_factor in sofr_simulation_applied_today_discount_factor:
    monte_carlo_full_reval_payer_simulation = PV_payer_swap0(simulation_discount_factor, fixed_leg_ir, swap_notional)
    monte_carlo_full_reval_payer.append( monte_carlo_full_reval_payer_simulation )

monte_carlo_full_reval_payer = np.array(monte_carlo_full_reval_payer)

def calculate_var(swap_value_changes, confidence_level):
    """
    Calculate the Value at Risk (VaR) at a specified confidence level.
    
    Parameters:
    - swap_value_changes: Array of swap value changes from simulations.
    - confidence_level: Confidence level for VaR calculation.
    
    Returns:
    - The calculated VaR value.
    """
    VaR = np.percentile(swap_value_changes, 100 - confidence_level)
    return VaR

# calculate change in swap value
monte_carlo_full_reval_payer = monte_carlo_full_reval_payer - swap_value_today

monte_carlo_full_reval_payer_VaR = calculate_var(monte_carlo_full_reval_payer, confidence_level)

print("1-day 95% Monte Carlo VaR (Full Revaluation): ", monte_carlo_full_reval_payer_VaR)

# compute payer swap value for each risk factor using PV01
monte_carlo_risk_based_payer = (pv01 * sofr_simulation).sum(axis = 1)

# 95% confidence level for VaR
monte_carlo_risk_based_payer_VaR = calculate_var(monte_carlo_risk_based_payer, confidence_level)

print("1-day 95% Monte Carlo VaR (Risk-Based): ", monte_carlo_risk_based_payer_VaR)

# # Number of simulations
# n_simulations = 10000

# # Simulate daily returns for the stocks portion of portfolio using Monte Carlo simulation
# # Assuming normally distributed returns based on historical mean and std deviation
# print("returns_mean: ", returns_mean, "returns_cov: ", returns_cov)
# simulated_stock_returns = np.random.multivariate_normal(returns_mean, returns_cov, n_simulations)
# print("simulated_stock_returns: ", simulated_stock_returns)
# # Assuming the swap's value changes are represented by `swap_value_change`, include them in the simulations
# # For simplicity, let's assume a constant or randomly varied swap change. In practice, you'd model this based on interest rate changes.
# swap_value_change_simulated = np.random.normal(parametric_swap_mean, parametric_swap_std, n_simulations)
# print("swap_value_change_simulated: ", swap_value_change_simulated)
# # Calculate the simulated portfolio value changes, assuming equal weights for simplicity
# portfolio_change_simulated = np.sum(simulated_stock_returns, axis=1) + swap_value_change_simulated
# print("portfolio_change_simulated: ", portfolio_change_simulated)

# VaR_95_mc_full = -np.percentile(portfolio_change_simulated, 5)
# print("1-day 95% Monte Carlo VaR (Full Revaluation):", VaR_95_mc_full)


# # Combine stock and swap returns for the historical VaR calculation
# # Here, let's assume 'portfolio_returns' includes both stock and estimated swap returns
# historical_portfolio_changes = portfolio_returns.sum(axis=1)  # Adjust this calculation as needed

# # Calculate VaR
# VaR_95_historical = -np.percentile(historical_portfolio_changes, 5)
# print("1-day 95% Historical VaR:", VaR_95_historical)

# # Calculate the 1-day 95% VaR for the portfolio using historical returns
# historical_var_95_full_reval = -portfolio_daily_returns.quantile(0.05)

# print("historical_var_95_full_reval: ", historical_var_95_full_reval)

# # Simplified annual fixed payment calculation
# fixed_rate = 0.042  # Fixed leg interest rate
# notional = 100000000  # Notional amount
# annual_fixed_payment = fixed_rate * notional

# # Assuming a flat discount rate for simplification
# discount_rate = 0.04
# present_value_fixed = annual_fixed_payment / (1 + discount_rate)

# # For simplicity, assume the floating payment equals the fixed payment
# # In practice, this would be based on the actual SOFR rate projections
# present_value_floating = present_value_fixed  # Simplification for illustration

# # Calculate the swap's net present value (simplified)
# swap_value = present_value_floating - present_value_fixed

# # Assuming the portfolio is equally weighted across the stocks and the swap
# # For simplicity, let's assume the stocks' combined value equals the swap's notional amount
# stocks_value = notional  # Simplified assumption
# portfolio_value = stocks_value + swap_value

# # Calculate portfolio standard deviation (simplified)
# # In practice, this would involve calculating the standard deviation of returns, including the swap's sensitivity to SOFR rates
# portfolio_std_dev = np.sqrt((aapl_returns.std().mean() ** 2) + (sofr_1y_rates.std() ** 2))

# # Calculate portfolio VaR at 95% confidence level
# z_score = norm.ppf(0.95)
# portfolio_var = -z_score * portfolio_std_dev * portfolio_value

# print(f"Estimated Portfolio VaR at 95% Confidence Level: ", portfolio_value)




