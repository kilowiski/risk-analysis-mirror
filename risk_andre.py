import pandas as pd
from scipy.stats import qmc, norm
import numpy as np


#############
# CONSTANTS
#############
SWAP_NOTIONAL = 100000000 # 100 million
FIXED_LEG_IR = 0.042 # 4.2%
TODAY_DATE = '2023-10-30'  # Starting date for calculation
CONFIDENCE_LEVEL = 95

#############
# UTIL FUNC
#############

def PV_payer_swap(discount_rate_today, fixed_leg_ir, swap_notional):
    """
    Calculate the present value (PV) of a payer interest rate swap.

    The function computes the net present value of the swap by considering the
    difference between the floating leg (based on the last discount rate) and
    the fixed leg (based on the fixed interest rate and the sum of today's discount rates).

    Parameters:
    - discount_rate_today (np.ndarray): An array of discount rates applicable for today,
                                        typically derived from SOFR rates adjusted for each tenor.
    - fixed_leg_ir (float): The fixed interest rate agreed upon in the swap contract.
    - swap_notional (float): The notional amount of the swap.

    Returns:
    - float: The net present value (PV) of the payer swap, which is the difference between
             the floating leg and fixed leg values, scaled by the notional amount.
    """
    # Calculate the value of the floating leg using the last discount rate
    float_leg = 1 - discount_rate_today[-1]

    # Calculate the value of the fixed leg as the product of the fixed interest rate
    # and the sum of today's discount rates
    fix_leg = fixed_leg_ir * sum(discount_rate_today)

    # Compute the net present value of the swap
    payer_swap_PV = swap_notional * (float_leg - fix_leg)

    return payer_swap_PV


def PV_payer_swap0(sofr_rates, discount_rates, fixed_leg_ir, swap_notional, start_date):
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
    return abs(VaR)

#############
# DATA PREP
#############

def data_prep():
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

    # compute daily returns for the stocks
    aapl_returns = aapl_data['Adj Close']
    msft_returns = msft_data['Adj Close']
    f_returns    = f_data['Adj Close']
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

    # Convert the index of the sofr_rates DataFrame to datetime
    sofr_rates.index = pd.to_datetime(sofr_rates.index)
    discount_rates.index = pd.to_datetime(discount_rates.index)

    # track daily change in sofr rates
    sofr_rates_change = sofr_rates.diff().dropna()

    # compute mean, std, cov matrix in sofr rates
    sofr_mean = sofr_rates_change.mean(axis = 0).to_numpy()
    sofr_std  = sofr_rates_change.std(axis = 0).to_numpy()
    sofr_cov  = (sofr_rates_change).cov().to_numpy()

    # compute current value of swap
    discount_rate_today = discount_rates.loc[TODAY_DATE].to_numpy()

    # swap_value_today = PV_payer_swap(sofr_rates, discount_rates, fixed_leg_ir, swap_notional, start_date)
    swap_value_today   = PV_payer_swap(discount_rate_today, FIXED_LEG_IR, SWAP_NOTIONAL)

    stock_prices       = portfolio.drop(columns=['SOFR'])
    stock_prices.index = pd.to_datetime(stock_prices.index)
    stock_prices_today = stock_prices.loc[TODAY_DATE]


    # number of shares held today per stock with 1M USD
    stock_values_today    = np.array([1000000] * 4)
    num_shares_held_today = stock_values_today / stock_prices_today
    portfolio_value_today = swap_value_today + sum(stock_values_today)

    print(f'Stock Value Today (30 Oct 2023): ${round(sum(stock_values_today), 2):,}')
    print(f'Swap Value Today (30 Oct 2023): ${round(swap_value_today, 2):,}')
    print(f'Portfolio Value Today (30 Oct 2023): ${round(portfolio_value_today, 2):,}')


    stock_prices_array = stock_prices.to_numpy()

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
    returns_std  = returns_array.std(axis = 0)
    returns_cov  = np.cov(returns_array, rowvar=False)

    basis_point_change = 0.0001

    swap_value_change = []

    # filter for sofr rates on today
    sofr_today = sofr_rates.loc[TODAY_DATE].to_numpy()

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
        swap_value_change_tenor = PV_payer_swap(discount_factor_adj_sofr_rates, FIXED_LEG_IR, SWAP_NOTIONAL) - swap_value_today
        swap_value_change.append( swap_value_change_tenor )

    # compute pv01 from bp change
    swap_value_change = np.array(swap_value_change)
    pv01 = swap_value_change / basis_point_change

    return discount_rates, sofr_rates_change, returns_array, returns_mean, returns_std, returns_cov, sofr_mean, sofr_std, sofr_cov, stock_values_today, sofr_today, swap_value_today, pv01

discount_rates, sofr_rates_change, returns_array, returns_mean, returns_std, returns_cov, sofr_mean, sofr_std, sofr_cov, stock_values_today, sofr_today, swap_value_today, pv01 = data_prep()

#############
# PARAMETRIC
#############

def compute_parametric_portfolio_statistics(asset_means, covariance_matrix, asset_values):
    """
    Calculate parametric statistics of a portfolio, including the mean and standard deviation, based on asset means, the covariance matrix, and the portfolio weights.
    This method assumes a normal distribution of returns and uses these parameters to model portfolio behavior and risk.

    Parameters:
    - asset_means (np.array): An array of mean returns for each asset in the portfolio, representing expected performance.
    - covariance_matrix (np.array): A 2D array representing the covariance between the portfolio assets, indicating how asset returns move relative to each other.
    - asset_values (np.array): An array representing the weights or notional values of each asset in the portfolio, used to calculate weighted returns and risk.

    Returns:
    - parametric_mean (float): The weighted average mean return of the portfolio, indicating expected portfolio performance.
    - parametric_std (float): The standard deviation of the portfolio's return, representing the portfolio's total risk in a parametric model.
    """
    # Step 1: Calculate the weighted mean return of the portfolio
    parametric_mean = np.dot(asset_means, asset_values)

    # Step 2: Calculate the portfolio variance using the covariance matrix and asset values
    weighted_covariances = np.dot(covariance_matrix, asset_values)
    parametric_variance = np.dot(asset_values, weighted_covariances)

    # Step 3: Calculate the portfolio standard deviation from the variance
    parametric_std = np.sqrt(parametric_variance)

    return parametric_mean, parametric_std

def parametric_model_var_95(sofr_rates_change, returns_mean, returns_cov, stock_values_today, sofr_mean, sofr_cov, pv01):

    parametric_stock_mean, parametric_stock_std = compute_parametric_portfolio_statistics(returns_mean, returns_cov, stock_values_today)

    print("parametric stock mean: ", parametric_stock_mean)
    print("parametric_stock_std: ", parametric_stock_std )

    parametric_swap_mean, parametric_swap_std = compute_parametric_portfolio_statistics(sofr_mean, sofr_cov, pv01)

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

    parametric_portfolio_mean = parametric_swap_mean + parametric_stock_mean
    parametric_portfolio_std = np.sqrt(parametric_swap_std ** 2 + parametric_stock_std ** 2)

    parametric_var_95_portfolio = -(parametric_portfolio_mean - z_score_95 * parametric_portfolio_std)

    print("Parametric VaR 95: ", parametric_var_95_portfolio)

    return parametric_var_95_swap, parametric_var_95_stock, parametric_var_95_portfolio

parametric_var_95_swap, parametric_var_95_stock, parametric_var_95_portfolio = parametric_model_var_95(sofr_rates_change, returns_mean, returns_cov, stock_values_today, sofr_mean, sofr_cov, pv01)

#############
# MONTE-CARLO
#############

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

def monte_carlo_model_var_95(sofr_rates_change, returns_array, returns_mean, returns_std, returns_cov, sofr_mean, sofr_std, sofr_cov, stock_values_today, sofr_today, swap_value_today):
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
        monte_carlo_full_reval_payer_simulation = PV_payer_swap(simulation_discount_factor, FIXED_LEG_IR, SWAP_NOTIONAL)
        monte_carlo_full_reval_payer.append( monte_carlo_full_reval_payer_simulation )

    monte_carlo_full_reval_payer = np.array(monte_carlo_full_reval_payer)

    # calculate change in swap value
    monte_carlo_full_reval_payer = monte_carlo_full_reval_payer - swap_value_today

    monte_carlo_full_reval_payer_var_95 = calculate_var(monte_carlo_full_reval_payer, CONFIDENCE_LEVEL)

    print("Monte Carlo VaR 95 (Full Revaluation): ", monte_carlo_full_reval_payer_var_95)

    # compute payer swap value for each risk factor using PV01
    monte_carlo_risk_based_payer = (pv01 * sofr_simulation).sum(axis = 1)

    # 95% confidence level for VaR
    monte_carlo_risk_based_payer_var_95 = calculate_var(monte_carlo_risk_based_payer, CONFIDENCE_LEVEL)

    print("Monte Carlo VaR 95 (Risk-Based): ", monte_carlo_risk_based_payer_var_95)

    return monte_carlo_full_reval_payer_var_95, monte_carlo_risk_based_payer_var_95

monte_carlo_full_reval_payer_var_95, monte_carlo_risk_based_payer_var_95 = monte_carlo_model_var_95(sofr_rates_change, returns_array, returns_mean, returns_std, returns_cov, sofr_mean, sofr_std, sofr_cov, stock_values_today, sofr_today, swap_value_today)

#############
# HISTORICAL
#############
def historical_model_var_95(discount_rates, sofr_rates_change, swap_value_today):
    # we already have previously computed historical discount rates at discount_rates variable
    sofr_historical_discount_factor = discount_rates.to_numpy()
    print("check123: ", sofr_historical_discount_factor)
    historical_full_reval_payer = []

    # compute payer swap value for each historical data
    for historical_discount_factor in sofr_historical_discount_factor:
        historical_full_reval_payer_data = PV_payer_swap(historical_discount_factor, FIXED_LEG_IR, SWAP_NOTIONAL)
        historical_full_reval_payer.append( historical_full_reval_payer_data )

    # calculate change in value of swap
    historical_full_reval_payer = historical_full_reval_payer - swap_value_today
    historical_full_reval_payer_95_var = calculate_var(historical_full_reval_payer, CONFIDENCE_LEVEL)

    # we combine interest rate sensitivity (PV01) with historical interest rate movements 
    # to estimate how the swap's value could change in response to rate fluctuations
    historical_swap_value_interest_rate_sensitivity = (pv01 * sofr_rates_change.to_numpy())
    historical_risk_based_payer = historical_swap_value_interest_rate_sensitivity.sum(axis = 1)

    historical_risk_based_payer_var_95 = calculate_var(historical_risk_based_payer, CONFIDENCE_LEVEL)

    return historical_full_reval_payer_95_var, historical_risk_based_payer_var_95

historical_full_reval_payer_95_var, historical_risk_based_payer_var_95 = historical_model_var_95(discount_rates, sofr_rates_change, swap_value_today)

print("Historical Var 95 (Full Revaluation): ", historical_full_reval_payer_95_var)
print("Historical Var 95 (Risk-Based): ", historical_risk_based_payer_var_95)


print("*** ANDRE ANSWERS ***")
print("Parametric VaR 95: ", parametric_var_95_portfolio)
print("Monte Carlo VaR 95 (Full Revaluation): ", monte_carlo_full_reval_payer_var_95)
print("Monte Carlo VaR 95 (Risk-Based): ", monte_carlo_risk_based_payer_var_95)
print("Historical VaR 95 (Full Revaluation): ", historical_full_reval_payer_95_var)
print("Historical VaR 95 (Risk-Based): ", historical_risk_based_payer_var_95)





