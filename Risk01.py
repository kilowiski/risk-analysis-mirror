# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:08:33 2024

@author: Hyperion
"""

import numpy as np
import pandas as pd
from scipy import stats

file_loc = r'C:\Users\zk_me\Downloads\hist_data.xlsm'

## Load data
sofr_df = pd.read_excel(file_loc, sheet_name = 'SofrCurve', index_col = 'Tenor').T.iloc[1:,:]
sofr_df = sofr_df[['1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y']]
sofr_df = sofr_df * 10000

sofr_diff = sofr_df.diff().dropna()

combined_stock_df = pd.concat([pd.read_excel(file_loc, sheet_name = 'AAPL', index_col = 'Date').pct_change().dropna(), 
                               pd.read_excel(file_loc, sheet_name = 'MSFT', index_col = 'Date').pct_change().dropna(),
                               pd.read_excel(file_loc, sheet_name = 'F', index_col = 'Date').pct_change().dropna(), 
                               pd.read_excel(file_loc, sheet_name = 'BAC', index_col = 'Date').pct_change().dropna()],axis = 1)

combined_stock_df.columns = ['AAPL','MSFT','F','BAC']


# Parametric VaR
# SOFR


def pv_payer_swap(rate_curve_df, notional, fixed_rate, tenor):
    
    # rate_curve is series of relevant rates
    # /10000 because in bps
    sofr_list = [rate/10000 for rate in list(rate_curve_df)]
    
    pv_fixed = notional * np.exp(-1 * tenor * sofr_list[-1])
    
    for i in range(tenor):
        discount_factor = np.exp(-1 * (i+1) * sofr_list[i])
        pv_fixed += fixed_rate * notional * discount_factor

    return notional - pv_fixed

def calc_pv01(rate_curve_df, notional, fixed_rate, tenor):
    
    p0 = pv_payer_swap(rate_curve_df, notional, fixed_rate, tenor)
    tmp_array = np.array([])
    
    for i in range(tenor):
        # shock by 1 bp
        sofr_list = rate_curve_df.copy()
        sofr_list[i] = sofr_list[i] + 1

        #calc pv01
        partial_pv01 = pv_payer_swap(sofr_list, notional, fixed_rate, tenor) - p0
        tmp_array = np.append(tmp_array, partial_pv01)
    
    return tmp_array

risk_sensitivity = np.append(calc_pv01(sofr_df.iloc[-1, :], 100000000, 0.042, 10),
                                np.array([1000000, 1000000, 1000000, 1000000]))

mean_change = np.array(pd.concat([sofr_diff.mean(), 
                                  combined_stock_df.mean()]))

cov_matrix = np.cov(np.array(pd.concat([sofr_diff.reset_index().iloc[:,1:], 
                                        combined_stock_df.reset_index().iloc[:,1:]],
                                axis = 1, ignore_index=True)).T)

var_portf_pl = np.matmul(np.matmul(risk_sensitivity.reshape(1, 14), cov_matrix),
                         risk_sensitivity)

mean_portf_pl = (risk_sensitivity * mean_change).sum()

parametric_var = abs(mean_portf_pl + (-1.6448536269514729)*np.sqrt(var_portf_pl))


def rng_changes(risk_factor_changes):
    
    a = np.array([])
    
    for i in range(risk_factor_changes.shape[1]):
        
        a = np.append(a,
                      stats.norm.ppf(np.random.uniform(),
                                     loc = risk_factor_changes.mean()[i],
                                     scale = risk_factor_changes.std()[i])
                      )
        
    b = np.linalg.cholesky(risk_factor_changes.corr())
    c = np.matmul(b,a)
    
    return c


risk_factor_changes = pd.concat([sofr_diff.reset_index().iloc[:,1:], 
                                 combined_stock_df.reset_index().iloc[:,1:]], 
                                axis = 1, ignore_index=True)

monte_carlo_risk_based_var = np.array([])

for i in range(10000):
    a = rng_changes(risk_factor_changes)
    mean_portf_pl = (risk_sensitivity * a).sum()
    
    monte_carlo_risk_based_var = np.append(monte_carlo_risk_based_var, mean_portf_pl)
    

monte_carlo_risk_based_var = np.sort(monte_carlo_risk_based_var)[int(monte_carlo_risk_based_var.shape[0]*0.05)]

monte_carlo_full_rep_var = np.array([])

for i in range(100000):
    a = rng_changes(risk_factor_changes)
    mean_portf_pl = (risk_sensitivity * (mean_change+a)).sum()
    
    monte_carlo_full_rep_var = np.append(monte_carlo_full_rep_var, mean_portf_pl)
    
monte_carlo_full_rep_var = np.sort(monte_carlo_full_rep_var)[int(monte_carlo_full_rep_var.shape[0]*0.05)]

