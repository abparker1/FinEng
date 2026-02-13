'''
This module performs historical mean-variance analysis of a given collection
of stocks over a set historical period with data pulled from Yahoo Finance using
the yfinance Python library.

The program is able to determine an 'optimal' weighting of the stocks 
when given an upper bound in volatility or a lower bound on returns. Alternatively,
it might conclude that no portfolio weighting is able to meet the desired constraints.

Here, optimal means one of two things:
    1) minimizing volatility while still reaching the lower bound on returns
    2) maximizing returns while staying below the upper bound on volatility
    
The variables risk_allowance and desired_returns are assumed to be percentages
in decimal form (i.e. 50% should be entered as 0.5)
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

#pulls relevant data from yfinance, builds covariance matrix and mean returns over given period
def build_stock_data(tickers, interval, period):
    stock_data = yf.download(tickers, interval = interval, period = period)['Close'].pct_change()
    cov_matrix = stock_data.cov()
    mean_returns = stock_data.mean()
    
    return (np.array(cov_matrix), np.array(mean_returns))

#objective function for maximizing expected returns using a weight vector x
#(This function multiplies by -1 because we will be using scipy.optimize.minimize)
def return_objective(x, mean_returns):
    return -1*np.dot(x, mean_returns)

#objective function for minimizing historical volatility using weight vector x
def vol_objective(x, cov_matrix):
    return np.sqrt(x @ cov_matrix @ np.reshape(x, (-1, 1)))


#maximize returns at (at_most == False) or below (at_most == True) a fixed level
#assumes short positions are allowed unless short_allowed == False
def max_returns_sol(tickers, interval, period, risk_allowance, at_most = True, short_allowed = True):
    (cov_matrix, mean_returns) = build_stock_data(tickers, interval, period)
    x = np.ones(len(mean_returns))/len(mean_returns)
    
    cons = ()
    if at_most:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1},
                {'type': 'ineq', 'fun': lambda x: risk_allowance - np.sqrt((x @ cov_matrix @ np.reshape(x, (-1, 1))))})
    
    else:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1},
                {'type': 'eq', 'fun': lambda x: risk_allowance - np.sqrt((x @ cov_matrix @ np.reshape(x, (-1, 1))))})
    
    bnds = None
    if not short_allowed:
        bnds = ((0, None) for i in range(len(mean_returns)))
        
    res = minimize(return_objective, x, args = (mean_returns), bounds = bnds, constraints = cons)
    
    print('\n\n\n')
    if res.success:
        weights = res.x
        percent_returns = -100*res.fun
        
        print("Asset Weights\n"+'-'*25)
        for i in range(len(tickers)):
            print(f'Stock: {tickers[i]}, Weight: {weights[i]}')
            
            
        print(2*('-'*25 + '\n'))
        print(f'Expected Return: {percent_returns:.3f}%')
        print(f'Volatility: {vol_objective(weights, cov_matrix)[0]*100:.3f}%')
    
    else:
        print('No portfolio exists with given constraints')
    
#minimize risk (historical volatility) while maintaining returns at (at_least == False) or above (at_least == True) a fixed level
#assumes short positions are allowed unless short_allowed == False
def min_risk_sol(tickers, interval, period, desired_returns, at_least = True, short_allowed = True):
    
    (cov_matrix, mean_returns) = build_stock_data(tickers, interval, period)
    x = np.ones(len(mean_returns))/len(mean_returns)
    cons = ()
    if at_least:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1},
                {'type': 'ineq', 'fun': lambda x: np.dot(x, mean_returns) - desired_returns})
    
    else:
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - desired_returns})
    
    bnds = None
    if not short_allowed:
        bnds = ((0, None) for i in range(len(mean_returns)))
        
    res = minimize(vol_objective, x, args = (cov_matrix), bounds = bnds, constraints = cons)
    
    print('\n\n\n')
    if res.success:
        weights = res.x
        vol = 100*res.fun
        
        print("Asset Weights\n"+'-'*25)
        for i in range(len(tickers)):
            print(f'Stock: {tickers[i]}, Weight: {weights[i]}')
            
        print(2*('-'*25 + '\n'))
        print(f'Expected Return: {-100*return_objective(weights, mean_returns):.3f}%')
        print(f'Volatility: {vol:.3f}%')
        
    else:
        print('No portfolio exists with the given constraints')


#------------------------------------------------
#------------------------------------------------
#------------------------------------------------

tickers = ['AAPL', 'GOOG', 'AMZN', 'TSLA', 'NVDA']

#interval options
#------------------------
#1d, 5d, 1wk, 1mo, 3mo
interval = '3mo'


#period options
#------------------------
#1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max, (start='YYYY-MM-DD', end='YYYY-MM-DD')
period = '5y'

short_allowed = False

at_most = True
risk_allowance = 0.15
max_returns_sol(tickers, interval, period, risk_allowance, at_most = at_most, short_allowed = short_allowed)

# at_least = True
# desired_returns = 0.08
# min_risk_sol(tickers, interval, period, desired_returns, at_least = at_least, short_allowed = short_allowed)



