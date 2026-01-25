'''
This module reads in the available treasury yield rates (1,2,3,5,7,10,20,30 yrs)
via fredapi and uses linear interpolation for remaining years.
This is the assumed spot market rates.

We assume a rates tree follows the BDT model (r_{i,j} = a_i*exp(b*j)).
b will be assumed to be fixed and the a_i's are calibrated using the market rates
to ensure that the yield of a zcb is aligned with the market rates.
We use the calibrated parameters to compute coupon bond prices for maturities
between 1 and t years (t <= 30).

NOTE: All rates are assumed to be in decimal form.
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fredapi import Fred

apikey = 'INSERT KEY HERE'
fred = Fred(api_key=apikey)

def load_market_rates(t):
    fred_yield_yrs = [1, 2, 3, 5, 7, 10, 20, 30]
    fred_yield_yrs_upto_t = [x for x in fred_yield_yrs if x <= t]
    fred_yields = [fred.get_series('DGS' + str(x)).iloc[-1] for x in fred_yield_yrs if x <= t]
    linear_int = [x for x in range(1, 31) if x not in fred_yields and x <= t]
    yields = np.interp(linear_int, fred_yield_yrs_upto_t, fred_yields)
    
    return yields
'''
Builds binomial lattice of interest rates with initial rate r,
up factor u, down factor d, and t time periods
rt[i, j] = 0 for i > j
'''
'''
Returns t-period rates tree using the BDT model (r_{i,j} = a + exp(b*j) 
for fixed a and b and R-N probs q_{i,j} = q = 1 - q = 1/2
'''
def rates_tree(a, b, t):
    rt = np.zeros((t + 1, t + 1))
    for i in range(0, t+1):
        rt[:i+1, i] = a[i]*np.exp(b*(np.ones(i+1).cumsum() - 1)[::-1])
    return rt/100

'''
Returns t-period elementary price lattice using the BDT model (r_{i,j} = a + exp(b*j) 
for fixed a and b and R-N probs q_{i,j} = q = 1 - q = 1/2
'''
def elementary_price_tree(a, b, t, qu):
    qd = 1 - qu
    rt = rates_tree(a,b,t-1)
    
    ept = np.zeros((t + 1, t + 1))
    ept[0,0] = 1
    for i in range(1, t+1):
        ept[0, i] = qu*(ept[0,i-1]/(1+rt[0,i-1]))
        ept[i,i] = qd*(ept[i-1,i-1]/(1+rt[i-1,i-1]))
        
    for i in range(1, t+1):
        ept[1:i, i] = qu*(1/(1+rt[0:i-1,i-1]))*ept[0:i-1,i-1]+qd*(1/(1+rt[1:i,i-1]))*ept[1:i,i-1]
        
    return ept

'''
Returns ZCB prices for all periods from 1 to t using the BDT model (r_{i,j} = a + exp(b*j) 
for fixed a and b and R-N probs q_{i,j} = q = 1 - q = 1/2
'''
def ZCB_prices(a,b,t,qu):
    ept = elementary_price_tree(a, b, t, qu)
    return np.sum(ept, axis = 0)[1:]


'''
Returns spot rates for all periods from 1 to t using the BDT model (r_{i,j} = a + exp(b*j) 
for fixed a and b and R-N probs q_{i,j} = q = 1 - q = 1/2
'''
def BDT_spot_rates(a,b,t,qu):
    zcb = ZCB_prices(a,b,t,qu)
    return np.power(1/zcb, 1/(np.ones(t).cumsum()))-1

'''
Returns spot rates for all periods from 1 to t using the BDT model (r_{i,j} = a + exp(b*j) 
for fixed a and b and R-N probs q_{i,j} = q = 1 - q = 1/2
'''
def objective(a, b, t, qu, mr):
    return np.sum(np.power(100*(mr - BDT_spot_rates(a,b,t,qu)), 2))

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

'''
Print ZCB prices and calibrated spot rates
'''
b = 0.005
t = 30
qu = 0.5
face_value = 100

mr = load_market_rates(t)/100
a = 5*np.ones(t)
res = minimize(objective, a, args=(b, t, qu, mr))
optimal_a = res.x

zcb = face_value*ZCB_prices(optimal_a, b, t, qu)
print(f'Zero Coupon Bond Prices with face value ${face_value}')
print('----------------------------------------------------')
      
for i in range(t):
    print(f'{i+1}-year: ${zcb[i]:.2f}')
    