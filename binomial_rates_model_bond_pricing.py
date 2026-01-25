'''
This module contains a number of different functions to price various
bonds and bond derivatives. All functions utilize a rates binomial tree
model with up factor u, down factor d, and RN-probabilities qu and qd, resp.

NOTE: All rates are assumed to be in decimal form.
'''

import numpy as np

'''
Builds binomial lattice of interest rates with initial rate r,
up factor u, down factor d, and t time periods
rt[i, j] = 0 for i > j
'''

def rates_tree(r, u, d, t):
    rt = np.zeros((t + 1, t + 1))
    rt[0,:] = r*(u**(np.linspace(0, t, t+1)))
    for i in range(1, t+1):
        rt[i, i:] = d*rt[i-1, i-1:t]
    return rt

'''
Returns elementary price lattice with initial rate r, up factor u, down factor d,
t time periods, and qu/qd risk-neutral probabilities
'''
def elementary_price_tree(r, u, d, t, qu, qd):
    rt = rates_tree(r,u,d,t-1)
    ept = np.zeros((t + 1, t + 1))
    ept[0,0] = 1
    for i in range(1, t+1):
        ept[0, i] = qu*(ept[0,i-1]/(1+rt[0,i-1]))
        ept[i,i] = qd*(ept[i-1,i-1]/(1+rt[i-1,i-1]))
        
    for i in range(1, t+1):
        ept[1:i, i] = qu*(1/(1+rt[0:i-1,i-1]))*ept[0:i-1,i-1]+qd*(1/(1+rt[1:i,i-1]))*ept[1:i,i-1]
        
    return ept


'''
Returns price of t-period zero coupon bond using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs
'''
def zcb_price(face_value, t, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    zcb_lattice = np.zeros((t+1,t+1))
    zcb_lattice[:,-1] = face_value
    for i in range(t-1, -1, -1):
        zcb_lattice[:i+1,i] = (1/(1+rt[:i+1,i]))*(qu*zcb_lattice[:i+1, i+1] + qd*zcb_lattice[1:i+2, i+1])
        
    return zcb_lattice

'''
Returns price of t-period bond using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs,
coupon rate c
'''
def cb_price(face_value, t, c, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    cb_lattice = np.zeros((t+1,t+1))
    cb_lattice[:,-1] = (1+c)*face_value
    for i in range(t-1, -1, -1):
        cb_lattice[:i+1,i] = face_value*c + (1/(1+rt[:i+1,i]))*(qu*cb_lattice[:i+1, i+1] + qd*cb_lattice[1:i+2, i+1])
        
    return cb_lattice[0,0]


'''
Returns forward price on t-period bond using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs,
coupon rate c, and forward exercise date in ft periods
'''
def cb_forward_price(face_value, ft, t, c, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    cb_lattice = np.zeros((t+1,t+1))
    cb_lattice[:,-1] = (1+c)*face_value
    for i in range(t-1, ft, -1):
        cb_lattice[:i+1,i] = face_value*c + (1/(1+rt[:i+1,i]))*(qu*cb_lattice[:i+1, i+1] + qd*cb_lattice[1:i+2, i+1])
        
    for i in range(ft, -1, -1):
        cb_lattice[:i+1,i] = (1/(1+rt[:i+1,i]))*(qu*cb_lattice[:i+1, i+1] + qd*cb_lattice[1:i+2, i+1])
        
    return cb_lattice[0,0]/zcb_price(1, ft, r, u, d, qu, qd)

'''
Returns futures on t-period bond using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs,
coupon rate c, and forward exercise date in ft periods
'''
def cb_futures_price(face_value, ft, t, c, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    cb_lattice = np.zeros((t+1,t+1))
    cb_lattice[:,-1] = (1+c)*face_value
    for i in range(t-1, ft, -1):
        cb_lattice[:i+1,i] = face_value*c + (1/(1+rt[:i+1,i]))*(qu*cb_lattice[:i+1, i+1] + qd*cb_lattice[1:i+2, i+1])
    
    cb_lattice[:ft+1,ft] = (1/(1+rt[:ft+1,ft]))*(qu*cb_lattice[:ft+1, ft+1] + qd*cb_lattice[1:ft+2, ft+1])

    for i in range(ft-1, -1, -1):
        cb_lattice[:i+1,i] = qu*cb_lattice[:i+1, i+1] + qd*cb_lattice[1:i+2, i+1]
        
    return cb_lattice[0,0]

'''
Returns caplet price expiring in t periods using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs, and
strike rate c
'''
def caplet_price(notional_value, c, t, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    caplet_lattice = np.zeros((t,t))
    caplet_lattice[:,-1] = notional_value*np.maximum(rt[:,-1] - c*np.ones(t), 0)/(1 + rt[:,-1])
    for i in range(t-2, -1, -1):
        caplet_lattice[:i+1,i] = (1/(1 + rt[:i+1, i]))*(qu*caplet_lattice[:i+1, i+1] + qd*caplet_lattice[1:i+2, i+1])
        
    return caplet_lattice[0,0]

'''
Returns floorlet price expiring in t periods using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs, and
strike rate c
'''
def floorlet_price(notional_value, c, t, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    floorlet_lattice = np.zeros((t,t))
    floorlet_lattice[:,-1] = notional_value*np.maximum(c*np.ones(t) - rt[:,-1], 0)/(1 + rt[:,-1])
    for i in range(t-2, -1, -1):
        floorlet_lattice[:i+1,i] = (1/(1 + rt[:i+1, i]))*(qu*floorlet_lattice[:i+1, i+1] + qd*floorlet_lattice[1:i+2, i+1])
        
    return floorlet_lattice[0,0]


'''
Returns swap price expiring in t periods using a binomial lattice term structure with
initial rate r, up factor u, down factor d, qu/qd risk-neutral probs, and
strike rate c (assumes first payment is at time 1 and last payment is at time t)
'''
def swap_price(notional_value, c, t, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    swap_lattice = np.zeros((t,t))
    swap_lattice[:,-1] = (rt[:,-1] - c*np.ones(t))/(1 + rt[:,-1])
    for i in range(t-2, -1, -1):
        swap_lattice[:i+1,i] = (1/(1 + rt[:i+1, i]))*((rt[:i+1,i] - c*np.ones(i+1)) + qu*swap_lattice[:i+1, i+1] + qd*swap_lattice[1:i+2, i+1])
        
    return notional_value*swap_lattice[0,0]

'''
Returns swaption price expiring in ot periods for an underlying swap expiring in t (t > ot) periods
using a binomial lattice term structure with initial rate r, up factor u, down factor d,
qu/qd risk-neutral probs, and swap strike rate c
'''
def swaption_price(notional_value, ot, c, t, r, u, d, qu, qd):
    rt = rates_tree(r, u, d, t-1)
    swaption_lattice = np.zeros((t,t))
    swaption_lattice[:,-1] = (rt[:,-1] - c*np.ones(t))/(1 + rt[:,-1])
    for i in range(t-2, ot-1, -1):
        swaption_lattice[:i+1,i] = (1/(1 + rt[:i+1, i]))*((rt[:i+1,i] - c*np.ones(i+1)) + qu*swaption_lattice[:i+1, i+1] + qd*swaption_lattice[1:i+2, i+1])
    
    swaption_lattice[:ot + 1, ot] = np.maximum(swaption_lattice[:ot + 1, ot], 0)
    for i in range(ot-1, -1, -1):
        swaption_lattice[:i+1,i] = (1/(1 + rt[:i+1, i]))*(qu*swaption_lattice[:i+1, i+1] + qd*swaption_lattice[1:i+2, i+1])
    
    return notional_value*swaption_lattice[0,0]





