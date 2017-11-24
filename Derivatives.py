
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from scipy.stats import norm
# Project 4
# ctrl shift i for help
# ctrl space for code completion
# Ctrl+Alt+I indentation

###################BINOMIAL TREE#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# Binomial Tree using for loops PUT and Call
def binomial(strike,Cp):
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    M = 5  # number of Time steps
    dt = T / M
    df = np.exp(-r * dt)  # discount factor per time interval
    # binomial parameters

    u = np.exp(sigma * np.sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (np.exp(r * dt) - d) / (u - d)  # martingale probability

    #   Index level simulation Simulate step by step the index levels. Loop 1
    S = np.zeros((M + 1, M + 1))  # STOCK TREE
    # index level array
    S[0, 0] = S0
    z1 = 0
    for j in range(1, M + 1):
        z1 = z1 + 1
        for i in range(z1 + 1):
            S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2)) # To fill binomial tree
    # To fill inner values
    iv = np.zeros((M + 1, M + 1), dtype=np.float64)
    # inner value array OPTION PRICES
    z2 = 0
    for j in range(0, M + 1, 1):
        for i in range(z2 + 1):
            iv[i, j] = max(Cp*(S[i, j] - strike), 0)
        z2 = z2 + 1
    # PV
    pv = np.zeros((M + 1, M + 1), dtype=np.float64)
    # present value array
    pv[:, M] = iv[:, M] # initialize last time point
    z3 = M + 1
    for j in range(M - 1, -1, -1):
        z3 = z3 - 1
        for i in range(z3):
            pv[i, j] = (q * pv[i, j + 1] +
                        (1 - q) * pv[i + 1, j + 1]) * df
    return(pv[0,0])
def binomial_np(strike):
    # index levels with numpy
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    M = 5 # number of Time steps
    dt = T / M
    df = np.exp(-r * dt)  # discount factor per time interval
    # binomial parameters
    u = np.exp(sigma * np.sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (np.exp(r * dt) - d) / (u - d)  # martingale probability
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md
    # Valuation Loop
    pv = np.maximum(S - 100, 0)
    z = 0
    for t in range(M - 1, -1, -1):  # backward iteration
        pv[0:M - z, t] = (q * pv[0:M - z, t + 1]
                          + (1 - q) *pv[1:M - z + 1, t + 1]) *df
    z = z + 1
    return pv[0, 0]

######################################################################
""" BLack scholes Call"""

from scipy import log,exp,sqrt,stats
def bs_call(S,X,T,r,sigma):
    d1=(np.log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
bs_call(100,100,1,0.05,0.2)
""" BLack scholes Put"""
def bs_put(S,X,T,r,sigma):
    d1=(np.log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return X*exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
bs_put(100,100,1,0.05,0.2)
'''American/European Option Binomial Put and Call'''
# index levels with numpy
# S0 = 100;T = 1;r = 0.05;sigma = 0.2;N = 10  # number of Paths
# K =100;M = 10 # time steps
def BinomialTree_all(type, S0, K, r, sigma, T, N=5, american="false"):
    # we improve the previous tree by checking for early exercise for american options

    # calculate delta T
    deltaT = float ( T ) / N

    # up and down factor will be constant for the tree so we calculate outside the loop
    u = np.exp ( sigma * np.sqrt ( deltaT ) )
    d = 1.0 / u

    # to work with vector we need to init the arrays using numpy
    fs = np.asarray ( [0.0 for i in range ( N + 1 )] )

    # we need the stock tree for calculations of expiration values
    fs2 = np.asarray ( [(S0 * u ** j * d ** (N - j)) for j in range ( N + 1 )] )

    # we vectorize the strikes as well so the expiration check will be faster
    fs3 = np.asarray ( [float ( K ) for i in range ( N + 1 )] )

    # rates are fixed so the probability of up and down are fixed.
    # this is used to make sure the drift is the risk free rate
    a = np.exp ( r * deltaT )
    p = (a - d) / (u - d)
    q = 1.0 - p

    # Compute the leaves, f_{N, j}
    if type == "C":
        fs[:] = np.maximum ( fs2 - fs3, 0.0 )
    else:
        fs[:] = np.maximum ( -fs2 + fs3, 0.0 )

    # calculate backward the option prices
    for i in range ( N - 1, -1, -1 ):
        fs[:-1] = np.exp ( -r * deltaT ) * (p * fs[1:] + q * fs[:-1])
        fs2[:] = fs2[:] * u

        if american == 'true':
            # Simply check if the option is worth more alive or dead
            if type == "C":
                fs[:] = np.maximum ( fs[:], fs2[:] - fs3[:] )
            else:
                fs[:] = np.maximum ( fs[:], -fs2[:] + fs3[:] )

    # print fs
    return fs[0]

'''ASIAN OPTIONS'''
'''Normal MC for European Call/Put'''
def MC(S0,K,r,sigma,T,M,I,seed, type = 'C'):
    # S0 = 100;K = 100;T = 1.0;r = 0.05;sigma = 0.2;M = 100;I = 50000
    dt = T / M;
    # Simulating I paths with M time steps
    S = np.zeros ((I,M + 1))
    S[:, 0] = S0
    np.random.seed ( seed )
    for t in range ( 1, M + 1 ):
        z = np.random.standard_normal(I)  # pseudorandom numbers
        S[:, t] = S[:, t - 1] * np.exp ( (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt ( dt ) * z )
    # vectorized operation per time step over all paths
    # Calculating the Monte Carlo estimator
    if type == 'C':
        price = math.exp ( -r * T ) * np.sum ( np.maximum ( S[:, -1] - K, 0 ) ) / I
    else:
        price = math.exp ( -r * T ) * np.sum ( np.maximum (K - S[:, -1], 0 ) ) / I
    return price

MC(100,100,0.05,0.2,1,100,250000,12345, type = 'C')

'''Average Options'''

def MCsim(S0,K,r,sigma,T,M,I,seed):
    # S0 = 100;K = 100;T = 1.0;r = 0.05;sigma = 0.2;M = 100;I = 50000
    dt = T / M;
    # Simulating I paths with M time steps
    S = np.zeros ((I,M + 1))
    S[:, 0] = S0
    np.random.seed ( seed )
    for t in range ( 1, M + 1 ):
        z = np.random.standard_normal (I)  # pseudorandom numbers
        S[:, t] = S[:, t - 1] * np.exp ( (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt ( dt ) * z)
    return S

Stock_path = MCsim(100,100,0.05,0.2,1,50,10,12345)
price_avg  = np.mean(Stock_path,1)
call =  np.mean(np.maximum(price_avg - 100,0)) * np.exp(-.05*1)

''''Pricing barrier options using the Monte Carlo simulation'''
'http://www.fintools.com/resources/online-calculators/exotics-calculators/exoticscalc-barrier/ ' \
'down and in - In this barrier option, the price starts above a barrier and has'
'to reach the barrier to be activated.'

def down_and_in(S0,K,r,sigma,T,M,I,seed, barrier,type = 'C'):
    # I = 250000
    st = MCsim(40,40,0.05,0.2,0.5,50,I,12345)
    smin = np.min(st,1) # minimum across rows
    payoff = np.zeros(I)
    for i in range(I):
        if smin[i] <= barrier: # if the price is below the barrier then the option is ITM
            if type == 'C':
                payoff[i] = np.maximum(st[i,-1] - K,0)
            else: # if price is greater than the barrier the option is knocked out
                payoff[i] = np.maximum (K - st[i, -1], 0 )
        else:
            payoff[i] = 0
    price_dic = np.exp(-0.05*0.5)*np.mean(payoff)
    return price_dic

def down_and_out(S0,K,r,sigma,T,M,I,seed, barrier,type = 'C'):
    # I = 250000
    st = MCsim(S0,K,r,sigma,T,M,I,seed)
    smin = np.min(st,1) # minimum across rows
    payoff = np.zeros(I)
    for i in range(I):
        if smin[i] > barrier:
            if type == 'C':
                payoff[i] = np.maximum(st[i,-1] - K,0)
            else:
                payoff[i] = np.maximum(K - st[i,-1],0)
        else:
            payoff[i] = 0
    price_doc = np.exp(-0.05*0.5)*np.mean(payoff)
    return  price_doc

bs_call(40,40,0.5,0.05,0.2) - (down_and_in(40,40,0.05,0.2,0.5,50,100000,12345, 35,'C') + down_and_out(40,40,0.05,0.2,0.5,50,100000,12345,35,'C'))
bs_put(40,40,0.5,0.05,0.2) - (down_and_in(40,40,0.05,0.2,0.5,50,100000,12345, 35,'P') + down_and_out(40,40,0.05,0.2,0.5,50,100000,12345,35,'P'))

def up_and_in(S0,K,r,sigma,T,M,I,seed, barrier,type = 'C'):
    # I = 250000
    # barrier greater than S0
    st = MCsim(40,40,0.05,0.2,0.5,50,I,12345)
    smax = np.max(st,1) # minimum across rows
    payoff = np.zeros(I)
    for i in range(I):
        if smax[i] >= barrier:
            if type == 'C':
                payoff[i] = np.maximum(st[i,-1] - K,0)
            else:
                payoff[i] = np.maximum (K - st[i, -1], 0 )
        else:
            payoff[i] = 0
    price_upin = np.exp(-0.05*0.5)*np.mean(payoff)
    return price_upin

def up_and_out(S0,K,r,sigma,T,M,I,seed, barrier,type = 'C'):
    I = 10
    # barrier greater than S0
    st = MCsim(40,40,0.05,0.2,0.5,50,I,12345)
    smax = np.max(st,1) # maximum across rows
    payoff = np.zeros(I)
    for i in range(I):
        if smax[i] < 42:
            if type == 'C':
                payoff[i] = np.maximum(st[i,-1] - K,0)
            else:
                payoff[i] = np.maximum(K - st[i,-1],0)
        else:
            payoff[i] = 0
    price_upo = np.exp(-0.05*0.5)*np.mean(payoff)
    return  price_upo

bs_call(40,40,0.5,0.05,0.2) - (up_and_in(40,40,0.05,0.2,0.5,50,100000,12345, 42,'C') + up_and_out(40,40,0.05,0.2,0.5,50,100000,12345,42,'C'))
bs_put(40,40,0.5,0.05,0.2) - (up_and_in(40,40,0.05,0.2,0.5,50,100000,12345, 42,'P') + up_and_out(40,40,0.05,0.2,0.5,50,100000,12345,42,'P'))

plt.plot(np.transpose(st))
plt.axhline(y=50, color='r', linestyle='-')

def lookback(S0,K,r,sigma,T,M,I,seed, type = 'C'):
    np.random.seed(seed)
    st = MCsim (S0,K,r,sigma,T,M,I,seed)
    L_m = np.zeros(I) # checking for maximum and minimum for call and put
    L_option_price = np.zeros(I)
    for i in range(I):
        if type == 'C':
            L_m[i] = np.max(st[i,:])#stock prices
            L_option_price[i] = np.exp(-r*T)* np.maximum(L_m[i] - K,0)
        else:
            L_m[i] = np.min ( st[i, :] )  # stock prices
            L_option_price[i] = np.exp ( -r * T ) * np.maximum (K - L_m[i], 0 )
    Option_Price = np.mean(L_option_price)
    return(Option_Price)
lookback(98,100,0.03,0.12,1,1000,100000,12345, type = 'P')

'''MC with antithetic'''
def gen_sn(M,I,anti):
    if anti is True:
        sn = np.random.standard_normal((int(I/2),int(M)))
        sn = np.concatenate((sn,-sn),axis = 0)
    else:
        sn = np.random.standard_normal(I)
    # if moment_match is True:
    #     sn = (sn-sn.mean())/sn.std()
    return sn
A = gen_sn(50,10000,True)

def MC_antithetic(S0,K,r,sigma,T,M,I,seed, anti):
    # S0 = 100;K = 100;T = 1.0;r = 0.05;sigma = 0.2;M = 100;I = 10
    dt = T / M
    # Simulating I paths with M time steps
    S = np.zeros ((I,M + 1))
    S[:, 0] = S0
    np.random.seed(123)
    z = gen_sn(M+1,I,anti)
    for t in range ( 1, M + 1 ):
        if anti == True:
            S[:, t] = S[:, t - 1] * np.exp ( (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt ( dt ) * z[:,t])
        else:
            z = gen_sn ( M + 1, I, anti)
            S[:, t] = S[:, t - 1] * np.exp ( (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt ( dt ) * z )
    return S

#### 8/18/17####



# Stock_path1 = MC_antithetic(100,100,0.05,0.2,1,50,10000,123, anti = False)
# Stock_path2 = MCsim(100,100,0.05,0.2,1,50,10000,123)
# call = np.mean(np.maximum (Stock_path1[:,-1] - 100, 0 ) ) * np.exp ( -.05 * 1 )
# call1 = np.mean(np.maximum (Stock_path2[:,-1] - 100, 0 ) ) * np.exp ( -.05 * 1 )


# # Parameters
# S0 = 100; K = 100; T = 1.0; r = 0.05; sigma = 0.2
# M = 100; dt = T / M; I = 50000
# # Simulating I paths with M time steps
# S1 = np.zeros((M + 1, I))
# S1[0] = S0
# np.random.seed ( 12345 )
# for t in range(1, M + 1):
#     z = np.random.standard_normal(I) # pseudorandom numbers
#     S1[t] = S1[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
#     + sigma * math.sqrt(dt) * z)
# # vectorized operation per time step over all paths
# # Calculating the Monte Carlo estimator
# C1 = math.exp(-r * T) * np.sum(np.maximum(S1[-1] - K, 0)) / I pri

# '''Generating correlated Random Numbers'''
# def problem2q2(seed,I,M,rho,sd):
#      np.random.seed(seed)
#      Z = np.random.standard_normal(2*I*M)
#      z1 = Z[:I*M]
#      z2 = Z[I*M:]
#      a1 = rho/sd # a = rho/SDZ1; in this case SDZ1 = 1
#      b1 = np.sqrt(1-a1**2)
#      X = z1
#      Y = a1*z1 + b1*z2
#      output = {}
#      output['X'] = X
#      output['Y'] = Y
#      return output
#
# def problem4(seed,I,T,M):
#     np.random.seed(12345)# fix later
#     I = 10
#     M = 50
#     Temp  = problem2q2(12345,I,M,-0.6,1)
#     W4_1 = Temp['X']
#     W4_2 = Temp['Y']
#     W4_1.shape = (I, M)
#     W4_2.shape = (I, M)
#     r,s0,v0,sigma,alpha,beta = 0.03,48,0.05,0.42,5.8,0.0625
#     T = 2
#     h = T/M # T is my time  i.e 2 years and i am generating 100 paths
#
#     # Full Truncation
#     S_K =  np.zeros((I,M))
#     V_K = np.zeros((I,M))
#     S_K[:,0] = s0
#     V_K[:,0] = v0
#     for j in range (1,M):
#         # for i in range(0,I):
#         V_K[:, j] = V_K[:, j-1] + alpha * (beta - np.maximum(V_K[:, j-1], 0 )) * h + sigma * np.sqrt(np.maximum( V_K[:, j-1],0)) * W4_2[:, j] * np.sqrt ( h )
#         S_K[i,j+1] = S_K[i,j] + r * S_K[i,j] * h + np.sqrt(max(V_K[i,j],0)) * S_K[i,j] * W4_1[i,j] * np.sqrt(h)
#
#     K = 50
#     Call_full = np.maximum((S_K[:,M-1] - K),0)
#     c1 = np.exp(-r * T) * np.mean(Call_full)

