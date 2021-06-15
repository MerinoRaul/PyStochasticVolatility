import numpy as np
import matplotlib.pylab
from AnalyticEngines.FourierMethod.CharesticFunctions import HestonCharesticFunction
from functools import partial
from Tools import Types
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import csv
import time

# European option price
no_strikes = 11
#k_s = np.array([50, 60, 70, 75, 80.0, 82.5, 85, 87.5, 90.0, 92.5, 95, 97, 99,100.0, 101, 103, 105, 107.5,  110.0, 112.5, 115, 117.5, 120.0, 130.0, 140, 150])

k_s = np.linspace(50.0, 150.0, no_strikes)
no_strikes=len(k_s)
f0 = 100.0
x0 = np.log(f0)
#T = 2.0
T_s=np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25,  1.5, 1.75, 2, 2.5,3])

num_e=24
num_k=10
num_rho=9
num_v0=10#25
num_theta=10#25


# Heston parameters
epsilon =np.linspace(0.125,3,num_e)
k = np.linspace(0.1,1,num_k)
rho = np.linspace(-1,1,num_rho)
v0 = np.linspace(0.1,1,num_v0)
theta = np.linspace(0.1,1,num_theta)
epsilon[0]=0.1
k[0]=0.1
v0[0]=0.1
theta[0]=0.1
#b2 = k
u2 = -0.5

total_size=num_e*num_k*num_rho*num_v0*num_theta*len(T_s)*no_strikes
data=np.zeros((total_size,10))
# Upper and lower bound for cos integral
a = -10.0
b = 10.0
notional=1


i = 0
options = []

start_time = time.time()
for i_e in range(0, num_e):
    for i_k in range(0, num_k):
        for i_rho in range(0, num_rho):
            for i_v0 in range(0, num_v0):
                for i_theta in range(0, num_theta):
                    for i_Ts in range(0,len(T_s)):
                        b2 = k[i_k]

                        cf_heston = partial(HestonCharesticFunction.get_trap_cf, t=T_s[i_Ts], r_t=0.0, x=x0, v=v0[i_v0], theta=theta[i_theta], rho=rho[i_rho], k=k[i_k], epsilon=epsilon[i_e], b=b2, u=u2)
                        cos_price= COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 2**12, k_s, cf_heston)

                        for j in range(0, no_strikes):
                            data[i, 0] = f0
                            data[i, 1] = epsilon[i_e]
                            data[i, 2] = k[i_k]
                            data[i, 3] = rho[i_rho]
                            data[i, 4] = v0[i_v0]
                            data[i, 5] = theta[i_theta]
                            data[i, 6] = T_s[i_Ts]
                            data[i, 7] = k_s[j]
                            data[i, 8] = cos_price[j]
                            try:
                                data[i, 9] = implied_volatility(data[i, 8], f0, k_s[j], T_s[i_Ts], 0.0, 0.0, 'c')
                            except:
                                continue

                            i=i+1



end_time = time.time()
diff_time_cf = end_time - start_time

print(diff_time_cf)

myfile=open('csvHeston_prices', 'w', newline='')
with myfile:
    writer=csv.writer(myfile)
    writer.writerows(data)