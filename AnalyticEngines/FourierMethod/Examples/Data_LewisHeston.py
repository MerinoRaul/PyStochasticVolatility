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
no_strikes = 7
k_s = np.array([50, 60, 70, 75, 80.0, 82.5, 85, 87.5, 90.0, 92.5, 95, 96,97,98, 99,100.0, 101, 102, 103, 104, 105, 107.5,  110.0, 112.5, 115,117.5, 120.0, 130.0, 135,140, 150])
#k_s = np.linspace(70.0, 130.0, no_strikes)
f0 = 100.0
x0 = np.log(f0)
#T = 2.0
T_s=np.array([0.25, 0.5, 0.75, 1, 1.5, 2,2.5,3])

num_e=25
num_k=10
num_rho=9
num_v0=10#25
num_theta=10#25


# Heston parameters
epsilon =np.linspace(0,3,num_e)
k = np.linspace(0,1,num_k)
rho = np.linspace(-1,1,num_rho)
v0 = np.linspace(0,1,num_v0)
theta = np.linspace(0,1,num_theta)
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

                        for j in range(0, no_strikes):
                            european_option = EuropeanOption(k_s[j], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0,T_s[i_Ts])

                            data[i, 0] = f0
                            data[i, 1] = epsilon[i_e]
                            data[i, 2] = k[i_k]
                            data[i, 3] = rho[i_rho]
                            data[i, 4] = v0[i_v0]
                            data[i, 5] = theta[i_theta]
                            data[i, 6] = T_s[i_Ts]
                            data[i, 7] = k_s[j]
                            data[i, 8] = european_option.get_analytic_value(0.0,  theta[i_theta], rho[i_rho], k[i_k],  epsilon[i_e], v0[i_v0], 0.0,model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_LEWIS,compute_greek=False)
                            try:
                                data[i, 9] = implied_volatility(data[i, 8], f0, k_s[j], T_s[i_Ts], 0.0, 0.0, 'c')
                            except:
                                continue

                            i=i+1



end_time = time.time()
diff_time_cf = end_time - start_time

print(diff_time_cf)

myfile=open('csvHestonLewis_prices', 'w', newline='')
with myfile:
    writer=csv.writer(myfile)
    writer.writerows(data)