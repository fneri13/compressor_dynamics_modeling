#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:51:23 2022

@author: fn

stall propagations speed plot, as described in equation 46 of the paper: "A Theory of Rotating Stall of Multistage 
Axial Compressors: Part I—Small Disturbances"

"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (9,6)


#%%
"""
FUNCTION FOR THE PROPAGATING SPEED
the numerical values of different parameters have been set up in order to obtain the functional shape of the paper
"""

def PropagationSpeed_m(m,tau_c_star,phi=1,n=1,D=1,N=1,L=1,tau_v_star=1):
    return 0.5 / ( 1 + m*phi*D/(n*2*N*L*tau_c_star) + tau_v_star/(tau_c_star*N) ) 


m = np.linspace(1,2,100)

tau_c_star = np.linspace(1,11,5)
plt.figure()
for s in range(len(tau_c_star)):
    tau = tau_c_star[s]
    PropSpeed = PropagationSpeed_m(m, tau)
    plt.plot(m,PropSpeed, label = r'$ \tau =$ ' + str(tau))
plt.xlabel('m')
plt.ylabel('f')
plt.title('Propagation speed coefficient for positive lags')
plt.legend()


tau_c_star = np.linspace(-60,-10,5)
plt.figure()
for s in range(len(tau_c_star)):
    tau = tau_c_star[s]
    PropSpeed = PropagationSpeed_m(m, tau)
    plt.plot(m,PropSpeed, label = r'$ \tau =$ ' + str(tau))
plt.xlabel('m')
plt.ylabel('f')
plt.title('Propagation speed coefficient for negative lags')
plt.legend()


#here we see the dependence on tau, i.e. the ranges of interest
# tau = np.linspace(-10,10,100)
# m = 1.5
# plt.figure()
# plt.plot(tau,PropagationSpeed_m(m,tau))
# plt.xlabel(r'$\tau$')
# plt.ylabel('f')
# plt.title('Propagation speed at constant m')
    



