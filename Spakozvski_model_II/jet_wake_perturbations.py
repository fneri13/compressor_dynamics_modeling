#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:58:47 2023

@author: fneri
exercise taken from 2.5.2 of Spakovszky model
"""


import matplotlib.pyplot as plt
import numpy as np
from functions import *

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (10,8)

#%%DATA INPUT OF THE EXERCISE
R2 = 1 #initial radius
R3 = 1.5 #outlet radius
Nbl = 20 #number of blades
Q = 0.215 #source term
GAMMA = 0.7032 #circulation term
theta_max = 2*np.pi/Nbl #period of the perturbation
azimuthal_sampling_points = 100 #number of points along the period
theta = np.linspace(0,theta_max,azimuthal_sampling_points) #theta domain
theta_deg = theta *180/np.pi #theta domain in degrees
n = 20 #interested in the 20th harmonic, since we are looking at perturbation coming from the 20th harmonics
theta0 = theta[0]

#INITIAL CONDITIONS
Wr_2 = np.exp(1j*n*theta0)
Wt_2 = np.exp(1j*n*theta0)*(GAMMA-1)/Q

#radial position where we want the data
radii = np.array([1.0,1.05,1.1,1.15,1.2])
# radii = np.array([1,1.05,1.1])

r0 = R2 #this is the big problem. no reason for it. there is no way to understand what is its meaning
#boundary conditions are prescribed inlet velocities, and outlet pressure at R3 = 0
T2 = Trad_n(R2, r0, n, -1j*n, theta0, Q, GAMMA)
T3 = Trad_n(R3, r0, n, -1j*n, theta0, Q, GAMMA)
Y = np.zeros((3,3), dtype=complex)
Y[0,:] = T2[0,:]
Y[1,:] = T2[1,:]
Y[2,:] = T3[2,:]
BC_vec = np.zeros((3,1),dtype=complex)
BC_vec[0] = np.exp(1j*n*theta0)
BC_vec[1] = ((GAMMA-1)/Q)*np.exp(1j*n*theta0)
BC_vec[2] = 0
#find the potential and vortical modes in the system that satisfy the BC
DEN_mode = np.matmul(np.linalg.inv(Y),BC_vec)

# fig, axes = plt.subplots(3,1, figsize=format_fig)
# axes[0].set_ylabel(r'$\delta w_{r}$')
# axes[1].set_ylabel(r'$\delta w_{\theta}$')
# axes[2].set_ylabel(r'$\delta p $')
# axes[2].set_xlabel(r'$\theta $')

# for k in range(0,len(radii)):
#     radius = radii[k]
#     #compute now the flow solutions
#     vec = np.zeros((3,len(theta)),dtype=complex)
#     i = 0
#     for t in theta:
#         vec[:,i] = np.matmul(Trad_n(radius, r0 , n, -1j*n, t, Q, GAMMA),DEN_mode).reshape(3)
#         i = i+1       
#     axes[0].plot(theta_deg, vec[0,:], label='r='+str(radius))
#     axes[1].plot(theta_deg, vec[1,:])
#     axes[2].plot(theta_deg, vec[2,:])
# fig.legend()

#%% VARIATIONS WITH RADIUS
radii = np.linspace(R2,R3,100)
fig, axes = plt.subplots(3,1, figsize=format_fig)
axes[0].set_ylabel(r'$\delta w_{r}$')
axes[1].set_ylabel(r'$\delta w_{\theta}$')
axes[2].set_ylabel(r'$\delta p $')
axes[2].set_xlabel(r'$r $')
vec = np.zeros((3,len(radii)),dtype=complex)
for k in range(0,len(radii)):
    vec[:,k] = np.matmul(Trad_n(radii[k], r0 , n, -1j*n, theta0, Q, GAMMA),DEN_mode).reshape(3)
axes[0].plot(radii, vec[0,:],'-o')
axes[1].plot(radii, vec[1,:],'-o')
axes[2].plot(radii, vec[2,:],'-o')









