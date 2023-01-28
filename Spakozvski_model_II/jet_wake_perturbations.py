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
format_fig = (12,8)

#%%DATA INPUT OF THE EXERCISE
R2 = 1 #initial radius
R3 = 1.5 #outlet radius
Nbl = 20 #number of blades
Q = 0.215 #source term
GAMMA = 0.7032 #circulation term
theta_max = 2*np.pi/Nbl
azimuthal_sampling_points = 100
theta = np.linspace(0,theta_max,azimuthal_sampling_points)
theta_deg = theta *180/np.pi
n = 20 #interested in the 20 harmonic, since we are looking at perturbation coming from the 20th harmonics
theta0 = theta[0]

#INITIAL CONDITIONS
Wr_2 = np.exp(1j*n*theta0)
Wt_2 = np.exp(1j*n*theta0)*(GAMMA-1)/Q
s = -1j*n #the mode is the 20th harmonics

#radial position where we want the data
radii = np.array([1.0,1.05,1.1,1.15,1.2,])

#boundary conditions are prescribed inlet velocities, and outlet pressure at R3 = 0
firstC = np.array([[1,0,0]])
secondC = np.array([[0,1,0]])
thirdC = np.array([[0,0,1]])
T2 = Trad_n(R2, n, s, theta0, Q, GAMMA)
T3 = Trad_n(R3, n, s, theta0, Q, GAMMA)
Y = np.zeros((3,3), dtype=complex)
Y[0,:] = np.matmul(firstC,T2)
Y[1,:] = np.matmul(secondC,T2)
Y[2,:] = np.matmul(thirdC,T3)
BC_vec = np.zeros((3,1),dtype=complex)
BC_vec[0] = np.exp(1j*n*theta0)
BC_vec[1] = ((GAMMA-1)/Q)*np.exp(1j*n*theta0)
BC_vec[2] = 0

#find the potential and vortical moes in the system that satisfy the BC
DEN_mode = np.matmul(np.linalg.inv(Y),BC_vec)

plt.figure()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$w_r$')
for k in range(0,len(radii)):
    radius = radii[k]
    #compute now the flow solutions
    vec = np.zeros((3,len(theta)),dtype=complex)
    i = 0
    for t in theta:
        vec[:,i] = np.matmul(Trad_n(radius, n, s, theta[i], Q, GAMMA),DEN_mode).reshape(3)
        i = i+1
        
    plt.plot(theta_deg,vec[2,:],label='r='+str(radius))

    
    # plt.figure()
    # plt.plot(theta_deg,vec[1,:])
    # plt.xlabel(r'$\theta$')
    # plt.ylabel(r'$w_{\theta}$')
    
    # plt.figure()
    # plt.plot(theta_deg,vec[2,:])
    # plt.xlabel(r'$\theta$')
    # plt.ylabel(r'$p$')
plt.legend()









