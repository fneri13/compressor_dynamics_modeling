#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn

try to replicate section 5.1 of Spakozsvki PhD thesis
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

#%% exercise rotor stator pre-stall modes

#Geometrical and compressor parameters (pag. 147)
Q = 0.215 #source term of the swirling flow
GAMMA = 0.7032 #rotational term of the swirling flow
N = 20 #number of blades
Vx = 0.34 #non dimensional background axial flow velocity at inlet
Vy = 0 #non dimensional background azimuthal flow velocity at outlet
DeltaX = 0.3 #gap between rotor and stator

#rotor parameters (pag. 147)
beta1 = -71.1*np.pi/180 #relative inlet swirl
alfa1 = 0*np.pi/180 #absolute inlet swirl
beta2 = -35*np.pi/180 #relative outlet swirl
alfa2 = 65.7*np.pi/180 #absolute outlet swirl
dLr_dPhi = -0.6938 #steady state rotor loss derivative at background condition
dLr_dTanb = dLr_dPhi/((np.tan(alfa1)-np.tan(beta1))**2) #steady state rotor loss derivative at background condition
c_r = 0.135 #blade chord
gamma_r = -50.2*np.pi/180 #stagger angle rotor blades
lambda_r = 0.212 #inertia parameter rotor

#stator parameters (pag. 147)
beta3 = -35*np.pi/180 #relative inlet swirl
alfa3 = 65.7*np.pi/180 #absolute inlet swirl
beta4 = -71.1*np.pi/180 #relative outlet swirl
alfa4 = 0.0*np.pi/180 #absolute outlet swirl
dLs_dTana = 0.0411 #steady state stator loss at inlet condition of the stator
c_s = 0.121 #blade chord
gamma_s = 61.8*np.pi/180 #stagger angle rotor blades
lambda_s = 0.256 #inertia parameter rotor
  

#%%debug parameters
r = 1
n = 1
s = 1+1j
theta = np.pi/4
x = 0.3
Vx1 = 1
Vx2 = 0.9
Vr1 = 0.7
Vr2 = 0.5
Vy1 = 0.3
Vy2 = 0.2
r1 = 0.3
r2 = 0.8
rho1 = 1.014
rho2 = 1.3
A1 = 1
A2 = 0.8
s_i = 0.8
s_dif = 0.7
dLi_dTanb = -0.9
dLd_dTana = -0.5

#build all the functions written in the function file
swirler = Trad_n(r, n, s, theta, Q, GAMMA)  
axial = Tax_n(x, s, theta, n, Vx1, Vy2)  
stator = Bsta_n(s, theta, n, Vx, Vy1, Vy2, alfa1, alfa2, lambda_s, dLs_dTana)
rotor = Brot_n(s, theta, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb) 
impeller = Bimp_n(s, theta, n, Vx1, Vr2, Vy1, Vy2, alfa1, beta1, beta2, r1, r2, rho1, rho2, A1, A2, s_i, dLi_dTanb)
diffuser = Bdif_n(s, theta, n, Vr1, Vr2, Vy1, Vy2, alfa1, beta1, alfa2, r1, r2, rho1, rho2, A1, A2, s_dif, dLd_dTana)
vaneless_diffuser = Bvlsd_n(s,theta,n,r1,r2,Q,GAMMA)    
    
    
    
 #%%   
#test function per implementare shot gun method
def test_fun(s):
    #test function to implement shot gun method. The roots are at -0.5+/-0.5j*sqrt(3)
    return np.linalg.det(Trad_n(r, n, s, theta, Q, GAMMA))
    # return s**3-7*s**2+s+1




def mapping_poles(s_r_min,s_r_max,s_i_min,s_i_max,Myfunc):
    grid_real = 10
    grid_im = 10
    s_real = np.linspace(s_r_min,s_r_max,grid_real)
    s_im = np.linspace(s_i_min,s_i_max,grid_im)
    real_grid, imag_grid = np.meshgrid(s_real,s_im)
    poles = np.zeros((len(s_real),len(s_im)))
    for i in range(0,len(s_real)):
        for j in range(0,len(s_im)):
            if np.abs(Myfunc(s_real[i]+1j*s_im[j]))<0.01:
                poles[i,j] = 1
            else:
                poles[i,j] = 0
            # poles[i,j] = np.abs(Myfunc(s_real[i]+1j*s_im[j]))
    
    plt.figure(figsize=(10,6))
    plt.contourf(real_grid, imag_grid, poles, cmap='bwr')
    plt.xlabel(r'$\sigma_{n}$')
    plt.ylabel(r'$j \omega_{n}$')
    plt.title('Stability Map')
    plt.colorbar()
    # plt.savefig(path+'/stability_map.png')
    return poles


s_r_min = -10
s_r_max = +10
s_i_min = -10
s_i_max = +10

# pole_map = mapping_poles(s_r_min, s_r_max, s_i_min, s_i_max, test_fun)
pole = shot_gun_method(test_fun,0,5,10, 1)




















