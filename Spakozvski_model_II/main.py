#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn

try to replicate section 5.1 of Spakovszky PhD thesis
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import *
from cxroots import Rectangle
from mpmath import findroot
from scipy.optimize import root



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
x1 = 0
x2 = 0.3
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
gap = Bgap_n(x1,x2,s,theta,n,Vx,Vy)
    
    
    
 #%%   
#test function per implementare shot gun method

def test_fun(s):
    #stupid function to test
    return s**7+1

# def test_fun_prime(s):
#     fun_right = test_fun(s+0.001)
#     fun_left = test_fun(s-0.001)
#     fun_up = test_fun(s+0.001)
#     fun_down = test_fun(s-0.001)
#     fun_prime_real = (fun_right-fun_left)/(0.002)
#     fun_prime_imag = (fun_up-fun_down)/(0.002)
#     return fun_prime_real+1j*fun_prime_imag

def derivative(f, x=s, h=0.001):
    return (f(x + h) - f(x - h)) / (2*h)


def system_fun(s, n=1, theta=0):
    m1 = np.linalg.inv(Tax_n(0.3, s, theta, n, Vx, Vy))
    m2 = Bsta_n(s, theta, n, Vx, Vy1, Vy2, alfa1, alfa2, lambda_s, dLs_dTana)
    m3 = Bgap_n(x1, x2, s, theta, n, Vx, Vy)
    m4 = Brot_n(s, theta, n, Vx, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb)
    m5 = Tax_n(0.1, s, theta, n, Vx, Vy)
    m6 = np.matmul(m1,m2)
    m6 = np.matmul(m6,m3)
    m6 = np.matmul(m6,m4)
    m6 = np.matmul(m6,m5)
    EC = np.array([[1,0,0]])
    IC = np.array([[0,1,0],
                   [0,0,1]])
    Y = np.concatenate((np.matmul(EC,m6),IC))
    return np.linalg.det(Y)


s_r_min = -3
s_r_max = +1
s_i_min = -3
s_i_max = +3
s = -1
radius = 10
i=1
pole_list = mapping_poles(s_r_min, s_r_max, s_i_min, s_i_max, system_fun)


#%%
n_grid = 5
s_real = np.linspace(-10,10,n_grid)
s_imag = np.linspace(-10,10,n_grid)
span = (s_real[-1]-s_real[0])/(n_grid-1)
real_grid, imag_grid = np.meshgrid(s_real,s_imag)
radius = span/np.sqrt(2)


for i in range(0,len(s_real)):
    for j in range(0,len(s_imag)):
        pole_list=shot_gun_method(system_fun, s_real[i]+s_imag[j], radius, N=100, i=0, tol=1e-2, attempts_max=30)

pole_list = np.array([i for i in pole_list if i is not None])
plt.figure(figsize=(10,6))
plt.plot(pole_list.real,pole_list.imag,'ko') 
plt.xlim([s_real[0],s_real[-1]])
plt.ylim([s_imag[0],s_imag[-1]])
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.grid()
plt.title('Root locus')
#%%
# pole_list=shot_gun_method(test_fun, 0, 5, N=50, i=0, tol=1e-3, attempts_max=30)
n_grid = 10
s_real = np.linspace(-10,10,n_grid)
s_imag = np.linspace(-10,10,n_grid)
span = (s_real[-1]-s_real[0])/(n_grid-1)
real_grid, imag_grid = np.meshgrid(s_real,s_imag)
radius = span/np.sqrt(2)
pole_list=[]
for i in range(0,len(s_real)):
    for j in range(0,len(s_imag)):
        pole_list.append(shot_gun_method(test_fun, s_real[i]+s_imag[j], radius, N=50, i=0, tol=1e-2, attempts_max=30))

# flattent the list
poles = [item for sublist in pole_list for item in sublist]
#cancel the copies
copies = []
for ii in range(1,len(poles)):
    difference = np.abs(poles[ii]-poles[0])
    if difference<0.001:
        copies.append(ii)
for kk in sorted(copies, reverse=True):
    del poles[kk]
poles = np.array(poles)

plt.figure(figsize=(10,6))
plt.scatter(poles.real,poles.imag)
plt.xlim([s_real[0],s_real[-1]])
plt.ylim([s_imag[0],s_imag[-1]])
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.grid()
plt.title('Root locus')
















