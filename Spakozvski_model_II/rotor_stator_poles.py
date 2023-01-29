#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:11:33 2023

@author: fneri
Exercise 5.4.3 of Spakovszky thesis
"""


import matplotlib.pyplot as plt
import numpy as np
from functions import *
import os



# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (9,5)

#create directory for pictures
path = "pics"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

#%%INPUT DATA
Vx1 = 0.34 #non dimensional background axial flow velocity at inlet
Vy1 = 0 #non dimensional background azimuthal flow velocity at outlet
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

#axial cordinates
x1 = 0
x2 = c_r*np.cos(gamma_r)
x3 = DeltaX
x4 = x3 + c_s*np.cos(gamma_s)

#velocities across the stages
Vx2 = Vx1
Vy2 = Vx2*np.tan(alfa2)
Vx3 = Vx1
Vy3 = Vy2
Vx4 = Vx1
Vy4 = 0

#%% compute the results for each harmonic

#system function
def rotor_stator(s, n, theta=0):
    m1 = np.linalg.inv(Tax_n(x4, s, theta, n, Vx4, Vy4))
    m2 = Bsta_n(s, theta, n, Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana)
    m3 = Bgap_n(x2, x3, s, theta, n, Vx2, Vy2)
    m4 = Brot_n(s, theta, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb)
    m5 = Tax_n(x1, s, theta, n, Vx1, Vy1)
    m6 = np.linalg.multi_dot([m1,m2,m3,m4,m5])
    EC = np.array([[1,0,0]])
    IC = np.array([[0,1,0],
                    [0,0,1]])
    Y = np.concatenate((np.matmul(EC,m6),IC))
    return np.linalg.det(Y)

domain = [-2.5,0.5,-0.5,4.5]
grid = [5,5]
n=np.arange(1,7)
plt.figure(figsize=format_fig)
for nn in n:
    poles = shot_gun_method2(rotor_stator,domain, grid, nn, attempts = 3)
    plt.plot(poles.real,-poles.imag,'o', label='n '+str(nn))
real_axis_x = np.linspace(domain[0],domain[1],100)
real_axis_y = np.zeros(len(real_axis_x))   
imag_axis_y = np.linspace(domain[2],domain[3],100)
imag_axis_x = np.zeros(len(imag_axis_y))
plt.plot(real_axis_x,real_axis_y,'--k', linewidth=0.5)
plt.plot(imag_axis_x,imag_axis_y,'--k', linewidth = 0.5)
plt.xlim([domain[0],domain[1]])
plt.ylim([domain[2],domain[3]])
plt.legend()
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.title('Root locus')
plt.savefig(path+'/poles_rotor_stator.png')



















