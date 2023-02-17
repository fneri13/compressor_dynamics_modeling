#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:55:21 2023

@author: fneri
chapter 5.6.1 spakovszky thesis
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


#%%DATA INPUT OF THE COMPRESSOR
mdot = 4.54 #mass flow rate
PR = 4 #pressure ratio impeller and vaned diffuser
Corr_speed = 21789*2*np.pi/60 #standard day corrected speed
U2 = 492 #exit tip speed
Nbl = 15 #impeller blades
Nvd = 24 #vaned diffuser blades
Nspl = 15 #splitter blades
Nspl = 15 #splitter blades
beta2 = -50*np.pi/180 #backsweep at exit of blades
R1t = 105*1e-3 #inlet tip radius
blade_height_1 = 64*1e-3 #inlet blade height
R2 = 215.5*1e-3 #inlet tip radius
blade_height_1 = 17*1e-3 #exit blade height
R3 = R2*1.078 #LE radius of vaned diffuser
div_angle = 7.8*np.pi/180 #divergence angle of vane blades
R4 = 181.5*1e-3 #diffuser outlet radius
s_i = 1.3064 #gas path length in impeller
s_d = 1.1187 #diffuser path length
lambda_i = 1.1508 #impeller inertia factor
lambda_d = 0.8518 #diffuser inertia factor

#STATIONS
x0 = -1.5957 #non dimensionalized by impeller exit radius
r0 = 0.7483 
A0 = 4.1276 #area at station 0 non-dimensionalized by impeller exit radius
x1 = 0 
r1 = 0.3533
A1 = 0.6276
x2 = 0
r2 = 1.0
A2 = 0.4965
x3 = 0
r3 = 1.0779
A3 = 0.5330
x4 = 0
r4 = 1.6838
A4 = 0.8339
x5 = 0.2211
r5 = 1.8428
A5 = 0.8189

#%% LOSSES DATA INTERPOLATION
beta1_data = np.array([-61.9, -61.4, -61.2, -60.7, -60.2, -59.7, -58.9, -58])*np.pi/180 #inlet relative flow angle
Loss_i_data = np.array([0.231, 0.229, 0.224, 0.215, 0.201, 0.19, 0.188, 0.193]) #loss
z_coeff_i = np.polyfit(beta1_data, Loss_i_data, 3)
beta1 = np.linspace(beta1_data[0], beta1_data[-1], 100)
Loss_i = np.polyval(z_coeff_i, beta1)

alpha3_data = np.array([76.75, 77.4, 77.75, 78, 78.3, 78.55, 78.65, 79])*np.pi/180 #inlet relative flow angle
Loss_d_data = np.array([0.27, 0.217, 0.18, 0.163, 0.15, 0.142, 0.14, 0.147]) #loss
z_coeff_d = np.polyfit(alpha3_data, Loss_d_data, 3)
alpha3 = np.linspace(alpha3_data[0], alpha3_data[-1], 100)
Loss_d = np.polyval(z_coeff_d, alpha3)

plot_bool = True #True to plot the data

if plot_bool:
    fig, ax = plt.subplots(1,2,figsize=(17,8))
    ax[0].plot(beta1_data*180/np.pi, Loss_i_data,'ko', label='Data')
    ax[0].plot(beta1*180/np.pi, Loss_i, label='3rd order polynomial fit')
    ax[1].plot(alpha3_data*180/np.pi, Loss_d_data,'ko')
    ax[1].plot(alpha3*180/np.pi, Loss_d)
    ax[0].set_title('Impeller Loss')
    ax[0].set_xlabel(r'$\beta_1$'+' [deg]')
    ax[0].set_ylabel(r'$L_{imp}$')
    ax[1].set_title('Diffuser Loss')
    ax[1].set_xlabel(r'$\alpha_3$'+' [deg]')
    ax[1].set_ylabel(r'$L_{dif}$')
    fig.legend()

#%%ESTIMATION OF OPERATING POINT USED BY SPAKOVSZKY, based on 0.149 seen on the poles plot
rho = 1.014 #my assumption on air density
U_ref = U2 #reference velocity, it should be inlet tip speed
R_ref = R2 #the reference length should be R2
V2r_op = 0.149*U_ref
mdot_op = A2*R_ref**2 * rho * V2r_op
V1_x_op = mdot_op/(A1*R_ref**2)/rho
U1_op = U_ref*r1/r2
beta1_op = -np.arctan(U1_op/V1_x_op)
beta1_op_deg = beta1_op*180/np.pi

W2_op = V2r_op/np.sin(beta2)
alpha3_op = np.arctan((U2-W2_op*np.sin(beta2))/V2r_op)
alpha3_op_deg = alpha3_op *180/np.pi












