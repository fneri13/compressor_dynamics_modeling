#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:41:32 2023

@author: Francesco Neri, TU Delft

Exercise on poles location for the centrifugal compressor described in "The Effect of Size and Working Fluid on the
Multi-Objective Design of High-Speed Centrifugal Compressors" by Andrea Giuffre. The compressor selected is the one
picked from the pareto front.
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
format_fig = (15,10)

#%% Input parameters for the compressor selected by Andrea on the Pareto front. All the variables that begin with capital
# letters are dimensional. Otherwise they have been non-dimensionalized

SP = 0.9 #impeller shape factor
blades = 14 #number of blades (normal +split)
n_max = blades//4+1 #maximum harmonic to take in consideration (2 blades in a half-wavelength)
diff_radius_ratio = 1.5 #diffuser radius ratio
diff_height_ratio = 0.85 #diffuser blade height ratio
diff_pinch = 1.3 #pinch ratio
L_ax = 16e-3 #dimensional x position of impeller exit - impeller inlet 

#COMPRESSOR DESIGN SELECTED ALONG THE PARETO FRONT (SI units) - INPUT DATA
PRtt_design = 3.45 #totl to total pressure ratio
mdot_design = 0.114 #nominal mass flow rate
fluid = 'R1233zd(E)' #fluid
Pt1_design = 47.789*1e3 #inlet total pressure [kPa]
Tt1 = 278.13 #inlet total temperature [K]
Ns = 1.208 #specific speed 
Mw1s = 1.12 #relative mach number at shroud at 1 location
phi_t1 = 0.106 #swallowing capacity at location 1
SP = 0.019 #size parameter
Omega = np.array([68,77,85,90,94])*1e3*2*np.pi/60 #angular velocities [rad/s]
R1s = 15.2*1e-3 #shroud radius inlet [m]
R1h = 3.4*1e-3 #hub radius inlet [m]
R2 = 22.8*1e-3 #impeller exit radius [m]
R3 = 35.2*1e-3 #diffuser outlet radius [m]
H2 = 2.3*1e-3 #blade heigth exit impeller [m]
H3 = 1.6*1e-3 #diffuser height [m]
Lax = 16*1e-3 #axial length [m]
Nbl = 7 #number of blades
Nsplit = 7 #splitter blades
Ds = 1.870 #specific diameter
Mu2 = 1.5 #peripheral velocity at station 2
psi_is = 0.793 #isentropic work coefficient 
k = 0.95 #impeller shape factor 
beta1_s = 60*np.pi/180 #inlet relative angle at shroud
beta1_h = 16.5*np.pi/180 #inlet relative angle at hub
beta2 = 29.6*np.pi/180 #outlet relative angle impeller
Ra = 0.032*1e-3 #roughness 
R4 = 49.3*1e-3 # external diameter compressor [m]
Dcn_out = 17.8*1e-3 #cone diameter out [m]
Ts = 0.3*1e-3 #blade trailing edge at shroud [m]
Th = 0.6*1e-3 #blade trailing edge at hub [m]
Eps_t = 0.15*1e-3 #tip clearance gap [m]
Eps_b = 0.15*1e-3 #back face clearance [m]

R1 = (R1s+R1h)/2 #radius at impeller inlet [m]

#Reference parameters for non-dimensionalization
R_Ref = R2
U_Ref = Omega[2]*R_Ref

#STATION LOCATION
x1 = 0
x2 = L_ax/R_Ref
x3 = 16
x4 = 16

r1 = R1/R_Ref
r2 = R2/R_Ref
r3 = R3/R_Ref
r4 = R4/R_Ref
#%%

#axial velocities
vx1 = 
vx2 = 
vx3 = 
vx4 = 
vx5 = 

#azimuthal velocities
vy1 = 
vy2 = 
vy3 = 
vy4 = 
vy5 = 
 
#radial velocities
vr1 = 
vr2 = 
vr3 = 
vr4 = 
vr5 = 

#static pressures [Pa]
p1 = 
p2 = 
p3 = 
p4 = 

#static density [kg/m3]
rho1 = 
rho2 = 
rho3 = 
rho4 = 


















