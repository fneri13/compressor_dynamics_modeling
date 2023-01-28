#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:41:32 2023

@author: Francesco Neri, TU Delft

Exercise on poles location for the centrifugal compressor described in "The Effect of Size and Working Fluid on the
Multi-Objective Design of High-Speed Centrifugal Compressors" by Andrea Giuffre
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
shape_fac = 0.9 #impeller shape factor
blades = 14 #number of blades (normal +split)
diff_radius_ratio = 1.5 #diffuser radius ratio
diff_height_ratio = 0.85 #diffuser blade height ratio
diff_pinch = 1.3 #pinch ratio
Lax = 0.7 #non dimensional length based on R2

#COMPRESSOR DESIGN SELECTED ALONG THE PARETO FRONT (SI units) - INPUT DATA
PRtt = 3.45 #totl to total pressure ratio
mdot = 0.114 #nominal mass flow rate
fluid = 'R1233zd(E)' #fluid
Pt1 = 47.789*1e3 #inlet total pressure
Tt1 = 278.13 #inlet total temperature
Ns = 1.208 #specific speed 
Mw1s = 1.12 #relative mach number at shroud at 1 location
phi_t1 = 0.106 #swallowing capacity at location 1
SP = 0.019 #size parameter
omega = np.array([68,77,85,90,94])*1e3*2*np.pi/60 #angular velocities
R1s = 15.2*1e-3 #shroud radius inlet
R1h = 3.4*1e-3 #hub radius inlet
R2 = 22.8*1e-3 #impeller exit radius
R3 = 35.2*1e-3 #diffuser outlet radius
H2 = 2.3*1e-3 #blade heigth exit impeller
H3 = 1.6*1e-3 #diffuser height
Lax = 16*1e-3 #axial length
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
R4 = 49.3*1e-3 # external diameter compressor
Dcn_out = 17.8*1e-3 #cone diameter out
ts = 0.3*1e-3 #blade trailing edge at shroud
th = 0.6*1e-3 #blade trailing edge at hub
eps_t = 0.15*1e-3 #tip clearance gap
eps_b = 0.15*1e-3 #back face clearance

#PARAMETERS COMPUTED
R1 = (R1s+R1h)/2 #radius at impeller inlet
























