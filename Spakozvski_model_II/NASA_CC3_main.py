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
Nspl = 15 #splitter blades
beta2 = -50*np.pi/180 #backsweep at exit of blades
R1t = 105*1e-3 #inlet tip radius
blade_height_1 = 64*1e-3 #inlet blade height
R2 = 215.5*1e-3 #inlet tip radius
blade_height_1 = 17*1e-3 #exit blade height
R3 = R2*1.078 #LE radius of vaned diffuser
div_angle = 7.8*np.pi/180 #divergence angle of vane blades
R4 = 181.5*1e-3 #diffuser outlet radius
x0 -1.5957
