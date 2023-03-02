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
Lax = 16e-3 #dimensional x position of impeller exit - impeller inlet 

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
Omega_range = np.array([70,79,88,93,97])*1e3*2*np.pi/60 #angular velocities [rad/s]
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

#STATION LOCATION
x1 = 0
x2 = Lax/R_Ref
x3 = 16
x4 = 16

r1 = R1/R_Ref
r2 = R2/R_Ref
r3 = R3/R_Ref
r4 = R4/R_Ref

A1 = np.pi*(R1s**2-R1h**2)
A2 = 2*np.pi*R2*H2
A4 = 2*np.pi*R4*H3

#%%IMPORT DATA FROM DATA FOLDER (IRIS COMPRESSOR ANDREA)
# note: STA numbers are shifted (1 mine = 0 Andrea, 2 mine = 1 Andrea, 4 mine = 2 Andrea)
import pickle
data_folder = "../data/IRIS_single_stage/design0_beta_3.450/operating_map/" #path to folder

#in the following arrays the zeros means that the working point was out of range, both choked or stalled
#every row is a different RPM, given in the RPM vector
#every column is a different mass flow rate
#all the rest is ordered accordingly
with open(data_folder + 'alpha2.pkl', 'rb') as f:
    alpha4 = pickle.load(f) 
with open(data_folder + 'beta0.pkl', 'rb') as f:
    beta1 = pickle.load(f)
with open(data_folder + 'beta1.pkl', 'rb') as f:
    beta2 = pickle.load(f)
with open(data_folder + 'P0.pkl', 'rb') as f:
    P1 = pickle.load(f) #static pressure
with open(data_folder + 'P1.pkl', 'rb') as f:
    P2 = pickle.load(f)
with open(data_folder + 'P2.pkl', 'rb') as f:
    P4 = pickle.load(f)
with open(data_folder + 'Vm2.pkl', 'rb') as f:
    Vm4 = pickle.load(f)
with open(data_folder + 'Vt2.pkl', 'rb') as f:
    Vt4 = pickle.load(f)
with open(data_folder + 'Wm0.pkl', 'rb') as f:
    Wm1 = pickle.load(f)
with open(data_folder + 'Wm1.pkl', 'rb') as f:
    Wm2 = pickle.load(f)
with open(data_folder + 'Wt0.pkl', 'rb') as f:
    Wt1 = pickle.load(f)
with open(data_folder + 'Wt1.pkl', 'rb') as f:
    Wt2 = pickle.load(f)
with open(data_folder + 'D0.pkl', 'rb') as f:
    Rho1 = pickle.load(f) #density
with open(data_folder + 'D1.pkl', 'rb') as f:
    Rho2 = pickle.load(f)
with open(data_folder + 'D2.pkl', 'rb') as f:
    Rho4 = pickle.load(f)
with open(data_folder + 'eta_tt.pkl', 'rb') as f:
    eta_tt = pickle.load(f)
with open(data_folder + 'eta_ts.pkl', 'rb') as f:
    eta_ts = pickle.load(f)
with open(data_folder + 'mass_flow.pkl', 'rb') as f:
    mass_flow = pickle.load(f)
with open(data_folder + 'beta_ts.pkl', 'rb') as f:
    beta_ts = pickle.load(f) #total to static pressure ratio
with open(data_folder + 'beta_tt.pkl', 'rb') as f:
    beta_tt = pickle.load(f)
with open(data_folder + 'rpm.pkl', 'rb') as f:
    rpm = pickle.load(f)
    
# in order to understand the ordering
plt.figure('total to total efficiency')
plt.plot(mass_flow[0,:], eta_tt[0,:],'o', label='70 krpm')
plt.plot(mass_flow[1,:], eta_tt[1,:],'o', label='79 krpm')
plt.plot(mass_flow[2,:], eta_tt[2,:],'o', label='88 krpm')
plt.plot(mass_flow[3,:], eta_tt[3,:],'o', label='93 krpm')
plt.plot(mass_flow[4,:], eta_tt[4,:],'o', label='97 krpm')
plt.xlim([0.04,0.14])
plt.ylim([0.6, 0.85])
plt.xlabel(r'$\dot{m}$ [kg/s]')
plt.ylabel(r'$\eta_{tt}$')
plt.legend()

plt.figure('total to total pressure ratio')
plt.plot(mass_flow[0,:], beta_tt[0,:],'o', label='70 krpm')
plt.plot(mass_flow[1,:], beta_tt[1,:],'o', label='79 krpm')
plt.plot(mass_flow[2,:], beta_tt[2,:],'o', label='88 krpm')
plt.plot(mass_flow[3,:], beta_tt[3,:],'o', label='93 krpm')
plt.plot(mass_flow[4,:], beta_tt[4,:],'o', label='97 krpm')
plt.xlim([0.04,0.14])
plt.ylim([1.5, 6])
plt.xlabel(r'$\dot{m}$ [kg/s]')
plt.ylabel(r'$\beta_{tt}$')
plt.legend()





#%% WORKING CONDITIONS, DEPENDENT ON THE WORKING POINT
#attempt = index [2,0] which means medium speed, and most left point

Omega = rpm[2]*2*np.pi/60
U_Ref = Omega*R_Ref #the reference velocity is the outlet impeller peripheral speed
A_Ref = R_Ref**2 
a1 = A1/A_Ref
a2 = A2/A_Ref
a4 = A4/A_Ref
p_ratio_tt = beta_tt[2,0] #across the whole compressor
mdot = mass_flow[2,0]


#axial velocities
vx1 = Wm1[2,0]/U_Ref
vx2 = 0
# vx3 = 0 
vx4 = 0 

#azimuthal velocities
vy1 = (Wt1[2,0]+Omega*R1)/U_Ref
vy2 = (-Wt2[2,0]+Omega*R1)/U_Ref #attnetion to the sign
# vy3 = 
vy4 = Vt4[2,0]/U_Ref
 
#radial velocities
vr1 = 0
vr2 = Wm2[2,0]/U_Ref
# vr3 = 
vr4 = Vm4[2,0]/U_Ref

#static pressures [Pa]
p1 = P1[2,0]
p1_t = p1 + 0.5*rho1*((vx1*U_Ref)**2+(vy1*U_Ref)**2+(vr1*U_Ref)**2)
p2 = P2[2,0]
p2_t = p2 + 0.5*rho2*((vx2*U_Ref)**2+(vy2*U_Ref)**2+(vr2*U_Ref)**2)
# p3 = not needed for the moment
p4 = P4[2,0]
p4_t = p4 + 0.5*rho4*((vx4*U_Ref)**2+(vy4*U_Ref)**2+(vr4*U_Ref)**2)


#static density [kg/m3]
rho1 = Rho1[2,0]
rho2 = Rho2[2,0]
# rho3 = 
rho4 = Rho4[2,0]

#some average for the impeller 
rho_imp = (rho1+rho2)/2

#sto diventando scemo con le densita
psi_ts_real = (p2-(p1_t))/(U_Ref**2)
phi = mdot/(rho1*U_Ref*A1) #inlet flow coefficient for the impeller

psi_tt_ideal = (Omega*(R2*vy2*U_Ref-R1*vy1*U_Ref))/(rho_imp*U_Ref**2)
psi_ts_ideal = psi_tt_ideal - 0.5*rho2*((vx2*U_Ref)**2+(vy2*U_Ref)**2+(vr2*U_Ref)**2)/(rho_imp*U_Ref**2)

deltah_st = 

#%% construct dynamic transfer function of the system

Q = 2*np.pi*r2*vr2 #non dimensional source term at station 2
GAMMA = 2*np.pi*r2*vy2 #non dimensional circulation term at station 2
beta1 = 1
beta2 = 1
alpha4 = 1
A1 = 1
A2 = 1
s_i = 1
dLi_dTanb =1
alfa1 = 1
def centrifugal_vaneless(s, n, theta=0):
    m1 = np.linalg.inv(Trad_n(r4, r4, n, s, Q, GAMMA))
    m2 = Bvlsd_n(s, n, r2, r4, r2, Q, GAMMA)
    m3 = Bimp_n(s, n, vx1, vr2, vy1, vy2, alfa1, beta1, beta2, r1, r2, rho1, rho2, A1, A2, s_i, dLi_dTanb)
    m4 = Tax_n(x1, s, n, vx1, vy1)
    m_res = np.linalg.multi_dot([m1,m2,m3,m4])
    EC = np.array([[1,0,0]])
    IC = np.array([[0,1,0],
                   [0,0,1]])
    Y = np.concatenate((np.matmul(EC,m_res),IC))
    return np.linalg.det(Y)


domain = [-3.5,0.5,-2.5,6]
grid = [1,1]
n=np.arange(1,3)
poles = {}
plt.figure(figsize=format_fig)
for nn in n:
    poles[nn] = Shot_Gun(centrifugal_vaneless, domain, grid, n=nn, attempts=50)
    plt.plot(poles[nn].real,-poles[nn].imag, 'o',label='n '+str(nn))
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
plt.savefig(path+'/poles_iris_compressor.png')













