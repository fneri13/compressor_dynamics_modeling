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
format_fig = (9,7)

#%% Relevant geometric parameters for the compressor selected by Andrea on the Pareto front. All the variables that begin with capital
# letters are dimensional. Otherwise they have been non-dimensionalized

blades = 14 #number of blades (normal +split)
main_blades = 7 #main blades
splitter_blades = 7 #splitter blades
n_max = blades//4+1 #maximum harmonic to take in consideration (2 blades in a half-wavelength)

#COMPRESSOR DESIGN SELECTED ALONG THE PARETO FRONT (SI units) - INPUT DATA
fluid = 'R1233zd(E)' #fluid
Omega_range = np.array([70,79,88,93,97])*1e3*2*np.pi/60 #angular velocities [rad/s]
R1s = 15.2*1e-3 #shroud radius inlet [m]
R1h = 3.4*1e-3 #hub radius inlet [m]
R2 = 22.8*1e-3 #impeller exit radius [m]
R3 = 35.2*1e-3 #diffuser outlet radius [m]
H2 = 2.3*1e-3 #blade heigth exit impeller [m]
H3 = 1.6*1e-3 #diffuser height [m]
H4 = H3
Lax = 16*1e-3 #axial length [m]
R4 = 49.3*1e-3 # external diameter compressor [m]
Ts = 0.3*1e-3 #blade trailing edge at shroud [m]
Th = 0.6*1e-3 #blade trailing edge at hub [m]
Tte_mean = 0.5*(Ts+Th)
R1 = (R1s+R1h)/2 #radius at impeller inlet [m]

#Reference parameters for non-dimensionalization
R_Ref = R2

#STATION LOCATIONS, non dimensionalized
x1 = 0
x2 = Lax/R_Ref
# x3 = x2
x4 = x2
r1 = R1/R_Ref
r2 = R2/R_Ref
r3 = R3/R_Ref
# r4 = R4/R_Ref


#%%IMPORT DATA FROM DATA FOLDER (IRIS COMPRESSOR ANDREA)
# note: my STA numbers are shifted (1 mine = 0 Andrea, 2 mine = 1 Andrea, 4 mine = 2 Andrea)
import pickle
data_folder = "../data/IRIS_single_stage/design0_beta_3.450/operating_map/" #path to folder

#in the following arrays the zeros means that the working point was out of range (choked)
#every row is a different speedline, given in the RPM vector
#every column is a different mass flow rate

with open(data_folder + 'alpha2.pkl', 'rb') as f:
    Alpha4 = pickle.load(f) 
with open(data_folder + 'beta0.pkl', 'rb') as f:
    Beta1 = pickle.load(f)
with open(data_folder + 'beta1.pkl', 'rb') as f:
    Beta2 = pickle.load(f)
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





#%% WORKING CONDITIONS, DEPENDENT ON THE SPEED LINE

speedline = 2 #index of the speedline to be used
index_max = np.where(mass_flow[speedline,:] == 0)
index_max = index_max[0]
index_max = index_max[0]
index_max = index_max-1 #index max in order to not consider the zeros

Omega = rpm[speedline]*2*np.pi/60
U_Ref = Omega*R_Ref #the reference velocity is the outlet impeller peripheral speed
A_Ref = R_Ref**2 
A1 = np.pi*(R1s**2 - R1h**2)
A2 = 2*np.pi*R2*H2
A4 = 2*np.pi*R4*H4
p_ratio_tt = beta_tt[speedline,0:index_max] #across the whole compressor
mdot = mass_flow[speedline,0:index_max] 


#axial velocities
vx1 = Wm1[speedline,0:index_max]/U_Ref
vx2 = np.zeros(index_max)
# vx3 = 0 
vx4 = np.zeros(index_max)

#azimuthal velocities
vy1 = np.zeros(index_max)
vy2 = (-Wt2[speedline,0:index_max]+Omega*R1)/U_Ref #attnetion to the sign
# vy3 = 
vy4 = Vt4[speedline,0:index_max]/U_Ref
 
#radial velocities
vr1 = np.zeros(index_max)
vr2 = Wm2[speedline,0:index_max]/U_Ref
# vr3 = 
vr4 = Vm4[speedline,0:index_max]/U_Ref

alpha1 = np.arctan(vy1/vx1)

#static pressures [Pa]
p1 = P1[speedline,0:index_max]
p1_t = p1 + 0.5*Rho1[speedline,0:index_max]*((vx1*U_Ref)**2+(vy1*U_Ref)**2+(vr1*U_Ref)**2)
# plt.figure()
# plt.plot(p1_t)
# plt.plot(p1)
# plt.figure()
# plt.plot(p2_t)
# plt.plot(p2)
# plt.figure()
# plt.plot(p4_t)
# plt.plot(p4)
p2 = P2[speedline,0:index_max]
p2_t = p2 + 0.5*Rho2[speedline,0:index_max]*((vx2*U_Ref)**2+(vy2*U_Ref)**2+(vr2*U_Ref)**2)
# p3 = not needed for the moment
p4 = P4[speedline,0:index_max]
p4_t = p4 + 0.5*Rho4[speedline,0:index_max]*((vx4*U_Ref)**2+(vy4*U_Ref)**2+(vr4*U_Ref)**2)


#static density [kg/m3]
rho1 = Rho1[speedline,0:index_max]
rho2 = Rho2[speedline,0:index_max]
# rho3 = 
rho4 = Rho4[speedline,0:index_max]

#some average for the impeller 
rho_imp = (rho1+rho2)/2
plt.figure()
plt.plot(rho_imp)

delta_htt_real = Omega*(R2*vy2*U_Ref - R1*vy1*U_Ref)
delta_htt_ideal = delta_htt_real/eta_tt[speedline,0:index_max]
psi_tt_real = delta_htt_real/U_Ref**2
psi_tt_ideal = delta_htt_ideal/U_Ref**2
delta_hts_real = Omega*(R2*vy2*U_Ref - R1*vy1*U_Ref) - 0.5*(vy2*U_Ref)**2
delta_hts_ideal = delta_hts_real/eta_ts[speedline,0:index_max]
L_imp = (delta_hts_ideal - delta_hts_real)/U_Ref**2
phi = mdot/(rho1*U_Ref*A1) #inlet flow coefficient for the impeller
dLimp_dphi = np.gradient(L_imp, phi)
dLi_dTanb = np.gradient(L_imp, np.tan(Beta1[speedline,0:index_max]))
# plt.figure()
# plt.plot(psi_tt_ideal)
# plt.plot(psi_tt_real)



plt.figure(figsize=format_fig)
plt.plot(phi,L_imp)
plt.ylabel(r'$L_{imp}$')
plt.xlabel(r'$\phi$')
plt.title('impeller loss')
plt.figure(figsize=format_fig)
plt.plot(phi,dLimp_dphi)
plt.ylabel(r'$dL_{imp} / d \phi$')
plt.xlabel(r'$\phi$')
plt.title('impeller loss derivative')
plt.figure(figsize=format_fig)
plt.plot(np.tan(Beta1[speedline,0:index_max]),dLi_dTanb)
plt.ylabel(r'$dL_{imp} / d \tan{\beta_1}$')
plt.xlabel(r'$\beta_1$')
plt.title('impeller loss derivative')


#%% construct dynamic transfer function of the system

Q = 2*np.pi*r2*vr2 #non dimensional source term at station 2
GAMMA = 2*np.pi*r2*vy2 #non dimensional circulation term at station 2
beta1 = Beta1[speedline,0:index_max]
beta2 = Beta2[speedline,0:index_max]
alpha4 = Alpha4[speedline,0:index_max]
A1_blade = A1/main_blades
A2_blade = A2/(main_blades+splitter_blades)
s_i = np.sqrt(Lax**2+(R2-R1)**2)*1.3 #approximation of the meridional path length in the impeller

wpoint = index_max//4

def centrifugal_vaneless(s, n, theta=0):
    m1 = np.linalg.inv(Trad_n(r4, r4, n, s, Q[wpoint], GAMMA[wpoint]))
    m2 = Bvlsd_n(s, n, r2, r4, r2, Q[wpoint], GAMMA[wpoint])
    m3 = Bimp_n(s, n, vx1[wpoint], vr2[wpoint], vy1[wpoint], vy2[wpoint], alpha1[wpoint], beta1[wpoint], 
                beta2[wpoint], r1, r2, rho1[wpoint], rho2[wpoint], A1_blade, A2_blade, s_i, dLi_dTanb[wpoint])
    m4 = Tax_n(x1, s, n, vx1[0], vy1[0])
    m_res = np.linalg.multi_dot([m1,m2,m3,m4])
    EC = np.array([[1,0,0]])
    IC = np.array([[0,1,0],
                   [0,0,1]])
    Y = np.concatenate((np.matmul(EC,m_res),IC))
    return np.linalg.det(Y)


domain = [-5,5,-10,10]
grid = [1,1]
n=np.arange(1,5)
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
plt.savefig('pics/poles_iris_compressor.png')













