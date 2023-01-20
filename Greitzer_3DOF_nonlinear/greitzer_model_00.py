#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve


# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (12,8)


#%%
#Greitzer coefficients span to understand B behavior
B = np.linspace(0,10,10)
G = 2

#Greitzer coefficients calculated for a certain compressor (se aumenti B ottiene il ciclo limite)
N = 10e3                                    #[rpm] --> large N enlarges the cirle
r_ref = 100e-3                              #radius [m]
omega = 2*np.pi*N/60                        #[rad/s]
U_ref = omega*r_ref                         #reference speed of compressor[m/s]
a = 340                                     #speed of sound [m/s]
Vp = 0.0005                                      #plenum volume [m3]
dc = 10e-3                                  #inlet diameter [m]
Ac = (np.pi*dc**2)/4                        #inlet area [m2]
Lc = 0.1                                    #inlet length [m]
dt = 10e-3                                  #throttle diameter [m]
At = (np.pi*dt**2)/4                        #throttle area [m2]
Lt = 0.1                                    #throttle length [m]
B_real = (U_ref/(2*a))*np.sqrt(Vp/(Ac*Lc))  #B parameter of the described compressor
G_real = Lt*Ac/(Lc*At)                      #G parameter of the described compressor
k_valve = 1 
# G_real = 1000

#%%
#compressor points
phi_p = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
psi_p = np.array([0.5, 0.8, 0.95, 1.0, 0.8])

#polynomial interpolation for the compressor curve
z_coeff = np.polyfit(phi_p, psi_p, 3)
phi = np.linspace(0,1,100)
psi_c = np.polyval(z_coeff,phi)

#throttle valve curve
psi_v = k_valve*phi**2

#plot the compressor curve
# plt.figure(figsize=format_fig)
# plt.scatter(phi_p, psi_p)
# plt.plot(phi,psi_c,label='Compressor')
# plt.plot(phi,psi_v,'--r',label='Throttle')
# plt.ylim(0,1.5)
# plt.ylabel('$\Psi$')
# plt.xlabel('$\Phi$')
# plt.legend()

#find the intersection flow coefficient between compressor and throttle
def func_work_coefficient(phi):
    return np.polyval(z_coeff,phi) - k_valve*phi**2

phi_eq = fsolve(func_work_coefficient,0.7)
psi_eq = k_valve*phi_eq**2

#%% solve the differential system of equations (as defined by Sundstrom)

def Greitzer(y, t, B, G, k_valve, closure_coeff):
    """
    Defines the differential equations for the Greitzer model
    (as found in the thesis of Sündstrom).

    Arguments:
        y :  vector of the state variables
        t :  time
        B :  B parameter
        G :  G parameter
        
    State variables:
        x1 : compressor flow coefficient
        x2 : throttle flow coefficient
        x3 : compressor work coefficient
    """
    x1, x2, x3 = y
    dydt = [B * np.polyval(z_coeff,x1) - x3,
            (x3 - closure_coeff*k_valve*x2**2) * B/G,
            (x1-x2)/B]
    return dydt

t = np.linspace(0,100,1000)


y0 = [phi_eq[0], 
      phi_eq[0], 
      psi_eq[0]] #initial conditions
IC_x = [y0[0]]
IC_y = [y0[2]]
IC_z = [y0[1]]

from scipy.integrate import odeint
# closure_coeff = 50
closure_coeff = 1000
sol = odeint(Greitzer, y0, t, args=(B_real, G_real, k_valve, closure_coeff))
initialDerivative = Greitzer(y0, t[0], B_real, G_real, k_valve, closure_coeff)

import os
path = "pics"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")

#temporal evolution plots
fig, axes = plt.subplots(3,1, figsize=format_fig)
axes[0].set_ylabel(r'$\Phi_{c}$')
axes[0].plot(t,sol[:,0])
axes[1].set_ylabel(r'$\Phi_{t}$')
axes[1].plot(t,sol[:,1])
axes[2].set_ylabel(r'$\Psi$')
axes[2].plot(t,sol[:,2])
axes[2].set_xlabel(r'$\xi $')
fig.suptitle('Transient after valve closure')
fig.savefig(path+'/transient_valve_closure_'+str(int(closure_coeff))+'.png')

#plots in the phase space
fig, axes = plt.subplots(1,3, figsize=format_fig)
axes[0].set_ylabel(r'$\Phi_{c}$')
axes[0].set_xlabel(r'$\Phi_{t}$')
axes[0].plot(sol[:,0], sol[:,1],linewidth=0.8)
axes[0].plot(IC_z, IC_x, marker='o',color='b', label='IC')
axes[1].set_ylabel(r'$\Psi$')
axes[1].set_xlabel(r'$\Phi_{c}$')
axes[1].plot(sol[:,0],sol[:,2],linewidth=0.8)
axes[1].plot(IC_x, IC_y, marker='o',color='b', label='IC')
axes[2].set_ylabel(r'$\Psi$')
axes[2].set_xlabel(r'$\Phi_{t}$')
axes[2].plot(sol[:,1], sol[:,2],linewidth=0.8)
axes[2].plot(IC_z, IC_y, marker='o',color='b', label='IC')
fig.suptitle('Phase Space trajectories')
fig.savefig(path+'/phase_trajectories_'+str(int(closure_coeff))+'.png')


#plot the compressor curve
plt.figure(figsize=format_fig)
# plt.scatter(phi_p, psi_p, 'v')
plt.plot(phi,psi_c,label='Compressor')
plt.plot(phi,psi_v,'--k',label='Throttle Initial')
plt.plot(phi,psi_v*closure_coeff,'--r',label='Throttle Final')
plt.plot(sol[:,0],sol[:,2],linewidth=0.8, label = 'Transient')
plt.plot(IC_x, IC_y, 'ko', label='Initial condition')
plt.ylabel(r'$\Psi$')
plt.xlabel(r'$\Phi$')
plt.legend()
plt.ylim(0,2)
plt.title('Transient after throttle closure')
plt.savefig(path+'/characteristic_valve_closure_'+str(int(closure_coeff))+'.png')

 

#%% 3D trajectory in phase space

fig = plt.figure(figsize=format_fig)
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = sol[:,2]
xline = sol[:,0]
yline = sol[:,1]
ax.plot3D(xline, yline, zline, linewidth=1)
ax.set_title('Trajectory')
ax.set_xlabel(r'$\Phi_{c}$')
ax.set_ylabel(r'$\Phi_{t}$')
ax.set_zlabel(r'$\Psi$')
fig.savefig(path+'/3D_phase_trajectory_'+str(int(closure_coeff))+'.png')









