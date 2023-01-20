#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn
version 03 adds the cubic characteristic described by Moore and Greitzer for the unstalled
version of the machine, instead of random points
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
#Parameter set
N = 20e3                                    #[rpm] --> large N enlarges the cirle
r_ref = 100e-3                              #radius [m]
omega = 2*np.pi*N/60                        #[rad/s]
U_ref = omega*r_ref                         #reference speed of compressor[m/s]
a = 340                                     #speed of sound [m/s]
Vp = 0.0005                                 #plenum volume [m3]
dc = 10e-3                                  #inlet diameter [m]
Ac = (np.pi*dc**2)/4                        #inlet area [m2]
Lc = 0.1                                    #inlet length [m]
lc = Lc/r_ref
dt = 10e-3                                  #throttle diameter [m]
At = (np.pi*dt**2)/4                        #throttle area [m2]
Lt = 0.1                                    #throttle length [m]
# B_real =0.3*(U_ref/(2*a))*np.sqrt(Vp/(Ac*Lc))   #B parameter of the described compressor
B_real =0.54
G_real = 1*Lt*Ac/(Lc*At)                    #G parameter of the described compressor
k_valve = 4.5                               #2.85 is unstable, 2.8 is stable. depends of course on parameters
tau = 1                                     #time lag of compressor
a_lag = r_ref/(U_ref*N*tau)                 #a parameter to be used in the equations
m = 2                                     #m parameter of duct (Moore)

#%%
# #compressor points if you want to use interpolation instead of cubic
# phi_p = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
# psi_p = np.array([0.5, 0.8, 0.95, 1.0, 0.8])

# #polynomial interpolation for the compressor curve
# z_coeff = np.polyfit(phi_p, psi_p, 3)

#domain of flow coefficient
phi = np.linspace(0,1,100)

#calculation of unstalled cubic curve 
H = 0.35/2
W = 0.2
psi_c_0=0.3
def unstalled_characteristic(phi):
    """
    It computes the unstalled characteristic of the compressor using the cubic model

    Arguments:
        phi :  flow coefficient
        H : H parameter
        W : W parameter
        psi_c_0 : performance at zero flow coefficient
    """
    return psi_c_0 + H * (1 + 1.5*(phi/W - 1) - 0.5*(phi/W -1)**3)


psi_c = unstalled_characteristic(phi)

#throttle valve curve, only specified from k_valve
psi_v = k_valve*phi**2

#find the intersection flow coefficient between compressor and throttle
def func_work_coefficient(phi):
    """
    Function needed by fsolve in order to find the intersection between unstalled
    compressor curve and throttle line
    """
    return unstalled_characteristic(phi) - k_valve*phi**2

initial_phi_guess = 0.3 #initial guess of intersection
phi_eq = fsolve(func_work_coefficient,initial_phi_guess) #intersection, operating point
psi_eq = k_valve*phi_eq**2 #intersection, operating point

#calculate derivatives of the compressor and throttle characteristic at equilibrium point
delta_phi = phi_eq*0.001
phi_left = phi_eq - delta_phi
phi_right = phi_eq + delta_phi
psi_c_right = unstalled_characteristic(phi_right)
psi_c_left = unstalled_characteristic(phi_left)
psi_c_prime = (psi_c_right - psi_c_left) / (2*delta_phi)
psi_v_prime = 2*k_valve*phi_eq

#assess the stability condition based on the eigenvalues of 2DOF system
gmma_param = 1/(psi_v_prime*B_real**2)
print('Assessment of the stability condition based on the eigenvalues of the reduced 2DOF system:')
print('The value of compressor slope is %.2f' %psi_c_prime[0])
print('The stability limit is %.2f' %gmma_param[0])
stability_condition = psi_c_prime < gmma_param
if stability_condition:
    print('The 2D model predicts stability')
else:
    print('The 2D model predicts instability')
# #%% solve the differential system of equations (as defined by Sundstrom)
# def Greitzer(y, xi, B, G, k_valve, closure_coeff):
#     """
#     Defines the differential equations for the Greitzer model
#     (as found in the thesis of Sündstrom).

#     Arguments:
#         y :  vector of the state variables
#         xi : non dimensional time
#         B :  B parameter
#         G :  G parameter
        
#     State variables:
#         x1 : compressor flow coefficient
#         x2 : throttle flow coefficient
#         x3 : compressor work coefficient
#     """
#     x1, x2, x3 = y
#     dydt = [B * (unstalled_characteristic(x1) - x3),
#             (x3 - closure_coeff*k_valve*x2**2) * B/G,
#             (x1-x2)/B]
#             # (x1-x2)/B + np.cos(omega*xi)] #non-authonomous version of equations
#     return dydt

# #time span
# t = np.linspace(0,1,5000)
# xi = t*a*np.sqrt(Ac/(Vp*Lc))

# #initial conditions
# throttle_closure = 0.01 #perturbation to replicate linear analysis
# y0 = [phi_eq[0]*(1-throttle_closure), 
#       phi_eq[0]*(1+throttle_closure), 
#       psi_eq[0]*(1-0.5*throttle_closure)] 

# IC_flow_c = [y0[0]]
# IC_flow_t = [y0[1]]
# IC_psi = [y0[2]]

# from scipy.integrate import odeint
# closure_coeff = 1.0
# sol = odeint(Greitzer, y0, xi, args=(B_real, G_real, k_valve, closure_coeff))
# initialDerivative = Greitzer(y0, t[0], B_real, G_real, k_valve, closure_coeff)


# #%%make the plots
# path = "pics"
# # Check whether the specified path exists or not
# isExist = os.path.exists(path)
# if not isExist:
#    # Create a new directory because it does not exist
#    os.makedirs(path)
#    print("The new directory is created!")

# #temporal evolution plots
# fig, axes = plt.subplots(3,1, figsize=format_fig)
# axes[0].set_ylabel(r'$\Phi_{c}$')
# axes[0].plot(xi,sol[:,0])
# axes[1].set_ylabel(r'$\Phi_{t}$')
# axes[1].plot(xi,sol[:,1])
# axes[2].set_ylabel(r'$\Psi$')
# axes[2].plot(xi,sol[:,2])
# axes[2].set_xlabel(r'$\xi $')
# fig.suptitle('Transient after initial perturbation')
# fig.savefig(path+'/transient_after_perturbation.png')

# #plots in the phase space
# fig, axes = plt.subplots(1,3, figsize=format_fig)
# axes[0].set_xlabel(r'$\Phi_{c}$')
# axes[0].set_ylabel(r'$\Phi_{t}$')
# axes[0].plot(sol[:,0], sol[:,1],linewidth=0.8)
# axes[0].plot(IC_flow_c, IC_flow_t, marker='o',color='b', label='IC')
# axes[1].set_ylabel(r'$\Psi$')
# axes[1].set_xlabel(r'$\Phi_{c}$')
# axes[1].plot(sol[:,0],sol[:,2],linewidth=0.8)
# axes[1].plot(IC_flow_c, IC_psi, marker='o',color='b', label='IC')
# axes[2].set_ylabel(r'$\Psi$')
# axes[2].set_xlabel(r'$\Phi_{t}$')
# axes[2].plot(sol[:,1], sol[:,2],linewidth=0.8)
# axes[2].plot(IC_flow_t, IC_psi, marker='o',color='b', label='IC')
# fig.suptitle('Phase Space trajectories')
# fig.savefig(path+'/phase_trajectories_perturbations.png')

# #plot the compressor curve
# plt.figure(figsize=format_fig)
# plt.plot(phi,psi_c,linewidth=0.6,label='Compressor line')
# plt.plot(phi,psi_v,linewidth=0.6,label='Throttle line')
# plt.plot(sol[:,0],sol[:,2],'--k',linewidth=0.8, label = 'Transient')
# plt.plot(IC_flow_c, IC_psi, 'ko', label='Initial condition')
# plt.ylabel(r'$\Psi$')
# plt.xlabel(r'$\Phi_c$')
# plt.legend()
# plt.xlim(-0.3,0.8)
# plt.ylim(0,0.8)
# plt.title('Transient after perturbation')
# plt.savefig(path+'/characteristic_valve_closure.png')

# #3D trajectory in phase space
# xline = sol[:,0] #compressor flow coeff
# yline = sol[:,1] #throttle flow coeff
# zline = sol[:,2] #work coeff.
# fig = plt.figure(figsize=format_fig)
# ax = plt.axes(projection='3d')
# ax.plot3D(xline, yline, zline, linewidth=1)
# ax.set_title('Trajectory')
# ax.set_xlabel(r'$\Phi_{c}$')
# ax.set_ylabel(r'$\Phi_{t}$')
# ax.set_zlabel(r'$\Psi$')
# fig.savefig(path+'/3D_phase_trajectory.png')


#%% solve the Greitzer-Moore model for post stall transient
def Moore_Greitzer(y, xi, B, k_valve, W, H, psi_c_0, a, m, lc):
    """
    Defines the differential equations for the Greitzer model
    (as found in the thesis of Sündstrom).

    Arguments:
        y :  vector of the state variables
        xi : non dimensional time
        B :  B parameter
        k_valve :  throttle line coefficient
        W : W parameter of cubic shape of compressor 
        H : H parameter of cubic shape of compressor
        psi_c_0 : work coefficient at zero flow rate parameter of cubic shape
        a : reciprocal time lag
        m : duct parameter, between 1 and 2
        lc : non dimensional compressor length
        
    State variables:
        x1 : total to static work coefficient \Psi
        x2 : azimuthally averaged flow coefficient \Phi
        x3 : squared amplitude of rotating stall cell J
    """
    x1, x2, x3 = y
    dydt = [ W/(4*lc*B**2) * (x2/W - (1/W)*np.sqrt(x1/k_valve)) ,
            (H/lc)*( -(x1-psi_c_0)/H +1 + 1.5*(x2/W-1)*(1-0.5*x3)-0.5*(x2/W-1)**3 ),
            x3*(1-(x2/W-1)**2 -0.25*x3)*(3*a*H)/(W*(1+m*a))]
    return dydt

#time span
t = np.linspace(0,3,10000)
xi = t*a*np.sqrt(Ac/(Vp*Lc))

#initial conditions
throttle_closure = 0.01 #perturbation to replicate linear analysis
y0 = [psi_eq[0]*(1-throttle_closure), 
      phi_eq[0]*(1+throttle_closure), 
      throttle_closure*1] 

IC_psi_tot = [y0[0]]
IC_flow_tot = [y0[1]]
IC_ampl = [y0[2]]

from scipy.integrate import odeint
sol = odeint(Moore_Greitzer, y0, xi, args=(B_real, k_valve, W, H, psi_c_0, a_lag, m, lc))


#PLOTS
path = "pics"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")

#temporal evolution plots
fig, axes = plt.subplots(3,1, figsize=format_fig)
axes[0].set_ylabel(r'$\Psi / \Psi_{0}$')
axes[0].plot(xi,sol[:,0]/sol[0,0])
axes[1].set_ylabel(r'$\Phi / \Phi_{0}$')
axes[1].plot(xi,sol[:,1]/sol[0,1])
axes[2].set_ylabel(r'$A/A_0$')
axes[2].plot(xi,np.sqrt(sol[:,2]/sol[0,2]))
axes[2].set_xlabel(r'$\xi $')
fig.suptitle('Post Stall Transient')
fig.savefig(path+'/post_stall_transient.png')

#plot the compressor curve
plt.figure(figsize=format_fig)
plt.plot(phi,psi_c,linewidth=0.6,label='Compressor line')
plt.plot(phi,psi_v,linewidth=0.6,label='Throttle line')
plt.plot(sol[:,1],sol[:,0],'--k',linewidth=0.8, label = 'Transient')
plt.plot(IC_flow_tot, IC_psi_tot, 'ko', label='Initial condition')
plt.ylabel(r'$\Psi$')
plt.xlabel(r'$\Phi_c$')
plt.legend()
plt.xlim(-0.3,0.8)
plt.ylim(0,0.8)
plt.title('Transient after perturbation')
plt.savefig(path+'/post_stall_evolution.png')

#%% Linearizer version analysis
from numpy import linalg as LA
B_span = np.linspace(0.01,5,100)
stability = np.zeros(len(B_span))
for i in range(len(B_span)):
    B_real = B_span[i]
    #construct the perturbation matrix
    A = np.zeros((3,3))
    phi_t_prime_eq = 1/(2*np.sqrt(k_valve*psi_eq[0]))
    J_eq = 4*(1-(phi_eq[0]/W-1)**2) #check if positive 
    A[0,0] = - phi_t_prime_eq/(4*B_real**2*lc)
    A[0,1] = W/(4*B_real**2*lc)
    A[0,2] = 0.0
    A[1,0] = -1/lc
    A[1,1] = (3*H/(2*W*lc))*((1-0.5*J_eq)-(phi_eq[0]-1)**2)
    A[1,2] = -3*H*(phi_eq[0]/W-1)/(4*lc)
    A[2,0] = 0.0
    A[2,1] = 6*a_lag*H*J_eq*(phi_eq[0]/W-1)/((1+m*a_lag)*W**2)
    A[2,2] = 3*a_lag*H*(1-(phi_eq[0]/W-1)**2-0.5*J_eq)/((1+m*a_lag)*W**2)
    
    #eigenvalues of the system
    w, v = LA.eig(A)
    
    # Find the roots of the polynomial
    roots_real = w.real
    if (roots_real[0]>=0 or roots_real[1]>=0 or roots_real[2]>=0):
        stability[i] = 1.0 #unstable
    else:
        stability[i] = -1.0 #stable
    
    #print the value of transition from stability to instability
    if (stability[i-1]*stability[i]<=0 and i>0):
        print('The critical B is %.3f' %B_span[i])
        
plt.figure()
plt.plot(B_span,stability)
plt.xlabel(r'$B$')
plt.ylabel('Stability')

#in realtà viene leggermente diverso dal lineare. 






















