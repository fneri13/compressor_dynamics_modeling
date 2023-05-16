#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
from src.compressor import Compressor
from src.grid import DataGrid
from src.sun_model import SunModel

#input data of the problem
r1 = 0.1826                             #inner radius [m]
r2 = 0.2487                             #outer radius [m]
M = 0.015                               #Mach number
p = 100e3                               #pressure [Pa]
T = 288                                 #temperature [K]
L = 0.08                                #length [m]
R = 287                                 #air gas constant [kJ/kgK]
gmma = 1.4                              #cp/cv ratio of air
rho = p/(R*T)                           #density [kg/m3]
a = np.sqrt(gmma*p/rho)                 #ideal speed of sound [m/s]

#%% ANALYTICAL PART OF THE PROBLEM

#radial cordinate array span
r = np.linspace(r1,r2,300)

#import Bessel functions
# x = np.linspace(0,100,1000)
# J1 = jv(1, x) #BF of the first kind of the first order
# N1 = yv(1, x) #BF of the second kind of the first order
# J1d = np.gradient(J1, x)
# N1d = np.gradient(N1, x)

# fig, ax = plt.subplots(2,2,figsize=(14,8))
# ax[0,0].plot(x, J1, 'b')
# ax[0,0].set_title(r'$J_{1}$')
# ax[0,0].set_ylabel(r'$J_{1}$')
# ax[0,0].set_ylim([-1,1])
# ax[0,1].plot(x, N1, 'r')
# ax[0,1].set_title(r'$Y_{1}$')
# ax[0,1].set_ylabel(r'$Y_{1}$')
# ax[0,1].set_ylim([-1,1])
# ax[1,0].plot(x, J1d, 'b')
# ax[1,0].set_title(r'$dJ_{1}/dx$')
# ax[1,0].set_xlabel(r'$x$')
# ax[1,0].set_ylabel(r'$dJ_{1}/dx$')
# ax[1,0].set_ylim([-1,1])
# ax[1,1].plot(x, N1d, 'r')
# ax[1,1].set_title(r'$dY_{1}/dx$')
# ax[1,1].set_xlabel(r'$x$')
# ax[1,1].set_ylabel(r'$dY_{1}/dx$')
# ax[1,1].set_ylim([-1,1])
# fig.suptitle("Bessel Functions")

# #analytical eigenvalues
# lambda_mn_span = np.linspace(0,300,300) #we will do a loop for every possible value of lambda
# m = 1 
# det = np.array(())
# det2 = np.array(())
# det3 = np.array(())

# for s in range(0,len(lambda_mn_span)):
#     lambda_mn = lambda_mn_span[s]
#     J1 = jvp(m, lambda_mn*r, n=0)
#     N1 = yvp(m, lambda_mn*r, n=0)
#     dJ1dr = jvp(m, lambda_mn*r, n=1)
#     dN1dr = yvp(m, lambda_mn*r, n=1)
#     det = np.append(det, dJ1dr[0]*dN1dr[-1]-dN1dr[0]*dJ1dr[-1])
    
#     J2 = jvp(m+1, lambda_mn*r, n=0)
#     N2 = yvp(m+1, lambda_mn*r, n=0)
#     dJ2dr = jvp(m+1, lambda_mn*r, n=1)
#     dN2dr = yvp(m+1, lambda_mn*r, n=1)
#     det2 = np.append(det2, dJ2dr[0]*dN2dr[-1]-dN2dr[0]*dJ2dr[-1])
    
#     J3 = jvp(m+2, lambda_mn*r, n=0)
#     N3 = yvp(m+2, lambda_mn*r, n=0)
#     dJ3dr = jvp(m+2, lambda_mn*r, n=1)
#     dN3dr = yvp(m+2, lambda_mn*r, n=1)
#     det3 = np.append(det3, dJ3dr[0]*dN3dr[-1]-dN3dr[0]*dJ3dr[-1])
# zeros = np.zeros(len(det))

# #plot the determinant value
# fig, ax = plt.subplots(figsize=(10,7))
# ax.plot(lambda_mn_span, det, label=r'$m=1$')
# ax.plot(lambda_mn_span, det2, label=r'$m=2$')
# ax.plot(lambda_mn_span, det3, label=r'$m=3$')
# ax.plot(lambda_mn_span, zeros,'--k', lw=0.5)
# ax.set_title('determinant value')
# ax.set_xlabel(r'$\lambda_{mn}$')
# ax.set_ylim([-0.25,0.25])
# ax.set_ylabel(r'$\det{A}$')
# ax.legend()

#this is valid only for the first frequency, then you need to modify omega evaluation
# omega = a*np.sqrt(((1-M**2)*m*np.pi/L)**2 + (1-M**2)*lambda_mn**2) #alpha is the number of the eigenvalue
# fig, ax = plt.subplots(figsize=(10,7))
# ax.plot(omega, np.abs(det))
# ax.set_title('determinant value')
# ax.set_xlabel(r'$f \ [Hz]$')
# ax.set_ylim([0,0.25])
# ax.set_ylabel(r'$\det{A}$')
# ax.legend()



#%%COMPUTATIONAL PART

#number of grid nodes in the computational domain
Nz = 10
Nr = 10

#implement a constant uniform flow in the annulus duct
density = np.random.rand(Nz, Nr)
axialVel = np.random.rand(Nz, Nr)
radialVel = np.random.rand(Nz, Nr)
tangentialVel = np.random.rand(Nz, Nr)
pressure = np.random.rand(Nz, Nr)
for ii in range(0,Nz):
    for jj in range(0,Nr):
        #there could be a need for normalizing the data? (or normalizing the NS equations directly)
        density[ii,jj] = rho
        # density[ii,jj] = 1
        axialVel[ii,jj] = M*a  
        # axialVel[ii,jj] = 1  
        radialVel[ii,jj] = 0
        tangentialVel[ii,jj] = 0
        pressure[ii,jj] = p
        # pressure[ii,jj] = 1

#build a grid object
duct = DataGrid(0, L, r1, r2, Nz, Nr, density, axialVel, radialVel, tangentialVel, pressure)


#general workflow
sunObj = SunModel(duct)
sunObj.ShowPhysicalGrid(save_filename='physical_grid_%1.d_%1.d' %(Nz,Nr))
sunObj.ComputeSpectralGrid()
sunObj.ShowSpectralGrid(save_filename='spectral_grid_%1.d_%1.d' %(Nz,Nr))
sunObj.ComputeJacobianSpectral(refinement=10)
sunObj.AddAMatrixToNodes()
sunObj.AddBMatrixToNodes()
sunObj.AddCMatrixToNodes()
sunObj.AddEMatrixToNodes()
sunObj.AddRMatrixToNodes()
sunObj.AddSMatrixToNodes()
sunObj.AddHatMatricesToNodes()
sunObj.ApplySpectralDifferentiation()

sunObj.ComputeSVD(omega_domain=[7.5e3, 35e3, -8e3, 8e3], grid_omega=[100,75])
sunObj.PlotInverseConditionNumber('chi_2D_map_%1.d_%1.d' %(Nz,Nr))



plt.figure()
plt.plot(sunObj.omegaR,sunObj.chi[:,1])



























 