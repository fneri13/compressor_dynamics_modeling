#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
import sys
sys.path.insert(1, '../../src/') #to add Classes folder

# #input data of the problem
# r1 = 0.1826
# r2 = 0.2487
# M = 0.015
# p = 100e3
# T = 288
# L = 0.08
# R = 287
# gmma = 1.4
# rho = p/(R*T)
# a = np.sqrt(gmma*p/rho)

# #radial cordinate array span
# r = np.linspace(r1,r2,250)

# #import Bessel functions
# x = np.linspace(0,100,1000)
# J1 = jv(1, x) #BF of the first kind of the first order
# N1 = yv(1, x) #BF of the second kind of the first order
# J1d = np.gradient(J1, x)
# N1d = np.gradient(N1, x)

# fig, ax = plt.subplots(2,2,figsize=(14,8))
# ax[0,0].plot(x, J1, 'b')
# ax[0,0].set_title(r'$J_{1}$')
# # ax[0,0].set_xlabel(r'$x$')
# ax[0,0].set_ylabel(r'$J_{1}$')
# ax[0,0].set_ylim([-1,1])
# ax[0,1].plot(x, N1, 'r')
# ax[0,1].set_title(r'$Y_{1}$')
# # ax[0,1].set_xlabel(r'$x$')
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

# #%% #analytical eigenvalues
# lambda_mn = np.linspace(0,300,250) #we will do a loop for every possible value of lambda, but now let's evalute the determinant
# m = 1 
# det = np.array(())
# for i in range(0,len(lambda_mn)):
#     Jm = jv(m, lambda_mn[i]*r)
#     Nm = yv(m, lambda_mn[i]*r)
#     # dJmdr = np.gradient(Jm, r)
#     # dNmdr = np.gradient(Nm, r)
#     dJmdr = jvp(m, lambda_mn[i]*r)
#     dNmdr = yvp(m, lambda_mn[i]*r)
#     det = np.append(det,dJmdr[0]*dNmdr[-1]-dNmdr[0]*dJmdr[-1])
# zeros = np.zeros(len(det))

# fig, ax = plt.subplots(figsize=(10,7))
# ax.plot(lambda_mn, det)
# ax.plot(lambda_mn, zeros,'--k', lw=0.5)
# ax.set_title('determinant value')
# ax.set_xlabel(r'$\lambda_{mn}$')
# ax.set_ylim([-0.25,0.25])
# ax.set_ylabel(r'$\det{A}$')
# ax.legend()

# #something is not clear
# omega = a*np.sqrt(((1-M**2)*m*np.pi/L)**2 + (1-M**2)*lambda_mn**2) #alpha is the number of the iegenvalue
# fig, ax = plt.subplots(figsize=(10,7))
# ax.plot(omega, np.abs(det))
# ax.set_title('determinant value')
# ax.set_xlabel(r'$f \ [Hz]$')
# ax.set_ylim([0,0.25])
# ax.set_ylabel(r'$\det{A}$')
# ax.legend()

# #%%
# from Grid import Node, AnnulusDuctGrid
# #debug
# Nz = 100
# Nr = 60
# duct = AnnulusDuctGrid(r1, r2, L, Nz, Nr)

# density = np.random.rand(Nz, Nr)
# axialVel = np.random.rand(Nz, Nr)
# radialVel = np.random.rand(Nz, Nr)
# tangentialVel = np.random.rand(Nz, Nr)
# pressure = np.random.rand(Nz, Nr)
# for ii in range(0,Nz):
#     for jj in range(0,Nr):
#         density[ii,jj] = rho
#         axialVel[ii,jj] = M*a
#         radialVel[ii,jj] = 0
#         tangentialVel[ii,jj] = 0
#         pressure[ii,jj] = p
        
# duct.AddDensityField(density)
# duct.AddVelocityField(axialVel, radialVel, tangentialVel)
# duct.AddPressureField(pressure)
# duct.ContourPlotDensity()
# duct.ContourPlotVelocity(1)
# duct.ContourPlotVelocity(2)
# duct.ContourPlotVelocity(3)
# duct.ContourPlotPressure()

#%%
import matplotlib.pyplot as plt
import numpy as np
from SunModel import SunModel
from Grid import AnnulusDuctGrid

#input data
r1 = 0.1826
r2 = 0.2487
M = 0.015
p = 100e3
T = 288
L = 0.08
R = 287
gmma = 1.4
rho = p/(R*T)
a = np.sqrt(gmma*p/rho)

#debug
Nz = 5
Nr = 5
duct = AnnulusDuctGrid(0, L, r1, r2, Nz, Nr)

#implement a constant uniform flow in the annulus duct
density = np.random.rand(Nz, Nr)
axialVel = np.random.rand(Nz, Nr)
radialVel = np.random.rand(Nz, Nr)
tangentialVel = np.random.rand(Nz, Nr)
pressure = np.random.rand(Nz, Nr)
for ii in range(0,Nz):
    for jj in range(0,Nr):
        density[ii,jj] = rho
        axialVel[ii,jj] = M*a  
        radialVel[ii,jj] = 0
        tangentialVel[ii,jj] = 0
        pressure[ii,jj] = p
        

duct.AddDensityField(density)
duct.AddVelocityField(axialVel, radialVel, tangentialVel)
duct.AddPressureField(pressure)
duct.ContourPlotDensity()
duct.ContourPlotVelocity(1)
duct.ContourPlotVelocity(2)
duct.ContourPlotVelocity(3)
duct.ContourPlotPressure()


sunObj = SunModel(duct)
sunObj.ShowPhysicalGrid()
sunObj.ComputeSpectralGrid()
sunObj.ShowSpectralGrid()
sunObj.ComputeJacobianSpectral()
sunObj.ComputeJacobianPhysical()

sunObj.ShowJacobianPhysicalAxis()
# sunObj.ShowJacobianSpectralAxis()
sunObj.CreateAllPhysicalMatrices()
sunObj.ComputeHatMatrices()
sunObj.CreateAMatrixCoefficients()

#%% time prediction for SVD computation
# import time
# from SunModel import SunModel

# start_time = time.time()
# sunObj.ComputeSVD(omega_domain=[-10,10,-10,10], grid_omega=[50,50])
# end_time = time.time()
# print('time %.2f s' %(end_time-start_time))
# sunObj.PlotInverseConditionNumber()









