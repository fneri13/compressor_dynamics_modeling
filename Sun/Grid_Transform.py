#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:35:36 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Jacobian transformation matrix

N_z = 20 #number of streamwise nodes
N_r = 10 #number of spanwise nodes
z1 = np.linspace(0,1,N_z) #z-cordinate
r1 = z1**2 #hub line
r2 = 1+z1**2 #shroud line

#physical grid (Z,R)
Z = z1
R = r1
for n in range(1,N_r):
    Z = np.concatenate((Z, z1))
    R = np.concatenate((R, r1+(n/N_r)*(r2-r1)))

#computational grid (X,Y), satisfying Gauss Lobatto points
x1 = np.array(())
y1 = np.array(())
for i in range(0,N_z):
    xnew = np.cos(i*np.pi/(N_z-1)) #gauss lobatto points
    x1 = np.append(x1, xnew)
for j in range(0,N_r):
    ynew = np.cos(j*np.pi/(N_r-1)) #gauss lobatto points
    y1 = np.append(y1, ynew)

X, Y = np.meshgrid(x1,y1)
X = np.matrix.flatten(X)
X = np.flip(X)
Y = np.matrix.flatten(Y)
Y = np.flip(Y)

fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].scatter(Z,R,c='b')
ax[0].set_xlabel(r'$Z \ \ [mm]$')
ax[0].set_ylabel(r'$R \ \ [mm]$')
ax[0].set_title(r'physical grid')
ax[1].scatter(X,Y,c='r')
ax[1].set_xlabel(r'$\xi$')
ax[1].set_ylabel(r'$\eta$')
ax[1].set_title(r'computational grid')



# Define a function that computes the partial derivatives of x and y with respect to r and z
"""
d/dz = d/dxi*dxi/dz + d/deta*deta/dz
d/dr = d/dxi*dxi/dr + d/deta*deta/dr
------
so we need dxi/dz, dxi/dr, deta/dz,deta/dr
"""

def jacobian_cmp(z, r, x, y):
    """
    derivatives of the computational grid as a function of the physical.
    (z,r) = physical curvilinear grid, structured mesh of the meridional passage
    (x,y) = computational cartesian grid
    """
    dxdr = np.gradient(x, r)
    dxdz = np.gradient(x, z)
    dydr = np.gradient(y, r)
    dydz = np.gradient(y, z)
    return dxdz, dxdr, dydz, dydr

def jacobian_phs(z, r, x, y):
    """
    derivatives of the physical grid as a function of the computational
    (z,r) = physical curvilinear grid, structured mesh of the meridional passage
    (x,y) = computational cartesian grid)
    """
    dzdx = np.gradient(z, x)
    dzdy = np.gradient(z, y)
    drdx = np.gradient(r, x)
    drdy = np.gradient(r, y)
    return dzdx, dzdy, drdx, drdy

# Compute the derivatives terms at each point
dxdz, dxdr, dydz, dydr = jacobian_cmp(R, Z, X, Y)
dzdx, dzdy, drdx, drdy = jacobian_phs(R, Z, X, Y)

#check the derivatives if the are really inverse (the commented ones are zero and infinities)
fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0].scatter(dzdx,1/dxdz)
ax[0,0].set_xlabel(r'$dz/d \xi$')
ax[0,0].set_ylabel(r'$1/ (d\xi /dz)$')
# ax[0,1].scatter(dzdy,1/dydz)
# ax[0,1].set_xlabel(r'$dz/dy$')
# ax[0,1].set_ylabel(r'$1/ (dy /dz)$')
ax[1,0].scatter(drdx,1/dxdr)
ax[1,0].set_xlabel(r'$dr/dx$')
ax[1,0].set_ylabel(r'$1/(dx/dr)$')
# ax[1,1].scatter(drdy,1/dydr)
# ax[1,1].set_xlabel(r'$dr/dy')
# ax[1,1].set_ylabel(r'$1/()dy/dr$')

#check the mapping
fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0].scatter(Z,X,c='b')
ax[0,0].set_xlabel(r'$Z \ \ [mm]$')
ax[0,0].set_ylabel(r'$\xi$')
ax[0,1].scatter(R,X,c='b')
ax[0,1].set_xlabel(r'$R \ \ [mm]$')
ax[0,1].set_ylabel(r'$\xi$')
ax[1,0].scatter(Z,Y,c='b')
ax[1,0].set_xlabel(r'$Z \ \ [mm]$')
ax[1,0].set_ylabel(r'$\eta$')
ax[1,1].scatter(R,Y,c='b')
ax[1,1].set_xlabel(r'$R\ \ [mm]$')
ax[1,1].set_ylabel(r'$\eta$')

#check the derivatives
fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0].scatter(Z,dxdz,c='b')
ax[0,0].set_xlabel(r'$Z \ \ [mm]$')
ax[0,0].set_ylabel(r'$\partial \xi/ \partial z$')
ax[0,1].scatter(R,dxdr,c='b')
ax[0,1].set_xlabel(r'$R \ \ [mm]$')
ax[0,1].set_ylabel(r'$\partial \xi / \partial r$')
ax[1,0].scatter(Z,dydz,c='b')
ax[1,0].set_xlabel(r'$Z \ \ [mm]$')
ax[1,0].set_ylabel(r'$\partial \eta/ \partial z$')
ax[1,1].scatter(R,dydr,c='b')
ax[1,1].set_xlabel(r'$R\ \ [mm]$')
ax[1,1].set_ylabel(r'$\partial \eta/ \partial r$')