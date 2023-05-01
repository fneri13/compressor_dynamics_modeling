#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:58:06 2023

@author: fneri
read the data from paraview csv output file. Still need to understand how the geometry works, in order to do circumferential average
"""
#imports
from Compressor import Compressor

#%% Jacobian transformation matrix

# N_z = 20 #number of streamwise nodes
# N_r = 10 #number of spanwise nodes
# z1 = np.linspace(0,1,N_z) #z-cordinate
# r1 = z1**2 #hub line
# r2 = 1+z1**2 #shroud line

# #physical grid (Z,R)
# Z = z1
# R = r1
# for n in range(1,N_r):
#     Z = np.concatenate((Z, z1))
#     R = np.concatenate((R, r1+(n/N_r)*(r2-r1)))

# #computational grid (X,Y)
# x1 = np.linspace(-1,+1,N_z)
# y1 = np.ones(N_z)*(-1)
# X = x1
# Y = y1
# for n in range(1,N_r):
#     X = np.concatenate((X, x1))
#     Y = np.concatenate((Y, y1+(n/N_r)*2))

# fig, ax = plt.subplots(1,2, figsize=(12,6))
# ax[0].scatter(Z,R,c='b')
# ax[0].set_xlabel(r'$Z \ \ [mm]$')
# ax[0].set_ylabel(r'$R \ \ [mm]$')
# ax[1].scatter(X,Y,c='r')
# ax[1].set_xlabel(r'$\xi$')
# ax[1].set_ylabel(r'$\eta$')


# # Define a function that computes the partial derivatives of x and y with respect to r and z
# """
# d/dz = d/dxi*dxi/dz + d/deta*deta/dz
# d/dr = d/dxi*dxi/dr + d/deta*deta/dr
# ------
# so we need dxi/dz, dxi/dr, deta/dz,deta/dr
# """

# def jacobian_cmp(z, r, x, y):
#     """
#     derivatives of the computational grid as a function of the physical.
#     (z,r) = physical curvilinear grid, structured mesh of the meridional passage
#     (x,y) = computational cartesian grid
#     """
#     dxdr = np.gradient(x, r)
#     dxdz = np.gradient(x, z)
#     dydr = np.gradient(y, r)
#     dydz = np.gradient(y, z)
#     return dxdz, dxdr, dydz, dydr

# def jacobian_phs(z, r, x, y):
#     """
#     derivatives of the physical grid as a function of the computational
#     (z,r) = physical curvilinear grid, structured mesh of the meridional passage
#     (x,y) = computational cartesian grid)
#     """
#     dzdx = np.gradient(z, x)
#     dzdy = np.gradient(z, y)
#     drdx = np.gradient(r, x)
#     drdy = np.gradient(r, y)
#     return dzdx, dzdy, drdx, drdy

# # Compute the derivatives terms at each point
# dxdz, dxdr, dydz, dydr = jacobian_cmp(R, Z, X, Y)
# dzdx, dzdy, drdx, drdy = jacobian_phs(R, Z, X, Y)

# #check the derivatives if the are really inverse (the commented ones are zero and infinities)
# fig, ax = plt.subplots(2,2, figsize=(12,8))
# ax[0,0].scatter(dzdx,1/dxdz)
# ax[0,0].set_xlabel(r'$dz/d \xi$')
# ax[0,0].set_ylabel(r'$1/ (d\xi /dz)$')
# # ax[0,1].scatter(dzdy,1/dydz)
# # ax[0,1].set_xlabel(r'$dz/dy$')
# # ax[0,1].set_ylabel(r'$1/ (dy /dz)$')
# ax[1,0].scatter(drdx,1/dxdr)
# ax[1,0].set_xlabel(r'$dr/dx$')
# ax[1,0].set_ylabel(r'$1/(dx/dr)$')
# # ax[1,1].scatter(drdy,1/dydr)
# # ax[1,1].set_xlabel(r'$dr/dy')
# # ax[1,1].set_ylabel(r'$1/()dy/dr$')

# #check the mapping
# fig, ax = plt.subplots(2,2, figsize=(12,8))
# ax[0,0].scatter(Z,X,c='b')
# ax[0,0].set_xlabel(r'$Z \ \ [mm]$')
# ax[0,0].set_ylabel(r'$\xi$')
# ax[0,1].scatter(R,X,c='b')
# ax[0,1].set_xlabel(r'$R \ \ [mm]$')
# ax[0,1].set_ylabel(r'$\xi$')
# ax[1,0].scatter(Z,Y,c='b')
# ax[1,0].set_xlabel(r'$Z \ \ [mm]$')
# ax[1,0].set_ylabel(r'$\eta$')
# ax[1,1].scatter(R,Y,c='b')
# ax[1,1].set_xlabel(r'$R\ \ [mm]$')
# ax[1,1].set_ylabel(r'$\eta$')

# #check the derivatives
# fig, ax = plt.subplots(2,2, figsize=(12,8))
# ax[0,0].scatter(Z,dxdz,c='b')
# ax[0,0].set_xlabel(r'$Z \ \ [mm]$')
# ax[0,0].set_ylabel(r'$d\xi/dz$')
# ax[0,1].scatter(R,dxdr,c='b')
# ax[0,1].set_xlabel(r'$R \ \ [mm]$')
# ax[0,1].set_ylabel(r'$d\xi/dr$')
# ax[1,0].scatter(Z,dydz,c='b')
# ax[1,0].set_xlabel(r'$Z \ \ [mm]$')
# ax[1,0].set_ylabel(r'$d\eta/dz$')
# ax[1,1].scatter(R,dydr,c='b')
# ax[1,1].set_xlabel(r'$R\ \ [mm]$')
# ax[1,1].set_ylabel(r'$d\eta/dr$')

#%%

# Instantiate a compressor object with a compression ratio of 0.5
# my_compressor = Compressor(x,y,z,rho,p1/rho,p2/rho,p3/rho,p)

my_compressor = Compressor('data/eckardt_impeller.csv')
# my_compressor.AddDataSetZone('data/eckardt_inlet.csv')
# my_compressor.AddDataSetZone('data/eckardt_outlet.csv')


my_compressor.scatterPlot3D('theta')
my_compressor.scatterPlot3D('density')
my_compressor.scatterPlot3D('radial')
my_compressor.scatterPlot3D('tangential')
my_compressor.scatterPlot3D('axial')
my_compressor.scatterPlot3D('pressure')

my_compressor.scatterPlot3DFull(20, 20, field='pressure',slices=500)


my_compressor.UnstructuredCircumferentialAverage(50, 50)
my_compressor.scatterPlot2D('density', size=50)
my_compressor.scatterPlot2D('radial', size=50)
my_compressor.scatterPlot2D('tangential', size=50)
my_compressor.scatterPlot2D('axial', size=50)
my_compressor.scatterPlot2D('pressure', size=50)





