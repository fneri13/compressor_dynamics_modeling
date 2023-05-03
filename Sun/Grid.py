#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:38:38 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    """
    Class of Nodes, contaning cordinates and boundary marker
    """
    def __init__(self, z, r, marker):
        self.z = z #axis of the machine
        self.r = r #axis of the machine
        self.marker = marker
        
    def AddDensity(self, rho):
        self.rho = rho #axis of the machine
    
    def AddVelocity(self, vz, vr, vtheta):
        self.vz = vz
        self.vr = vr
        self.vtheta = vtheta
    
    def AddPressure(self, p):
        self.p = p
    
    def PrintInfo(self, datafile='terminal'):
        if datafile == 'terminal':
            print('marker: ' + self.marker)
            print('r: %.2f' %(self.r))
            print('z: %.2f' %(self.z))
            print('rho: %.2f' %(self.rho))
            print('V: [%.2f , %.2f, %.2f]' %(self.vz, self.vr, self.vtheta))
            print('p: %.2f' %(self.p))
            print('-----------------------------------------------')
        else: 
            with open(datafile, 'a') as f:
                print('marker: ' + self.marker, file=f)
                print('r: %.2f' %(self.r), file=f)
                print('z: %.2f' %(self.z), file=f)
                print('rho: %.2f' %(self.rho), file=f)
                print('V: [%.2f , %.2f, %.2f]' %(self.vz, self.vr, self.vtheta), file=f)
                print('p: %.2f' %(self.p), file=f)
                print('-----------------------------------------------', file=f)
        
        

class AnnulusDuctGrid:
    """
    Class of Grid for cylindrical duct
    """
    def __init__(self, rmin, rmax, L, Nz, Nr):
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        z = np.linspace(0,L,Nz)
        r = np.linspace(rmin, rmax, Nr)
        self.r_grid, self.z_grid = np.meshgrid(r,z)
        self.grid = np.empty((Nz, Nr), dtype=Node)
        for ii in range(0,Nz):
            for jj in range(0,Nr):
                if ii==0:
                    self.grid[ii,jj] = Node(z[ii],r[jj],'inlet')
                elif ii==Nz-1:
                    self.grid[ii,jj] = Node(z[ii],r[jj],'outlet')
                elif jj==0 and ii!=0 and ii!=Nz-1:
                    self.grid[ii,jj] = Node(z[ii],r[jj],'hub')
                elif jj==Nr-1 and ii!=0 and ii!=Nz-1:
                    self.grid[ii,jj] = Node(z[ii],r[jj],'shroud')
                elif ii!=0 and ii!=Nz-1 and jj!=0 and jj!=Nr-1:
                    self.grid[ii,jj] = Node(z[ii],r[jj],'')
                else:
                    raise ValueError("The constructor of the grid has some problems")
        
        self.density = np.empty((self.nAxialNodes,self.nRadialNodes))
        self.axialVelocity = np.empty((self.nAxialNodes,self.nRadialNodes))
        self.radialVelocity = np.empty((self.nAxialNodes,self.nRadialNodes))
        self.tangentialVelocity = np.empty((self.nAxialNodes,self.nRadialNodes))
        self.pressure = np.empty((self.nAxialNodes,self.nRadialNodes))

        
    def PrintInfo(self, datafile='terminal'):
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].PrintInfo(datafile)
    
    def AddDensityField(self, rho):
        #structured density field to be added
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].AddDensity(rho[ii,jj])
    
    def AddVelocityField(self, vz, vr, vtheta):
        #structured velocity field to be added
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].AddVelocity(vz[ii,jj], vr[ii,jj], vtheta[ii,jj])
    
    def AddPressureField(self, p):
        #structured pressure field to be added
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].AddPressure(p[ii,jj])
    
    def GetPressureField(self, p):
        #structured pressure field to be otained
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].AddPressure(p[ii,jj])
    
    def ContourPlotDensity(self, formatFig=(10,6)):
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.density[ii,jj] = self.grid[ii,jj].rho
        plt.figure(figsize=formatFig)
        plt.contourf(self.z_grid, self.r_grid, self.density)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Density')
        cb = plt.colorbar()
        cb.set_label(r'$\rho \ \ [-]$')
        
    def ContourPlotVelocity(self, direction=1,formatFig=(10,6)):
        if direction==1:
            for ii in range(0,self.nAxialNodes):
                for jj in range(0,self.nRadialNodes):
                    self.axialVelocity[ii,jj] = self.grid[ii,jj].vz
            plt.figure(figsize=formatFig)
            plt.contourf(self.z_grid, self.r_grid, self.axialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Axial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{z} \ \ [-]$')
        elif direction==2:
            for ii in range(0,self.nAxialNodes):
                for jj in range(0,self.nRadialNodes):
                    self.radialVelocity[ii,jj] = self.grid[ii,jj].vr
            plt.figure(figsize=formatFig)
            plt.contourf(self.z_grid, self.r_grid, self.radialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Radial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{r} \ \ [-]$')
        elif direction==3:
            for ii in range(0,self.nAxialNodes):
                for jj in range(0,self.nRadialNodes):
                    self.tangentialVelocity[ii,jj] = self.grid[ii,jj].vtheta
            plt.figure(figsize=formatFig)
            plt.contourf(self.z_grid, self.r_grid, self.tangentialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Tangential Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{\theta} \ \ [-]$')
        else:
            raise ValueError('Insert a number from 1 to 3 to select the velocity component!')
    
    def ContourPlotPressure(self, formatFig=(10,6)):
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.pressure[ii,jj] = self.grid[ii,jj].p
        plt.figure(figsize=formatFig)
        plt.contourf(self.z_grid, self.r_grid, self.pressure)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Pressure')
        cb = plt.colorbar()
        cb.set_label(r'$p \ \ [-]$')
    
    
data = AnnulusDuctGrid(1, 2, 5, 50, 25)
rho = np.random.rand(50,25)
u = np.random.rand(50,25)
v = np.random.rand(50,25)
w = np.random.rand(50,25)
p = np.random.rand(50,25)
for ii in range(0,50):
    for jj in range(0,25):
        rho[ii,jj] = ii*jj
        u[ii,jj] = ii
        v[ii,jj] = jj
        w[ii,jj] = ii*jj**5
        p[ii,jj] = ii+jj
        
data.AddDensityField(rho)
data.AddVelocityField(u, v, w)
data.AddPressureField(p)
data.ContourPlotDensity()
data.ContourPlotVelocity(1)
data.ContourPlotVelocity(2)
data.ContourPlotVelocity(3)
# data.ContourPlotPressure()




data.PrintInfo()



