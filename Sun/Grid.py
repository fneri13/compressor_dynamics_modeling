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
    Class of Nodes, contaning cordinates, boundary marker and fluid dynamics field [rho,u,v,w,p]
    """
    def __init__(self, z, r, marker):
        self.z = z #axis of the machine
        self.r = r #axis of the machine
        self.marker = marker
    
    def PrintInfo(self, datafile='terminal'):
        if datafile == 'terminal':
            print('marker: ' + self.marker)
            print('r: %.2f' %(self.r))
            print('z: %.2f' %(self.z))
            print('-----------------------------------------------')
        else: 
            with open(datafile, 'a') as f:
                print('marker: ' + self.marker, file=f)
                print('r: %.2f' %(self.r), file=f)
                print('z: %.2f' %(self.z), file=f)
                print('-----------------------------------------------', file=f)
        
        

class AnnulusDuctGrid():
    """
    Class of Grid for cylindrical duct. It contains a grid of Node objects, on which every node has the properties
    """
    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, mode='default'):
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        if mode == 'default':
            self.z = np.linspace(zmin, zmax, Nz)
            self.r = np.linspace(rmin, rmax, Nr)
        elif mode == 'gauss-lobatto':
            x = np.array(()) #xi direction
            y = np.array(()) #eta direction
            for i in range(0,self.nAxialNodes):
                xnew = np.cos(i*np.pi/(self.nAxialNodes-1)) #gauss lobatto points
                x = np.append(x, xnew)
            for j in range(0,self.nRadialNodes):
                ynew = np.cos(j*np.pi/(self.nRadialNodes-1)) #gauss lobatto points
                y = np.append(y, ynew)
            self.z = np.flip(x)
            self.r = np.flip(y)
        
        self.r_grid, self.z_grid = np.meshgrid(self.r,self.z)
        self.grid = np.empty((Nz, Nr), dtype=Node)
        for ii in range(0,Nz):
            for jj in range(0,Nr):
                if ii==0:
                    self.grid[ii,jj] = Node(self.z[ii],self.r[jj],'inlet')
                elif ii==Nz-1:
                    self.grid[ii,jj] = Node(self.z[ii],self.r[jj],'outlet')
                elif jj==0 and ii!=0 and ii!=Nz-1:
                    self.grid[ii,jj] = Node(self.z[ii],self.r[jj],'hub')
                elif jj==Nr-1 and ii!=0 and ii!=Nz-1:
                    self.grid[ii,jj] = Node(self.z[ii],self.r[jj],'shroud')
                elif ii!=0 and ii!=Nz-1 and jj!=0 and jj!=Nr-1:
                    self.grid[ii,jj] = Node(self.z[ii],self.r[jj],'')
                else:
                    raise ValueError("The constructor of the grid has some problems")
        
        self.density = np.zeros((self.nAxialNodes,self.nRadialNodes))
        self.axialVelocity = np.zeros((self.nAxialNodes,self.nRadialNodes))
        self.radialVelocity = np.zeros((self.nAxialNodes,self.nRadialNodes))
        self.tangentialVelocity = np.zeros((self.nAxialNodes,self.nRadialNodes))
        self.pressure = np.zeros((self.nAxialNodes,self.nRadialNodes))

        
    def PrintInfo(self, datafile='terminal'):
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].PrintInfo(datafile)
    
    def AddDensityField(self, rho):
        #structured density field to be added
        self.density = rho
    
    def AddVelocityField(self, vz, vr, vtheta):
        self.axialVelocity = vz
        self.radialVelocity = vr
        self.tangentialVelocity = vtheta
    
    def AddPressureField(self, p):
        #structured pressure field to be added
        self.pressure = p
    
    def ContourPlotDensity(self, formatFig=(10,6)):
        plt.figure(figsize=formatFig)
        plt.contourf(self.z_grid, self.r_grid, self.density)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Density')
        cb = plt.colorbar()
        cb.set_label(r'$\rho \ \ [-]$')
        
    def ContourPlotVelocity(self, direction=1,formatFig=(10,6)):
        if direction==1:
            
            plt.figure(figsize=formatFig)
            plt.contourf(self.z_grid, self.r_grid, self.axialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Axial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{z} \ \ [-]$')
        elif direction==2:
            
            plt.figure(figsize=formatFig)
            plt.contourf(self.z_grid, self.r_grid, self.radialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Radial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{r} \ \ [-]$')
        elif direction==3:
            
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
        plt.figure(figsize=formatFig)
        plt.contourf(self.z_grid, self.r_grid, self.pressure)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Pressure')
        cb = plt.colorbar()
        cb.set_label(r'$p \ \ [-]$')
    
    def PhysicalToSpectralData(self):
        x = np.array(()) #xi direction
        y = np.array(()) #eta direction
        for i in range(0,self.nAxialNodes):
            xnew = np.cos(i*np.pi/(self.nAxialNodes-1)) #gauss lobatto points
            x = np.append(x, xnew)
        for j in range(0,self.nRadialNodes):
            ynew = np.cos(j*np.pi/(self.nRadialNodes-1)) #gauss lobatto points
            y = np.append(y, ynew)
        x = np.flip(x)
        y = np.flip(y)
        newGridObj = AnnulusDuctGrid(-1, 1, -1, 1, self.nAxialNodes, self.nRadialNodes, mode='gauss-lobatto')
        newGridObj.AddDensityField(self.density) #add the same fields from the physical grid
        newGridObj.AddVelocityField(self.axialVelocity, self.radialVelocity, self.tangentialVelocity)
        newGridObj.AddPressureField(self.pressure)
        return newGridObj
    
    def ShowGrid(self, formatFig=(10,6)):
        # inlet_set = (self.grid.)
        mark = np.empty((self.nAxialNodes, self.nRadialNodes), dtype=str)
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                if self.grid[ii,jj].marker=='inlet':
                    mark[ii,jj] = "i"
                elif self.grid[ii,jj].marker=='outlet':
                    mark[ii,jj] = "o"
                elif self.grid[ii,jj].marker=='hub':
                    mark[ii,jj] = "h"
                elif self.grid[ii,jj].marker=='shroud':
                    mark[ii,jj] = "s"
                else:
                    mark[ii,jj] = ''

        plt.figure(figsize=formatFig)
        condition = mark == 'i'  # Define a boolean condition
        plt.scatter(self.z_grid[condition], self.r_grid[condition], label='inlet')
        condition = mark == 'o'  # Define a boolean condition
        plt.scatter(self.z_grid[condition], self.r_grid[condition], label='outlet')
        condition = mark == 'h'  # Define a boolean condition
        plt.scatter(self.z_grid[condition], self.r_grid[condition], label='hub')
        condition = mark == 's'  # Define a boolean condition
        plt.scatter(self.z_grid[condition], self.r_grid[condition], label='shroud')
        condition = mark == ''  # Define a boolean condition
        plt.scatter(self.z_grid[condition], self.r_grid[condition], c='black')
        
        
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.legend()
        
    
    



