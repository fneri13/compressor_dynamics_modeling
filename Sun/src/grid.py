#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:38:38 2023
@author: F. Neri, TU Delft
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.node import Node
from src.styles import *
from src.general_functions import GaussLobattoPoints

import os

folder_name = 'pictures/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

class DataGrid():
    """
    Class of Grid. It contains a grid of Node objects, on which every node has the properties that we need.
    """
    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, rho, vz, vr, vt, p, mode='default'):
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        self.nPoints = Nz*Nr
        if mode == 'default':
            self.z = np.linspace(zmin, zmax, Nz)
            self.r = np.linspace(rmin, rmax, Nr)
        elif mode == 'gauss-lobatto': #construct a gauss-lobatto grid for the spectral dataset
            self.z = GaussLobattoPoints(self.nAxialNodes)
            self.r = GaussLobattoPoints(self.nRadialNodes)
        
        self.rGrid, self.zGrid = np.meshgrid(self.r,self.z)
        
        self.dataSet = np.empty((Nz,Nr), dtype=Node) #an array of Node elements
        counter = 0
        for ii in range(0,Nz):
            for jj in range(0,Nr):
                if ii==0:
                    self.dataSet[ii,jj] = Node(self.z[ii],self.r[jj],rho[ii,jj], vz[ii,jj], vr[ii,jj], vt[ii,jj], p[ii,jj],'inlet', counter)
                elif ii==Nz-1:
                    self.dataSet[ii,jj] = Node(self.z[ii],self.r[jj],rho[ii,jj], vz[ii,jj], vr[ii,jj], vt[ii,jj], p[ii,jj],'outlet', counter)
                elif jj==0 and ii!=0 and ii!=Nz-1:
                    self.dataSet[ii,jj] = Node(self.z[ii],self.r[jj],rho[ii,jj], vz[ii,jj], vr[ii,jj], vt[ii,jj], p[ii,jj],'hub', counter)
                elif jj==Nr-1 and ii!=0 and ii!=Nz-1:
                    self.dataSet[ii,jj] = Node(self.z[ii],self.r[jj],rho[ii,jj], vz[ii,jj], vr[ii,jj], vt[ii,jj], p[ii,jj],'shroud', counter)
                elif ii!=0 and ii!=Nz-1 and jj!=0 and jj!=Nr-1:
                    self.dataSet[ii,jj] = Node(self.z[ii],self.r[jj],rho[ii,jj], vz[ii,jj], vr[ii,jj], vt[ii,jj], p[ii,jj],'', counter)
                else:
                    raise ValueError("The constructor of the grid has some problems")
                counter = counter + 1
        
        #also construct matrices, for easy plotting
        self.density = rho
        self.axialVelocity = vz
        self.radialVelocity = vr
        self.tangentialVelocity = vt
        self.pressure = p

        
    def PrintInfo(self, datafile='terminal'):
        """
        print information about the nodes
        """
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].PrintInfo(datafile)
    
    def AddDensityField(self, rho):
        """
        add/overwrite the density field of the grid
        """
        if (len(rho)!=self.nPoints):
            raise Exception('Error of length')
        self.density = rho
    
    def AddVelocityField(self, vz, vr, vtheta):
        """
        add/overwrite the velocity field of the grid
        """
        self.axialVelocity = vz
        self.radialVelocity = vr
        self.tangentialVelocity = vtheta
    
    def AddPressureField(self, p):
        """
        add/overwrite the pressure field of the grid
        """
        self.pressure = p
    
    def ContourPlotDensity(self, formatFig=(10,6), save_filename=None):
        """
        contourf plot of the density
        """
        plt.figure(figsize=formatFig)
        plt.contourf(self.zGrid, self.rGrid, self.density)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Density')
        cb = plt.colorbar()
        cb.set_label(r'$\rho \ \ [-]$')
        if save_filename !=None: plt.savefig(folder_name + save_filename + '.pdf' ,bbox_inches='tight')
        
    def ContourPlotVelocity(self, direction=1,formatFig=(10,6), save_filename=None):
        """
        contourf plot of the velocity. Direction variable select one of the velocity components
        """
        if direction==1:
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.axialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Axial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{z} \ \ [-]$')
            if save_filename !=None: plt.savefig(folder_name + save_filename + '_01.pdf' ,bbox_inches='tight')
        elif direction==2:
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.radialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Radial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{r} \ \ [-]$')
            if save_filename !=None: plt.savefig(folder_name + save_filename + '_02.pdf' ,bbox_inches='tight')
        elif direction==3:
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.tangentialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Tangential Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{\theta} \ \ [-]$')
            if save_filename !=None: plt.savefig(folder_name + save_filename + '_03.pdf' ,bbox_inches='tight')
        else:
            raise ValueError('Insert a number from 1 to 3 to select a velocity component')
    
    def ContourPlotPressure(self, formatFig=(10,6), save_filename=None):
        """
        contourf plot of the pressure
        """
        plt.figure(figsize=formatFig)
        plt.contourf(self.zGrid, self.rGrid, self.pressure)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Pressure')
        cb = plt.colorbar()
        cb.set_label(r'$p \ \ [-]$')
        if save_filename !=None: plt.savefig(folder_name + save_filename + '.pdf' ,bbox_inches='tight')
    
    def PhysicalToSpectralData(self):
        """
        it returns a new Grid object with the same data of the original one, but with spectral cordinates, located on the gauss-lobatto 
        points between -1 and 1 in both the directions. It conserves the same amount of grid nodes
        """
        x = np.array(()) #synonim for xi direction = corresponding to streamwise direction
        y = np.array(()) #synonim for eta direction = corresponding to spanwise direction
        for i in range(0,self.nAxialNodes):
            xnew = np.cos(i*np.pi/(self.nAxialNodes-1)) #gauss lobatto points
            x = np.append(x, xnew)
        for j in range(0,self.nRadialNodes):
            ynew = np.cos(j*np.pi/(self.nRadialNodes-1)) #gauss lobatto points
            y = np.append(y, ynew)
        x = np.flip(x)
        y = np.flip(y)
        newGridObj = DataGrid(-1, 1, -1, 1, self.nAxialNodes, self.nRadialNodes, self.density, self.axialVelocity, \
                              self.radialVelocity, self.tangentialVelocity, self.pressure, mode='gauss-lobatto')
        return newGridObj
    
    def ShowGrid(self, formatFig=(10,6), save_filename=None):
        """
        Show a scatter plots of the grid, with different colors for the different zones
        """
        mark = np.empty((self.nAxialNodes, self.nRadialNodes), dtype=str)
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                if self.dataSet[ii,jj].marker=='inlet':
                    mark[ii,jj] = "i"
                elif self.dataSet[ii,jj].marker=='outlet':
                    mark[ii,jj] = "o"
                elif self.dataSet[ii,jj].marker=='hub':
                    mark[ii,jj] = "h"
                elif self.dataSet[ii,jj].marker=='shroud':
                    mark[ii,jj] = "s"
                else:
                    mark[ii,jj] = ''
        plt.figure(figsize=formatFig)
        condition = mark == 'i'  # plot only the inlet points
        plt.scatter(self.zGrid[condition], self.rGrid[condition], label='inlet')
        condition = mark == 'o'  # plot only the outlet points
        plt.scatter(self.zGrid[condition], self.rGrid[condition], label='outlet')
        condition = mark == 'h'  # plot only the hub
        plt.scatter(self.zGrid[condition], self.rGrid[condition], label='hub')
        condition = mark == 's'  # plot only shroud
        plt.scatter(self.zGrid[condition], self.rGrid[condition], label='shroud')
        condition = mark == ''  # plot all the remaining internal points
        plt.scatter(self.zGrid[condition], self.rGrid[condition], c='black')
        plt.legend()
        if save_filename !=None: plt.savefig(folder_name + save_filename + '.pdf' ,bbox_inches='tight')
        
    def GetDensity(self):
        """
        it returns the density 2D array corresponding to the nodes grid values
        """
        return self.density
    
    def GetAxialVelocity(self):
        """
        it returns the z-velocity 2D array corresponding to the nodes grid values
        """
        return self.axialVelocity
    
    def GetRadialVelocity(self):
        """
        it returns the r-velocity 2D array corresponding to the nodes grid values
        """
        return self.radialVelocity
    
    def GetTangentialVelocity(self):
        """
        it returns the theta-velocity 2D array corresponding to the nodes grid values
        """
        return self.tangentialVelocity
    
    def GetPressure(self):
        """
        it returns the pressure 2D array corresponding to the nodes grid values
        """
        return self.pressure
        
        
        
    