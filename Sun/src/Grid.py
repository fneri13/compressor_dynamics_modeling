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
    Class of Nodes, contaning cordinates, fluid dynamics field, and marker for BCs understanding
    """
    def __init__(self, z, r, rho, vz, vr, vt, p, marker, nodeCounter):
        self.z = z                          #axial cordinate
        self.r = r                          #radial cordinate
        self.marker = marker                #type of node if belonging to boundaries
        self.density = rho                  #density
        self.axialVelocity = vz             #axial velocity (z-axis)
        self.radialVelocity = vr            #radial velocity
        self.tangentialVelocity = vt        #tangential velocity
        self.pressure = p                   #pressure
        self.nodeCounter = nodeCounter      #identifier of the node
    
    def GetAxialCordinate(self):
        return self.z
    
    def GetRadialCordinate(self):
        return self.r
    
    def GetDensity(self):
        return self.density
    
    def GetAxialVelocity(self):
        return self.axialVelocity
    
    def GetRadialVelocity(self):
        return self.radialVelocity
    
    def GetTangentialVelocity(self):
        return self.tangentialVelocity
    
    def GetPressure(self):
        return self.pressure
    
    def GetMarker(self):
        return self.marker
    
    def PrintInfo(self, datafile='terminal'):
        #not important for the moment
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
    
    def AddAMatrix(self,A):
        self.A = A
    
    def AddBMatrix(self,B):
        self.B = B
        
    def AddCMatrix(self,C):
        self.C = C
    
    def AddEMatrix(self,E):
        self.E = E
    
    def AddRMatrix(self,R):
        self.R = R
        
    def AddSMatrix(self,S):
        self.S = S
        
    def AddJacobianGradients(self, dxdz, dxdr, dydz, dydr):
        self.dxdz,  self.dxdr, self.dydz, self.dydr = dxdz, dxdr, dydz, dydr
    
    def AddInverseJacobianGradients(self, dzdx, dzdy, drdx, drdy):
        self.dzdx,  self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy
    
    def AddHatMatrices(self,Bhat,Ehat):
        self.Bhat, self.Ehat = Bhat, Ehat
        

################################################################################################################################
################################################################################################################################
################################################################################################################################



class DataGrid():
    """
    Class of Grid. It contains a grid of Node objects, on which every node has the properties that we need. Conserving the 
    structured grid, so it will be easier to differentiate
    """
    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, rho, vz, vr, vt, p, mode='default'):
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        self.nPoints = Nz*Nr
        if mode == 'default':
            self.z = np.linspace(zmin, zmax, Nz)
            self.r = np.linspace(rmin, rmax, Nr)
        elif mode == 'gauss-lobatto':
            #construct a gauss-lobatto grid for the spectral dataset
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
        for ii in range(0,self.nAxialNodes):
            for jj in range(0,self.nRadialNodes):
                self.grid[ii,jj].PrintInfo(datafile)
    
    def AddDensityField(self, rho):
        #structured density field to be added
        if (len(rho)!=self.nPoints):
            raise Exception('Error of length')
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
        plt.contourf(self.zGrid, self.rGrid, self.density)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Density')
        cb = plt.colorbar()
        cb.set_label(r'$\rho \ \ [-]$')
        
    def ContourPlotVelocity(self, direction=1,formatFig=(10,6)):
        if direction==1:
            
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.axialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Axial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{z} \ \ [-]$')
        elif direction==2:
            
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.radialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Radial Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{r} \ \ [-]$')
        elif direction==3:
            
            plt.figure(figsize=formatFig)
            plt.contourf(self.zGrid, self.rGrid, self.tangentialVelocity)
            plt.xlabel(r'$Z$')
            plt.ylabel(r'$R$')
            plt.title('Tangential Velocity')
            cb = plt.colorbar()
            cb.set_label(r'$u_{\theta} \ \ [-]$')
        else:
            raise ValueError('Insert a number from 1 to 3 to select the velocity component!')
    
    def ContourPlotPressure(self, formatFig=(10,6)):
        plt.figure(figsize=formatFig)
        plt.contourf(self.zGrid, self.rGrid, self.pressure)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title('Pressure')
        cb = plt.colorbar()
        cb.set_label(r'$p \ \ [-]$')
    
    def PhysicalToSpectralData(self):
        """
        it returns a new object with the same data of the original one, but mapped to a spectral system of cordinates, on the gauss-lobatto 
        cordinates between [-1,1] 
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
    
    def ShowGrid(self, formatFig=(10,6)):
        """
        Show a scatter plots of the grid, with different colors for the different patches
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
        
    
    

################################################################################################################################
################################################################################################################################
################################################################################################################################


