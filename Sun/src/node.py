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
    Class of Nodes, contaning cordinates, fluid dynamics field, markers for BCs understanding and gric cordinates of the node
    """
    def __init__(self, z, r, rho, vz, vr, vt, p, marker, nodeCounter, gridCordinates):
        self.z = z                                  #axial cordinate
        self.r = r                                  #radial cordinate
        self.marker = marker                        #type of node if belonging to boundaries
        self.density = rho                          #density
        self.axialVelocity = vz                     #axial velocity (z-axis)
        self.radialVelocity = vr                    #radial velocity
        self.tangentialVelocity = vt                #tangential velocity
        self.pressure = p                           #pressure
        self.nodeCounter = nodeCounter              #identifier of the node
        self.gridCordinates = gridCordinates        #location of the node on the grid. needed to know neighbours
    
    def GetAxialCordinate(self):
        """
        it returns the z-cordinate of the node
        """
        return self.z
    
    def GetRadialCordinate(self):
        """
        it returns the r-cordinate of the node
        """
        return self.r
    
    def GetDensity(self):
        """
        it returns the density of the node
        """
        return self.density
    
    def GetAxialVelocity(self):
        """
        it returns the z-velocity of the node
        """
        return self.axialVelocity
    
    def GetRadialVelocity(self):
        """
        it returns the r-cordinate of the node
        """
        return self.radialVelocity
    
    def GetTangentialVelocity(self):
        """
        it returns the theta-cordinate of the node
        """
        return self.tangentialVelocity
    
    def GetPressure(self):
        """
        it returns the pressure value at the node
        """
        return self.pressure
    
    def GetMarker(self):
        """
        it returns the marker of the node
        """
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
    
    def AddAMatrix(self, A, omega):
        """
        It add the A matrix, already multiplied by omega and j at the node level
        """
        self.A = 1j*omega*A
    
    def AddBMatrix(self,B):
        """
        It add the B matrix at the node level
        """
        self.B = B
        
    def AddCMatrix(self, C, m):
        """
        It add the C matrix, already multiplied by m and j at the node level
        """
        self.C = C*1j*m/self.r
    
    def AddEMatrix(self,E):
        """
        It add the E matrix at the node level
        """
        self.E = E
    
    def AddRMatrix(self,R):
        """
        It add the R matrix at the node level
        """
        self.R = R
        
    def AddSMatrix(self,S):
        """
        It add the S matrix at the node level
        """
        self.S = S
        
    def AddJacobianGradients(self, dxdz, dxdr, dydz, dydr):
        """
        It add the spectral jacobian as a function of the physical cordinates at the node level
        """
        self.dxdz,  self.dxdr, self.dydz, self.dydr = dxdz, dxdr, dydz, dydr
    
    def AddInverseJacobianGradients(self, dzdx, dzdy, drdx, drdy):
        """
        It add the physical jacobian as a function of the  spectral cordinates at the node level (NOT CORRECT IF THE PHYSICAL GRID IS NOT CARTESIAN)
        """
        self.dzdx,  self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy
    
    def AddHatMatrices(self,Bhat,Ehat):
        """
        It add the \hat{B}, \hat{E} matrix at the node level
        """
        self.Bhat, self.Ehat = Bhat, Ehat
        
    def ApplyInletCondition(self):
        print('INLET condition virtually applied. (To be implemented when the full stability matrix will be ready)')
    
    def ApplyOutletCondition(self):
        print('OUTLET condition virtually applied. (To be implemented when the full stability matrix will be ready)')
        
    def ApplyWallCondition(self, flag='euler'):
        if flag=='euler':
            print('EULER WALL condition virtually applied. (To be implemented when the full stability matrix will be ready)')
        else:
            raise Exception('Apply a correct wall boundary condition')

