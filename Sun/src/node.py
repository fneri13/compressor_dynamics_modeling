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
        

