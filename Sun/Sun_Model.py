#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:32:43 2023
@author: F. Neri, TU Delft
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Grid import Node, AnnulusDuctGrid


class SunModel:
    """
    Class used to perform the Sun Model based on the data contained in a Grid object containing the CFD results. The matrix elements are
    taken from Aerodynamic Instabilities of Swept Airfoil Design in Transonic Axial-Flow Compressors, He et Al
    
    ATTRIBUTES:
        data : grid object contaning the CFD results
        nPoints : total number of grid points (boundaries included)
        nInternalPoints : number of grid points (excluding the boundaries points)
        A : coefficient matrix of temporal derivatives
        B : coefficient matrix of radial derivatives
        C : coefficient matrix of azimuthal derivative s
        E : coefficient matrix of axial derivatives
        R : coefficient matrix of the known mean flow terms
        S : coefficient matric of the body force model
    """
    def __init__(self, DuctObject):
        """
        it builds the object and the related data
        """
        self.data = DuctObject
        self.nPoints = (DuctObject.nAxialNodes)*(DuctObject.nRadialNodes)
        self.nInternalPoints = (DuctObject.nAxialNodes-2)*(DuctObject.nRadialNodes-2)
        self.A = np.zeros((self.nPoints*5, self.nPoints*5))
        self.B = np.zeros((self.nPoints*5, self.nPoints*5))
        self.C = np.zeros((self.nPoints*5, self.nPoints*5))
        self.E = np.zeros((self.nPoints*5, self.nPoints*5))
        self.R = np.zeros((self.nPoints*5, self.nPoints*5))
        self.S = np.zeros((self.nPoints*5, self.nPoints*5)) 
        self.gmma = 1.4
    
    def CreateAMatrix(self):
        """
        Diagonal matrix
        """
        self.A = np.eye(self.nPoints*5) #A is the diagonal matrix
    
    def CreateBMatrix(self):
        """
        Coefficient Matrix related to radial derivatives
        """
        nBlock = 0
        for IterZ in range(0,self.data.nAxialNodes):
            for IterR in range(0,self.data.nRadialNodes):
                #for every point on the grid, construct a matrix B, and then stack them along the diagonal
                B = np.zeros((5,5))
                
                B[0,0] = self.data.radialVelocity[IterZ,IterR]
                B[1,1] = self.data.radialVelocity[IterZ,IterR]
                B[2,2] = self.data.radialVelocity[IterZ,IterR]
                B[3,3] = self.data.radialVelocity[IterZ,IterR]
                B[4,4] = self.data.radialVelocity[IterZ,IterR]
                
                B[0,1] = self.data.density[IterZ,IterR]
                B[1,4] = 1/self.data.density[IterZ,IterR]
                B[4,1] = self.data.pressure[IterZ,IterR]*self.gmma
                
                self.B[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = B
                nBlock = nBlock+1
    
    def CreateCMatrix(self):
        """
        Coefficient Matrix related to tangential derivatives
        """
        nBlock = 0
        for IterZ in range(0,self.data.nAxialNodes):
            for IterR in range(0,self.data.nRadialNodes):
                #for every point on the grid, construct a matrix C, and then stack them along the diagonal
                C = np.zeros((5,5))
                
                C[0,0] = self.data.tangentialVelocity[IterZ,IterR]
                C[1,1] = self.data.tangentialVelocity[IterZ,IterR]
                C[2,2] = self.data.tangentialVelocity[IterZ,IterR]
                C[3,3] = self.data.tangentialVelocity[IterZ,IterR]
                C[4,4] = self.data.tangentialVelocity[IterZ,IterR]
                
                C[0,2] = self.data.density[IterZ,IterR]
                C[2,4] = 1/self.data.density[IterZ,IterR]
                C[4,2] = self.data.pressure[IterZ,IterR]*self.gmma
                
                self.C[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = C
                nBlock = nBlock+1
    
    def CreateEMatrix(self):
        """
        Coefficient Matrix related to axial derivatives
        """
        nBlock = 0
        for IterZ in range(0,self.data.nAxialNodes):
            for IterR in range(0,self.data.nRadialNodes):
                #for every point on the grid, construct a matrix E, and then stack them along the diagonal
                E = np.zeros((5,5))
                
                E[0,0] = self.data.axialVelocity[IterZ,IterR]
                E[1,1] = self.data.axialVelocity[IterZ,IterR]
                E[2,2] = self.data.axialVelocity[IterZ,IterR]
                E[3,3] = self.data.axialVelocity[IterZ,IterR]
                E[4,4] = self.data.axialVelocity[IterZ,IterR]
                
                E[0,3] = self.data.density[IterZ,IterR]
                E[3,4] = 1/self.data.density[IterZ,IterR]
                E[4,3] = self.data.pressure[IterZ,IterR]*self.gmma
                
                self.E[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = E
                nBlock = nBlock+1
    
    def CreateRMatrix(self):
        """
        Coefficient matrix related to the mean flow known terms. To be implemented correctly in the general case. Now it is hardcoded
        for the uniform acoustic duct.
        """
        print('attention because in this matrix, for the real case there are the gradients of the mean flow, that for the moment are not implemented, since the acoustic duct is a uniform flow. the zeros are hardcoded, so remember to implement the correct thing')
        print('you should also check the R[2,1], it seems that there is a typo')
        nBlock = 0
        for IterZ in range(0,self.data.nAxialNodes):
            for IterR in range(0,self.data.nRadialNodes):
                #for every point on the grid, construct a matrix R, and then stack them along the diagonal
                R = np.zeros((5,5))
                
                R[0,0] = self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR]
                R[0,1] = self.data.density[IterZ,IterR]/self.data.r[IterR] + 0
                R[0,2] = 0
                R[0,3] = 0 
                R[0,4] = 0 
                R[1,0] = -1/(self.data.density[IterZ,IterR]**2)*0
                R[1,1] = 0
                R[1,2] = -2*self.data.tangentialVelocity[IterZ,IterR]*0
                R[1,3] = 0
                R[1,4] = 0
                R[2,0] = 0
                R[2,1] = 0 #there is a doubt on this
                R[2,2] = self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR]
                R[2,3] = 0
                R[2,4] = 0
                R[3,:] = [0,0,0,0,0]
                R[4,0] = 0
                R[4,1] = self.data.pressure[IterZ,IterR]*self.gmma/self.data.r[IterR]
                R[4,2] = 0
                R[4,3] = 0
                R[4,4] = self.gmma*(0+0+self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR])
                
                
                self.R[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = R
                nBlock = nBlock+1
    
    def CreateAllMatrices(self):
        """
        Compute all the matrices together. The boundaries points will need to be treated later, modifying the corresponding equations
        """
        self.CreateAMatrix()
        self.CreateBMatrix()
        self.CreateCMatrix()
        self.CreateEMatrix()
        self.CreateRMatrix()
    
    def GlobalStabilityMatrix(self, omega):
        """
        Giving a complex eigenfrequency, it returns the global instability matrix (for the moment without caring about the boundary nodes)
        radialDiff and axialDiff will give the elements of radial and axial differentiations using chebyshev gauss-lobatto method.
        
        It needs to be implemented
        """
        #Q = 1j*omega*self.A + radialDiff(self.B) + 1j*m*self.Cr + axialDiff(E) + self.R + self.S
        # return Q
                        
                        

        






#%%
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
Nz = 3
Nr = 3
duct = AnnulusDuctGrid(r1, r2, L, Nz, Nr)

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

sunObj = SunModel(duct)
sunObj.CreateAllMatrices()





 
        
        
        
        
        
        
        
        
        
        
        
        
        