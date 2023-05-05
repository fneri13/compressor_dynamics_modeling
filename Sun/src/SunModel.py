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

plt.rc('text')      
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rcParams['font.size'] = 14


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
    def __init__(self, gridObject):
        """
        it builds the object and the related data
        """
        self.data = gridObject
        self.nPoints = (gridObject.nAxialNodes)*(gridObject.nRadialNodes)
        self.nInternalPoints = (gridObject.nAxialNodes-2)*(gridObject.nRadialNodes-2)
        self.A = np.zeros((self.nPoints*5, self.nPoints*5))
        self.B = np.zeros((self.nPoints*5, self.nPoints*5))
        self.C = np.zeros((self.nPoints*5, self.nPoints*5))
        self.E = np.zeros((self.nPoints*5, self.nPoints*5))
        self.R = np.zeros((self.nPoints*5, self.nPoints*5))
        self.S = np.zeros((self.nPoints*5, self.nPoints*5)) 
        self.gmma = 1.4 #cp/cv for standard air for the moment
    
    def CreateAMatrix(self):
        """
        Diagonal matrix
        """
        self.A = np.eye(self.nPoints*5) #A is the diagonal matrix
        #probably it is a better to create a 2D array of matrices, where for evere point we have 1 set of equations
    
    def CreateBMatrix(self):
        """
        Coefficient Matrix related to radial derivatives
        """
        nBlock = 0
        for IterZ in range(0,self.data.nAxialNodes):
            for IterR in range(0,self.data.nRadialNodes):
                #for every point on the grid, construct a matrix B, and then stack them along the diagonal. Remember than the blocks are taken going along the j (radial direction) axis and i axis later (axial direction).
                #this decides the flatten() method for the gradients used in ComputeHatMatrices
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
        # print('attention because in this matrix, for the real case there are the gradients of the mean flow, that for the moment are not implemented, since the acoustic duct is a uniform flow. the zeros are hardcoded, so remember to implement the correct thing')
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
                R[2,1] = 0 
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
    
    def CreateSMatrix(self):
        """
        Coefficient matrix related to the body force terms. To be implemented yet
        """
        self.S = np.ones((self.nPoints*5, self.nPoints*5)) #A is the diagonal matrix
    
    def CreateAllPhysicalMatrices(self):
        """
        Compute all the matrices together. The boundaries points will need to be treated later, modifying the corresponding equations
        """
        self.CreateAMatrix()
        self.CreateBMatrix()
        self.CreateCMatrix()
        self.CreateEMatrix()
        self.CreateRMatrix()
        self.CreateSMatrix()
    
    # def GlobalStabilityMatrix(self, omega):
    #     """
    #     Giving a complex eigenfrequency, it returns the global instability matrix (for the moment without caring about the boundary nodes)
    #     radialDiff and axialDiff will give the elements of radial and axial differentiations using chebyshev gauss-lobatto method.
        
    #     It needs to be implemented
    #     """
    #     #Q = 1j*omega*self.A + radialDiff(self.B) + 1j*m*self.Cr + axialDiff(E) + self.R + self.S
    #     # return Q
    
    def ComputeSpectralGrid(self):
        self.dataSpectral = self.data.PhysicalToSpectralData()
    
    def ShowPhysicalGrid(self):
        self.data.ShowGrid()
        plt.title('physical grid')
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
    
    def ShowSpectralGrid(self):
        self.dataSpectral.ShowGrid()
        plt.title('spectral grid')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
    
    def ComputeJacobianSpectral(self):
        """
        The Jacobian for the spectral grid as a function of the physical grid cordinates is implemented here. It computes the derivatives
        needed to obtain the matrices in the spectral space.
        if you need, just drag it out and use it how you want
        """
        #grids
        Z = self.data.z_grid
        R = self.data.r_grid
        X = self.dataSpectral.z_grid
        Y = self.dataSpectral.r_grid
        self.dxdz, self.dxdr, self.dydz, self.dydr = JacobianTransform(X,Y,Z,R)
    
    def ComputeJacobianPhysical(self):
        """
        The Jacobian for the physical grid as a function of the pspectral grid cordinates is implemented here. It computes the derivatives
        needed to obtain the matrices in the spectral space.
        if you need, just drag it out and use it how you want
        """
        #grids
        Z = self.data.z_grid
        R = self.data.r_grid
        X = self.dataSpectral.z_grid
        Y = self.dataSpectral.r_grid
        self.dzdx, self.dzdy, self.drdx, self.drdy = JacobianTransform(Z,R,X,Y)
        print('you should check that the derivatives are indeed opposites')
        
    
    def ShowJacobianSpectralAxis(self, formatFig=(10,6)):
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.z_grid, self.dataSpectral.r_grid, c=self.dxdz)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\frac{\partial \xi}{\partial z}$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \xi}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.z_grid, self.dataSpectral.r_grid, c=self.dxdr)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial r}$')
        cb.set_label(r'$\frac{\partial \xi}{\partial r}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.z_grid, self.dataSpectral.r_grid, c=self.dydz)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial z}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.z_grid, self.dataSpectral.r_grid, c=self.dydr)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial r}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial r}$')
        
    def ShowJacobianPhysicalAxis(self, formatFig=(10,6)):
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.z_grid, self.data.r_grid, c=self.dxdz)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title(r'$\frac{\partial \xi}{\partial z}$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \xi}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.z_grid, self.data.r_grid, c=self.dxdr)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial r}$')
        cb.set_label(r'$\frac{\partial \xi}{\partial r}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.z_grid, self.data.r_grid, c=self.dydz)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial z}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.z_grid, self.data.r_grid, c=self.dydr)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial r}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial r}$')

    
    def ComputeHatMatrices(self):
        Bhat = np.zeros(self.B.shape) #physical B
        Ehat = np.zeros(self.B.shape) #physical E
        Ni = Bhat.shape[0]
        Nj = Bhat.shape[1]
        nBlocks = Ni//5
        
        #flatten the gradients to be used later in the matrix multiplications. I am unrolling along the column (Fortran style), because the matrices were built going along the j axis before than the i axis.
        #you should check this in a later stage of the process, during code debugging and testing
        dxdz, dxdr, dydz, dydr = self.dxdz.flatten(order='F'), self.dxdr.flatten(order='F'), self.dydz.flatten(order='F'), self.dydr.flatten(order='F') 
        
        if nBlocks!=self.nPoints:
            raise ValueError('Error in Compute Modified B matrix method')    
            
        for blockIt in range(0,nBlocks):
            Bhat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdr[blockIt] \
                                                                        + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdz[blockIt]
            Ehat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydr[blockIt] \
                                                                        + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydz[blockIt]
        self.Bhat = Bhat
        self.Ehat = Ehat
    
    def ComputeSVD(self, omega_domain=[-1,1,-1,1], grid_omega=[10,10]):
        omR_min = omega_domain[0]
        omR_max = omega_domain[1]
        omI_min = omega_domain[2]
        omI_max = omega_domain[3]
        nR = grid_omega[0]
        nI = grid_omega[1]
        omR = np.linspace(omR_min, omR_max, nR)
        omI = np.linspace(omI_min, omI_max, nI)
        self.omegaR, self.omegaI = np.meshgrid(omR, omI)
        self.chi = np.zeros((nR,nI))
        for ii in range(0,nR):
            for jj in range(0,nI):
                omega = omR[ii]+1j*omI[jj]
                
                #to be implemented correctly later
                Q = 1j*omega*self.A+self.B+self.C+self.E+self.R+self.S

                U,S,V = np.linalg.svd(Q)
                self.chi[ii,jj] = np.min(S)/np.max(S)\
    
    
    def PlotInverseConditionNumber(self, formatFig=(10,6)):
        x = np.linspace(np.min(self.omegaR), np.max(self.omegaR))
        critical_line = np.zeros(len(x))
        
        
        plt.figure(figsize=formatFig)
        plt.contourf(self.omegaR, self.omegaI, self.chi)
        plt.plot(x, critical_line, '--r')
        plt.xlabel(r'$\omega_{R}$')
        plt.ylabel(r'$\omega_{I}$')
        plt.title(r'$\chi$ locus')
        cb = plt.colorbar()
        cb.set_label(r'$\chi$')
    
    # def CreateAMatrixCoefficients(self):
    #     #second version of the coefficients
    #     Nz = self.data.nAxialNodes
    #     Nr = self.data.nAxialNodes
    #     for ii in range(0,Nz):
    #         for jj in range(0,Nr):
    #             marker = print(self.data.grid[ii,jj].marker) #get the type of node
    #             self.data.grid[ii,jj].AddMatrixA()
                
            
        
        
                
                
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
# GENERAL FUNCTIONS USED IN THE CODE. THEY CAN BE ACCESSED ALSO FROM OUTSIDE, NOT EXCLUSIVELY BY CLASS MEMBERS  
def JacobianTransform(X,Y,Z,R):
    Nz, Nr = X.shape[0], X.shape[1]
    
    #instantiate matrices
    dxdr = np.zeros((Nz, Nr))
    dxdz = np.zeros((Nz, Nr))
    dydr = np.zeros((Nz, Nr))
    dydz = np.zeros((Nz, Nr))
    
    #2nd order central difference for the grid. First take care of the corners, then of the edges, and then of the central points
    for ii in range(0,Nz):
        for jj in range(0,Nr):
            if (ii==0 and jj==0): #lower-left corner
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
            elif (ii==Nz-1 and jj==0): #lower-right corner
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
            elif (ii==0 and jj==Nr-1): #upper-left corner
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
            elif (ii==Nz-1 and jj==Nr-1): #upper-right corner
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
            elif (ii==0): #left side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
            elif (ii==Nz-1): #right side
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
            elif (jj==0): #lower side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
            elif (jj==Nr-1): #upper side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
            else: #internal points
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(2*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/(2*(R[ii,jj+1]-R[ii,jj-1]))
    
    return dxdz, dxdr, dydz, dydr









 
        
        
        
        
        
        
        
        
        
        
        
        
        