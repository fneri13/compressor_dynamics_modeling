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
    
    def CreateAllMatrices(self):
        """
        Compute all the matrices together. The boundaries points will need to be treated later, modifying the corresponding equations
        """
        self.CreateAMatrix()
        self.CreateBMatrix()
        self.CreateCMatrix()
        self.CreateEMatrix()
        self.CreateRMatrix()
    
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
    
    def ComputeGridTransformationLaw(self):
        """
        The jacobian transform is implemented here. if you need, just drag it out and use it how you want
        """
        physicalGridObj = self.data
        spectralGridObj = self.dataSpectral
        
        #grids
        Z = physicalGridObj.z_grid
        R = physicalGridObj.r_grid
        X = spectralGridObj.z_grid
        Y = spectralGridObj.r_grid
        
        Nz = self.data.nAxialNodes
        Nr = self.data.nRadialNodes
        
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
  
        self.dxdz, self.dxdr, self.dydz, self.dydr = dxdz, dxdr, dydz, dydr
    
        
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
        
    
    def ComputeModifiedMatrices(self):
        Bhat = np.zeros(self.B.shape) #physical B
        Ehat = np.zeros(self.B.shape) #physical E
        Ni = Bhat.shape[0]
        Nj = Bhat.shape[1]
        nBlocks = Ni//5
        
        #flatten the gradients to be used later in the matrix multiplications (check the order of unroll, but it should be first z and then r, which is the same used for the matrices)
        dxdz, dxdr, dydz, dydr = self.dxdz.flatten(), self.dxdr.flatten(), self.dydz.flatten(), self.dydr.flatten() 
        if nBlocks!=self.nPoints:
            raise ValueError('Error in Compute Modified B matrix method')    
        for blockIt in range(0,nBlocks):
            Bhat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdr[blockIt] \
                                                                        + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdz[blockIt]
            Ehat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydr[blockIt] \
                                                                        + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydz[blockIt]
        self.Bhat = Bhat
        self.Ehat = Ehat
    
    def ComputeSVD(self):
        #to be implemented correctly
        Q = self.A+self.B+self.C+self.E+self.R+self.S
        U,S,V = np.linalg.svd(Q)
        return S
        










 
        
        
        
        
        
        
        
        
        
        
        
        
        