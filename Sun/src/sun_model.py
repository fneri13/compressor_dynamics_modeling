#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:32:43 2023
@author: F. Neri, TU Delft
"""

import numpy as np
import matplotlib.pyplot as plt
from src.grid import DataGrid
from src.general_functions import JacobianTransform, ChebyshevDerivativeMatrixBayliss
from src.styles import *
import os


class SunModel:
    """
    Class used to perform the Sun Model based on the data contained in a Grid object containing the CFD results. The matrix elements are
    taken from Aerodynamic Instabilities of Swept Airfoil Design in Transonic Axial-Flow Compressors, He et Al:
        (j*omega*A + \hat{B}*ddxi + j*m*C/r + \hat{E}*ddeta + R + S)*Phi' = 0
    
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
    
    # def CreateAMatrix(self):
    #     """
    #     Create the diagonal A matrix
    #     """
    #     self.A = np.eye(self.nPoints*5) #A is the diagonal matrix
    #     #probably it is a better to create a 2D array of matrices, where for evere point we have 1 set of equations
    
    # def CreateBMatrix(self):
    #     """
    #     Coefficient Matrix related to radial derivatives
    #     """
    #     nBlock = 0
    #     for IterZ in range(0,self.data.nAxialNodes):
    #         for IterR in range(0,self.data.nRadialNodes):
    #             #for every point on the grid, construct a matrix B, and then stack them along the diagonal. Remember than the blocks are taken going along the j (radial direction) axis and i axis later (axial direction).
    #             #this decides the flatten() method for the gradients used in ComputeHatMatrices
    #             B = np.zeros((5,5))
                
    #             B[0,0] = self.data.radialVelocity[IterZ,IterR]
    #             B[1,1] = self.data.radialVelocity[IterZ,IterR]
    #             B[2,2] = self.data.radialVelocity[IterZ,IterR]
    #             B[3,3] = self.data.radialVelocity[IterZ,IterR]
    #             B[4,4] = self.data.radialVelocity[IterZ,IterR]
                
    #             B[0,1] = self.data.density[IterZ,IterR]
    #             B[1,4] = 1/self.data.density[IterZ,IterR]
    #             B[4,1] = self.data.pressure[IterZ,IterR]*self.gmma
                
    #             self.B[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = B
    #             nBlock = nBlock+1
    
    # def CreateCMatrix(self):
    #     """
    #     Coefficient Matrix related to tangential derivatives
    #     """
    #     nBlock = 0
    #     for IterZ in range(0,self.data.nAxialNodes):
    #         for IterR in range(0,self.data.nRadialNodes):
    #             #for every point on the grid, construct a matrix C, and then stack them along the diagonal
    #             C = np.zeros((5,5))
                
    #             C[0,0] = self.data.tangentialVelocity[IterZ,IterR]
    #             C[1,1] = self.data.tangentialVelocity[IterZ,IterR]
    #             C[2,2] = self.data.tangentialVelocity[IterZ,IterR]
    #             C[3,3] = self.data.tangentialVelocity[IterZ,IterR]
    #             C[4,4] = self.data.tangentialVelocity[IterZ,IterR]
                
    #             C[0,2] = self.data.density[IterZ,IterR]
    #             C[2,4] = 1/self.data.density[IterZ,IterR]
    #             C[4,2] = self.data.pressure[IterZ,IterR]*self.gmma
                
    #             self.C[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = C
    #             nBlock = nBlock+1
    
    # def CreateEMatrix(self):
    #     """
    #     Coefficient Matrix related to axial derivatives
    #     """
    #     nBlock = 0
    #     for IterZ in range(0,self.data.nAxialNodes):
    #         for IterR in range(0,self.data.nRadialNodes):
    #             #for every point on the grid, construct a matrix E, and then stack them along the diagonal
    #             E = np.zeros((5,5))
                
    #             E[0,0] = self.data.axialVelocity[IterZ,IterR]
    #             E[1,1] = self.data.axialVelocity[IterZ,IterR]
    #             E[2,2] = self.data.axialVelocity[IterZ,IterR]
    #             E[3,3] = self.data.axialVelocity[IterZ,IterR]
    #             E[4,4] = self.data.axialVelocity[IterZ,IterR]
                
    #             E[0,3] = self.data.density[IterZ,IterR]
    #             E[3,4] = 1/self.data.density[IterZ,IterR]
    #             E[4,3] = self.data.pressure[IterZ,IterR]*self.gmma
                
    #             self.E[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = E
    #             nBlock = nBlock+1
    
    # def CreateRMatrix(self):
    #     """
    #     Coefficient matrix related to the mean flow known terms. To be implemented correctly in the general case. Now it is hardcoded
    #     for the uniform acoustic duct.
    #     """
    #     # print('attention because in this matrix, for the real case there are the gradients of the mean flow, that for the moment are not implemented, since the acoustic duct is a uniform flow. the zeros are hardcoded, so remember to implement the correct thing')
    #     nBlock = 0
    #     for IterZ in range(0,self.data.nAxialNodes):
    #         for IterR in range(0,self.data.nRadialNodes):
    #             #for every point on the grid, construct a matrix R, and then stack them along the diagonal
    #             R = np.zeros((5,5))
                
    #             R[0,0] = self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR]
    #             R[0,1] = self.data.density[IterZ,IterR]/self.data.r[IterR] + 0
    #             R[0,2] = 0
    #             R[0,3] = 0 
    #             R[0,4] = 0 
    #             R[1,0] = -1/(self.data.density[IterZ,IterR]**2)*0
    #             R[1,1] = 0
    #             R[1,2] = -2*self.data.tangentialVelocity[IterZ,IterR]*0
    #             R[1,3] = 0
    #             R[1,4] = 0
    #             R[2,0] = 0
    #             R[2,1] = 0 
    #             R[2,2] = self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR]
    #             R[2,3] = 0
    #             R[2,4] = 0
    #             R[3,:] = [0,0,0,0,0]
    #             R[4,0] = 0
    #             R[4,1] = self.data.pressure[IterZ,IterR]*self.gmma/self.data.r[IterR]
    #             R[4,2] = 0
    #             R[4,3] = 0
    #             R[4,4] = self.gmma*(0+0+self.data.radialVelocity[IterZ,IterR]/self.data.r[IterR])
                
    #             self.R[(nBlock*5):(nBlock+1)*5, (nBlock*5):(nBlock+1)*5] = R
    #             nBlock = nBlock+1
    
    # def CreateSMatrix(self):
    #     """
    #     Coefficient matrix related to the body force terms. To be implemented yet
    #     """
    #     self.S = np.ones((self.nPoints*5, self.nPoints*5)) #A is the diagonal matrix
    
    # def CreateAllPhysicalMatrices(self):
    #     """
    #     Compute all the matrices together. The boundaries points will need to be treated later, modifying the corresponding equations
    #     """
    #     self.CreateAMatrix()
    #     self.CreateBMatrix()
    #     self.CreateCMatrix()
    #     self.CreateEMatrix()
    #     self.CreateRMatrix()
    #     self.CreateSMatrix()
    
    # def GlobalStabilityMatrix(self, omega):
    #     """
    #     Giving a complex eigenfrequency, it returns the global instability matrix (for the moment without caring about the boundary nodes)
    #     radialDiff and axialDiff will give the elements of radial and axial differentiations using chebyshev gauss-lobatto method.
        
    #     It needs to be implemented
    #     """
    #     #Q = 1j*omega*self.A + radialDiff(self.B) + 1j*m*self.Cr + axialDiff(E) + self.R + self.S
    #     # return Q
    
    def ComputeSpectralGrid(self):
        """
        it creates a new grid object inside the Sun Model Object, which has the same info of the original grid, but stored 
        in the computational grid for spectral differentiation
        """
        self.dataSpectral = self.data.PhysicalToSpectralData()
    
    def ShowPhysicalGrid(self, save_filename=None):
        """
        it shows the physical grid points, with different colors for the different parts of the domain
        """
        self.data.ShowGrid()
        plt.title('physical grid')
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'.pdf',bbox_inches='tight')
    
    def ShowSpectralGrid(self, save_filename=None):
        """
        it shows the physical grid points, with different colors for the different parts of the domain
        """
        self.dataSpectral.ShowGrid()
        plt.title('spectral grid')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'.pdf',bbox_inches='tight')
    
    def ComputeJacobianSpectral(self):
        """
        The Jacobian for the spectral grid as a function of the physical grid cordinates is implemented here. 
        It computes the transformation derivatives for every grid point, and stores the value at the node level.
        NOTE: it could be needed to do it on a fine mesh, and then using the results on the coarser mesh, since the gradients are obtained 
        2nd order central finite differences
        """
        Z = self.data.zGrid
        R = self.data.rGrid
        X = self.dataSpectral.zGrid
        Y = self.dataSpectral.rGrid
        self.dxdz, self.dxdr, self.dydz, self.dydr = JacobianTransform(X,Y,Z,R)
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                #add the gradients information to every node
                self.data.dataSet[ii,jj].AddJacobianGradients(self.dxdz[ii,jj], self.dxdr[ii,jj], self.dydz[ii,jj], self.dydr[ii,jj])
        
    
    def ComputeJacobianPhysical(self):
        """
        The Jacobian for the physical grid as a function of the spectral grid cordinates is implemented here. 
        It computes the transformation derivatives for every grid point, and stores the value at the node level.
        NOTE: it could be needed to do it on a fine mesh, and then using the results on the coarser mesh, since the gradients are obtained 
        2nd order central finite differences.
        NOTE 2: this approach is wrong if the spectral system of cordinates is not cartesian!
        """
        #grids
        Z = self.data.zGrid
        R = self.data.rGrid
        X = self.dataSpectral.zGrid
        Y = self.dataSpectral.rGrid
        self.dzdx, self.dzdy, self.drdx, self.drdy = JacobianTransform(Z,R,X,Y)
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                #add the inverse gradients information to every node
                self.data.dataSet[ii,jj].AddInverseJacobianGradients(self.dzdx[ii,jj], self.dzdy[ii,jj], self.drdx[ii,jj], self.drdy[ii,jj])
        
        
    
    def ShowJacobianSpectralAxis(self, formatFig=(10,6)):
        """
        Show the spectral cordinates gradients info as a function of the spectral grid cordinates.
        """
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.zGrid, self.dataSpectral.rGrid, c=self.dxdz)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\frac{\partial \xi}{\partial z}$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \xi}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.zGrid, self.dataSpectral.rGrid, c=self.dxdr)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial r}$')
        cb.set_label(r'$\frac{\partial \xi}{\partial r}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.zGrid, self.dataSpectral.rGrid, c=self.dydz)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial z}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial z}$')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.dataSpectral.zGrid, self.dataSpectral.rGrid, c=self.dydr)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial r}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial r}$')
        
    def ShowJacobianPhysicalAxis(self, save_filename=None, formatFig=(10,6)):
        """
        Show the spectral cordinates gradients info as a function of the physical grid cordinates.
        """
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.zGrid, self.data.rGrid, c=self.dxdz)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        plt.title(r'$\frac{\partial \xi}{\partial z}$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \xi}{\partial z}$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'_1.pdf',bbox_inches='tight')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.zGrid, self.data.rGrid, c=self.dxdr)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial r}$')
        cb.set_label(r'$\frac{\partial \xi}{\partial r}$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'_2.pdf',bbox_inches='tight')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.zGrid, self.data.rGrid, c=self.dydz)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial z}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial z}$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'_3.pdf',bbox_inches='tight')
        
        plt.figure(figsize=formatFig)
        plt.scatter(self.data.zGrid, self.data.rGrid, c=self.dydr)
        plt.xlabel(r'$Z$')
        plt.ylabel(r'$R$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial r}$')
        cb.set_label(r'$\frac{\partial \eta}{\partial r}$')
        if save_filename!=None:
            plt.savefig(folder_name + save_filename+'_4.pdf',bbox_inches='tight')

    
    # def ComputeHatMatrices(self):
    #     """
    #     compute and store at the node level the matrices \hat{B}, \hat{E}.
    #     """
    #     Bhat = np.zeros(self.B.shape) #physical B
    #     Ehat = np.zeros(self.B.shape) #physical E
    #     Ni = Bhat.shape[0]
    #     Nj = Bhat.shape[1]
    #     nBlocks = Ni//5
        
    #     #flatten the gradients to be used later in the matrix multiplications. I am unrolling along the column (Fortran style), because the matrices were built going along the j axis before than the i axis.
    #     #you should check this in a later stage of the process, during code debugging and testing
    #     dxdz, dxdr, dydz, dydr = self.dxdz.flatten(order='F'), self.dxdr.flatten(order='F'), self.dydz.flatten(order='F'), self.dydr.flatten(order='F') 
        
    #     if nBlocks!=self.nPoints:
    #         raise ValueError('Error in Compute Modified B matrix method')    
            
    #     for blockIt in range(0,nBlocks):
    #         Bhat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdr[blockIt] \
    #                                                                     + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dxdz[blockIt]
    #         Ehat[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5] = self.B[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydr[blockIt] \
    #                                                                     + self.E[(blockIt*5):(blockIt+1)*5,(blockIt*5):(blockIt+1)*5]*dydz[blockIt]
    #     self.Bhat = Bhat
    #     self.Ehat = Ehat
    
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
                self.chi[ii,jj] = np.min(S)/np.max(S)
    
    
    def PlotInverseConditionNumber(self, save_filename=None, formatFig=(10,6)):
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
        if save_filename!=None:
            plt.savefig(folder_name + save_filename +'.pdf' ,bbox_inches='tight')
                
    def AddAMatrixToNodes(self, omega):
        """
        compute and store at the node level the A matrix, already multiplied by j*omega. Ready to be used in the final system of eqs.
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                A = np.eye(5)
                self.data.dataSet[ii,jj].AddAMatrix(A, omega)
                
    def AddBMatrixToNodes(self):
        """
        compute and store at the node level the B matrix, needed to compute \hat{B}
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                B = np.zeros((5,5))
                
                B[0,0] = self.data.dataSet[ii,jj].GetRadialVelocity()
                B[1,1] = self.data.dataSet[ii,jj].GetRadialVelocity()
                B[2,2] = self.data.dataSet[ii,jj].GetRadialVelocity()
                B[3,3] = self.data.dataSet[ii,jj].GetRadialVelocity()
                B[4,4] = self.data.dataSet[ii,jj].GetRadialVelocity()
                
                B[0,1] = self.data.dataSet[ii,jj].GetDensity()
                B[1,4] = 1/self.data.dataSet[ii,jj].GetDensity()
                B[4,1] = self.data.dataSet[ii,jj].GetPressure()*self.gmma
                
                self.data.dataSet[ii,jj].AddBMatrix(B)
    
    def AddCMatrixToNodes(self, m=1):
        """
        compute and store at the node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                C = np.zeros((5,5))
                
                C[0,0] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[1,1] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[2,2] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[3,3] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[4,4] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                
                C[0,2] = self.data.dataSet[ii,jj].GetDensity()
                C[2,4] = 1/self.data.dataSet[ii,jj].GetDensity()
                C[4,2] = self.data.dataSet[ii,jj].GetPressure()*self.gmma
                
                self.data.dataSet[ii,jj].AddCMatrix(C, m)
    
    def AddEMatrixToNodes(self):
        """
        compute and store at the node level the E matrix, needed to compute \hat{E}
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                E = np.zeros((5,5))
                
                E[0,0] = self.data.dataSet[ii,jj].GetAxialVelocity()
                E[1,1] = self.data.dataSet[ii,jj].GetAxialVelocity()
                E[2,2] = self.data.dataSet[ii,jj].GetAxialVelocity()
                E[3,3] = self.data.dataSet[ii,jj].GetAxialVelocity()
                E[4,4] = self.data.dataSet[ii,jj].GetAxialVelocity()
                
                E[0,3] = self.data.dataSet[ii,jj].GetDensity()
                E[3,4] = 1/self.data.dataSet[ii,jj].GetDensity()
                E[4,3] = self.data.dataSet[ii,jj].GetPressure()*self.gmma
                
                self.data.dataSet[ii,jj].AddEMatrix(E)
                
    def AddRMatrixToNodes(self):
        """
        compute and store at the node level the R matrix, ready to be used in the final system of eqs.
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                R = np.zeros((5,5))
                
                R[0,0] = self.data.dataSet[ii,jj].GetRadialVelocity()/self.data.dataSet[ii,jj].r
                R[0,1] = self.data.dataSet[ii,jj].GetDensity()/self.data.dataSet[ii,jj].r
                R[0,2] = 0
                R[0,3] = 0 
                R[0,4] = 0 
                R[1,0] = 0
                R[1,1] = 0
                R[1,2] = 0
                R[1,3] = 0
                R[1,4] = 0
                R[2,0] = 0
                R[2,1] = 0 
                R[2,2] = self.data.dataSet[ii,jj].GetRadialVelocity()/self.data.dataSet[ii,jj].r
                R[2,3] = 0
                R[2,4] = 0
                R[3,:] = [0,0,0,0,0]
                R[4,0] = 0
                R[4,1] = self.data.dataSet[ii,jj].GetPressure()*self.gmma/self.data.dataSet[ii,jj].r
                R[4,2] = 0
                R[4,3] = 0
                R[4,4] = self.gmma*self.data.dataSet[ii,jj].GetRadialVelocity()/self.data.dataSet[ii,jj].r
                
                self.data.dataSet[ii,jj].AddRMatrix(R)
        
    def AddHatMatricesToNodes(self):
        """
        compute and store at the node level the \hat{B},\hat{E} matrix, needed for following multiplication with the spectral
        differential operators
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                
                Bhat = self.data.dataSet[ii,jj].B * self.data.dataSet[ii,jj].dxdr + \
                        self.data.dataSet[ii,jj].E * self.data.dataSet[ii,jj].dxdz 
                Ehat = self.data.dataSet[ii,jj].B * self.data.dataSet[ii,jj].dydr + \
                        self.data.dataSet[ii,jj].E * self.data.dataSet[ii,jj].dydz
                        
                self.data.dataSet[ii,jj].AddHatMatrices(Bhat, Ehat)
    
    def CheckGradients(self):
        """
        check if the direct and inverse transformation gives the same results. 
        NOTE: correct only if both the grids are cartesian
        """
        
        test1 = True
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                if (self.data.dataSet[ii,jj].dxdz * self.data.dataSet[ii,jj].dzdx < 0.99 and \
                    self.data.dataSet[ii,jj].dxdz * self.data.dataSet[ii,jj].dzdx > 0.01):
                    test1 = False
        
        test2 = True
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                if (self.data.dataSet[ii,jj].dxdr * self.data.dataSet[ii,jj].drdx < 0.99 and \
                    self.data.dataSet[ii,jj].dxdr * self.data.dataSet[ii,jj].drdx > 0.01):
                    test2 = False
        
        test3 = True
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                if (self.data.dataSet[ii,jj].dydr * self.data.dataSet[ii,jj].drdy < 0.99 and \
                    self.data.dataSet[ii,jj].dydr * self.data.dataSet[ii,jj].drdy > 0.01):
                    test3 = False
                    
        test4 = True
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                if (self.data.dataSet[ii,jj].dydz * self.data.dataSet[ii,jj].dzdy < 0.99 and \
                    self.data.dataSet[ii,jj].dydz * self.data.dataSet[ii,jj].dzdy > 0.01):
                    test4 = False
        
        return [test1, test2, test3, test4]
        
    def ApplySpectralDifferentiation(self):
        #we need now to apply spectral differentiation, modifying all the matrices. 
       
        #the cordinates on the spectral grid determines the spectral matrix D
        x = self.dataSpectral.z
        y = self.dataSpectral.r
        
        #compute the spectral Matrices for x and y direction with the Bayliss formulation
        Dx = ChebyshevDerivativeMatrixBayliss(x)
        Dy = ChebyshevDerivativeMatrixBayliss(y)
        
        self.Q = np.zeros((self.nPoints*5, self.nPoints*5), dtype=complex) #instantiate the full matrix, that will be filled in blocks
        node_counter = 0
        #be careful to the direction m-j. maybe it is worth to just translate everything
        for ii in range(0,self.dataSpectral.nAxialNodes):
            for jj in range(0,self.dataSpectral.nRadialNodes):
                # node_counter = jj+self.dataSpectral.nAxialNodes*ii
                B_ij = self.data.dataSet[ii,jj].Bhat #Bhat matrix of the ij node
                E_ij = self.data.dataSet[ii,jj].Ehat #Ehat matrix of the ij node
                
                #it may be correct.
                for m in range(0,self.dataSpectral.nRadialNodes):
                    tmp = Dy[m,jj]*B_ij #5x5 matrix to be added to a certain block of Q
                    row = node_counter
                    column = (m*self.dataSpectral.nRadialNodes+jj)*5
                    self.AddToQ(tmp, row, column)
                
                #apply the same in the other direction
                for n in range(0,self.dataSpectral.nAxialNodes):
                    tmp = Dx[ii,n]*E_ij #5x5 matrix to be added to a certain block of Q
                    row = node_counter
                    column = (ii*self.dataSpectral.nRadialNodes+n)*5
                    self.AddToQ(tmp, row, column)
                
                #add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii,jj].A + self.data.dataSet[ii,jj].C + self.data.dataSet[ii,jj].R + self.data.dataSet[ii,jj].S
                row = node_counter
                column = node_counter
                self.AddToQ(diag_block_ij, row, column)
                
                node_counter += 5
                
                # for n in range(0,self.dataSpectral.nRadialNodes):
                #     tmp = Dy[n,jj]*E_ij
                #     self.AddToQ(tmp,ii,jj)
        
    def AddToQ(self, block, row, column):
        """
        add elements to the stability matrix
        """
        self.Q[row:row+5, column:column+5] += block
            
        
    def AddBoundaryConditions(self):
        """
        it applies the correct set of boundary conditions to all the points marked with a boundary marker. 
        The boundary conditions can be modified in a different method
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                marker = self.data.dataSet[ii,jj].marker
                if marker == 'inlet':
                    self.data.dataSet[ii,jj].ApplyInletCondition()
                if marker == 'outlet':
                    self.data.dataSet[ii,jj].ApplyOutletCondition()
                if (marker == 'hub' or marker == 'shroud'):
                    self.data.dataSet[ii,jj].ApplyWallCondition('euler')

        
        
        
        
        
        
        
        
        
        
        
        
        