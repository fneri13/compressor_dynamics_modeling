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
        self.gmma = 1.4 #cp/cv for standard air for the moment
    
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
                #add the gradients at every node object
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
    
    # def ComputeSVD(self, omega_domain=[-1,1,-1,1], grid_omega=[10,10]):
    #     omR_min = omega_domain[0]
    #     omR_max = omega_domain[1]
    #     omI_min = omega_domain[2]
    #     omI_max = omega_domain[3]
    #     nR = grid_omega[0]
    #     nI = grid_omega[1]
    #     omR = np.linspace(omR_min, omR_max, nR)
    #     omI = np.linspace(omI_min, omI_max, nI)
    #     self.omegaR, self.omegaI = np.meshgrid(omR, omI)
    #     self.chi = np.zeros((nR,nI))
    #     for ii in range(0,nR):
    #         for jj in range(0,nI):
    #             omega = omR[ii]+1j*omI[jj]
                
    #             #to be implemented correctly later
    #             Q = 1j*omega*self.A+self.B+self.C+self.E+self.R+self.S

    #             U,S,V = np.linalg.svd(Q)
    #             self.chi[ii,jj] = np.min(S)/np.max(S)
    
    
    # def PlotInverseConditionNumber(self, save_filename=None, formatFig=(10,6)):
    #     x = np.linspace(np.min(self.omegaR), np.max(self.omegaR))
    #     critical_line = np.zeros(len(x))
    #     plt.figure(figsize=formatFig)
    #     plt.contourf(self.omegaR, self.omegaI, self.chi)
    #     plt.plot(x, critical_line, '--r')
    #     plt.xlabel(r'$\omega_{R}$')
    #     plt.ylabel(r'$\omega_{I}$')
    #     plt.title(r'$\chi$ locus')
    #     cb = plt.colorbar()
    #     cb.set_label(r'$\chi$')
    #     if save_filename!=None:
    #         plt.savefig(folder_name + save_filename +'.pdf' ,bbox_inches='tight')
                
    def AddAMatrixToNodes(self, omega):
        """
        compute and store at the node level the A matrix, already multiplied by j*omega. Ready to be used in the final system of eqs.
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                A = np.eye(5, dtype=complex)*1j*omega
                self.data.dataSet[ii,jj].AddAMatrix(A)
                
    def AddBMatrixToNodes(self):
        """
        compute and store at the node level the B matrix, needed to compute \hat{B}
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                B = np.zeros((5,5), dtype=complex)
                
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
                C = np.zeros((5,5), dtype=complex)
                
                C[0,0] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[1,1] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[2,2] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[3,3] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                C[4,4] = self.data.dataSet[ii,jj].GetTangentialVelocity()
                
                C[0,2] = self.data.dataSet[ii,jj].GetDensity()
                C[2,4] = 1/self.data.dataSet[ii,jj].GetDensity()
                C[4,2] = self.data.dataSet[ii,jj].GetPressure()*self.gmma
                
                C = C*1j*m/self.data.dataSet[ii,jj].r
                
                self.data.dataSet[ii,jj].AddCMatrix(C)
    
    def AddEMatrixToNodes(self):
        """
        compute and store at the node level the E matrix, needed to compute \hat{E}
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                E = np.zeros((5,5), dtype=complex)
                
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
                R = np.zeros((5,5), dtype=complex)
                
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
                
    def AddSMatrixToNodes(self, BFM=None):
        """
        compute and store at the node level the S matrix, ready to be used in the final system of eqs. The matrix formulation
        depends on the selected body-force model
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                if BFM==None:
                    S = np.zeros((5,5), dtype=complex)
                self.data.dataSet[ii,jj].AddSMatrix(S)
        
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
        NOTE: it is correct only if both the grids are cartesian, otherwise we cannot compute the gradients on a
        curvilinear grid by means of finite differences.
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
        
    def ApplySpectralDifferentiation(self, verbose=False):
       """
       This method applies Chebyshev-Gauss-Lobatto differentiation method, to express the perturbation derivatives as a function
       of the perturbation at the other nodes. It saves a new matrix Q, which is the global stability matrix (nPoints*5 X nPoints*5)
       """
        
       #the cordinates on the spectral grid directions the spectral matrix Dx and Dy
       x = self.dataSpectral.z
       y = self.dataSpectral.r
        
       #compute the spectral Matrices for x and y direction with the Bayliss formulation
       Dx = ChebyshevDerivativeMatrixBayliss(x) #derivative operator in xi, Bayliss formulation
       Dy = ChebyshevDerivativeMatrixBayliss(y) #derivative operator in eta, Bayliss formulation
        
       self.Q = np.zeros((self.nPoints*5, self.nPoints*5), dtype=complex) #instantiate the full stability matrix, that will be filled block by block
        
       #differentiation of a general perturbation vector (for the node i,j) along xi  and eta
       for ii in range(0,self.dataSpectral.nAxialNodes):
            for jj in range(0,self.dataSpectral.nRadialNodes):
                B_ij = self.data.dataSet[ii,jj].Bhat #Bhat matrix of the ij node
                E_ij = self.data.dataSet[ii,jj].Ehat #Ehat matrix of the ij node
                node_counter = self.data.dataSet[ii,jj].nodeCounter #needed to keep track of the row in the stability equations. every new node, increase the row number by 5
                
                #xi differentiation. m is in the range of axial nodes, first axis of the matrix
                for m in range(0,self.dataSpectral.nAxialNodes):
                    tmp = Dx[ii,m]*B_ij #5x5 matrix to be added to a certain block of Q
                    row = node_counter*5 #this selects the correct block along i of Q
                    column = (m*self.dataSpectral.nRadialNodes + jj)*5 #it selects the correct block along the second axis the matrix
                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' %(ii,jj))
                        print('Element along i [m,j] = (%.1d,%.1d)' %(m,jj))
                        print('Derivative element [ii,m] = (%.1d,%.1d)' %(ii,m))
                        print('[row,col] = (%.1d,%.1d)' %(row,column))
                    self.AddToQ(tmp, row, column)
                    
                
                #xi differentiation. n is in the range of radial nodes, second axis of the matrix
                for n in range(0,self.dataSpectral.nRadialNodes):
                    tmp = Dy[jj,n]*E_ij #5x5 matrix to be added to a certain block of Q
                    row = node_counter*5 #this selects the correct block along i of Q
                    column = (ii*self.dataSpectral.nRadialNodes + n)*5 #this is the important point
                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' %(ii,jj))
                        print('Element along j [i,n] = (%.1d,%.1d)' %(jj,n))
                        print('Derivative element [jj,n] = (%.1d,%.1d)' %(jj,n))
                        print('[row,col] = (%.1d,%.1d)' %(row,column))
                    self.AddToQ(tmp, row, column)
                
                
        
    def AddToQ(self, block, row, column):
        """
        add elements to the stability matrix specifying the first top-left element location
        """
        self.Q[row:row+5, column:column+5] += block
            
        
    def ApplyBoundaryConditions(self):
        """
        it applies the correct set of boundary conditions to all the points marked with a boundary marker. 
        Every BC will modify the 5 equations for the respective node. For the moment only standard BC are implemented,
        but they can be extended quite easily
        """
        for ii in range(0,self.data.nAxialNodes):
            for jj in range(0,self.data.nRadialNodes):
                marker = self.data.dataSet[ii,jj].marker
                counter = self.data.dataSet[ii,jj].nodeCounter
                row = counter*5
                if marker == 'inlet':
                    self.ApplyInletCondition(row) #apply zero perturbation conditions
                elif marker == 'outlet':
                    self.ApplyOutletCondition(row) #apply zero pressure condition
                elif (marker == 'hub' or marker == 'shroud'):
                    self.ApplyWallCondition(row) #apply non-penetration condition
                elif (marker != ''):
                    raise Exception('Boundary condition unknown. Check the grid markers!')
                    
    def AddRemainingMatrices(self):
        """
        it adds the remaning block matrices to the full Q, on the diagonal blocks.
        """
        for ii in range(0,self.dataSpectral.nAxialNodes):
            for jj in range(0,self.dataSpectral.nRadialNodes):
                #add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii,jj].A + self.data.dataSet[ii,jj].C + self.data.dataSet[ii,jj].R + self.data.dataSet[ii,jj].S
                
                node_counter = self.data.dataSet[ii,jj].nodeCounter
                row = node_counter*5
                column = node_counter*5
                self.AddToQ(diag_block_ij, row, column)
    
    def ApplyInletCondition(self, row):
        """
        for the considered grid node, it substitutes the 5 equations with a zero perturbation condition
        """
        self.Q[row:row+5,:] = np.zeros(self.Q[row:row+5,:].shape, dtype=complex) #make it zero first
        self.Q[row:row+5,row:row+5] = np.eye(5, dtype=complex) #zero perturbation condition for every flow variable
        
    
    def ApplyOutletCondition(self, row):
        """
        for the considered grid node, it substitutes the pressure equation with a zero pressure condition
        """
        self.Q[row+4,:] = np.zeros(self.Q[row+4,:].shape, dtype=complex) #make first the pressure equation coefficients zero
        self.Q[row+4,row+4] = 1 #apply zero pressure condition
        
    def ApplyWallCondition(self, row, wall_normal = [1, 0, 0]):
        """
        for the considered grid node, it substitutes one of the velocity equation with a non-penetration condition
        """
        normal_wall = [1, 0, 0] #for this case specifically
        self.Q[row+1,:] = np.zeros(self.Q[row+1,:].shape, dtype=complex) #first make zero the radial velocity equation
        self.Q[row+1,row+1:row+4] = wall_normal #impose non-penetration (u*nr + v*nt + w*nz)
        

        
        
        
        
        
        
        
        
        
        
        
        
        