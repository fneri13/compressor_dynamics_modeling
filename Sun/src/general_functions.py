#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:22:40 2023
@author: F. Neri, TU Delft
"""
import numpy as np 

# GENERAL FUNCTIONS USED IN THE CODE. THEY CAN BE ACCESSED ALSO FROM OUTSIDE, NOT EXCLUSIVELY BY CLASS MEMBERS  
def JacobianTransform(X,Y,Z,R):
    """
    It computes the jacobian of the transformation between two sets of cordinates.
    X,Y are the spectral cordinate values
    Z,R are the physical cordinate values
    
    It returns the gradients of the spectral cordinates as a function of the physical
    dxdz, dxdr, dydz, dydr
    """
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
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/((R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj])/(1*(Z[ii+1,jj]-Z[ii,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/((R[ii,jj+1]-R[ii,jj-1]))
            elif (ii==Nz-1): #right side
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/((R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj])/(1*(Z[ii,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/((R[ii,jj+1]-R[ii,jj-1]))
            elif (jj==0): #lower side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj])/(1*(R[ii,jj+1]-R[ii,jj]))
            elif (jj==Nr-1): #upper side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1])/(1*(R[ii,jj]-R[ii,jj-1]))
            else: #internal points
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1])/(1*(R[ii,jj+1]-R[ii,jj-1]))
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj])/(1*(Z[ii+1,jj]-Z[ii-1,jj]))
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1])/(1*(R[ii,jj+1]-R[ii,jj-1]))
    
    return dxdz, dxdr, dydz, dydr




def ChebyshevDerivativeMatrixBayliss(x):
    """
    Define the first order derivative matrix, where x is the array of Gauss-Lobatto points. Given an x-array of cordinates
    (that should be located on the gauss lobatto points), it returns the first order derivative operator. It is based on the 
    Bayliss formulation, which artificially fix the diagonal elements.
    """
    N = len(x) #dimension of the square matrix
    D = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            xi = np.cos(np.pi*i/N)
            xj = np.cos(np.pi*j/N)
            # compute off-diagonal before
            if (i!=j):
                if (i==0 or i==N):
                    ci = 2
                elif (i>0 or i <N):
                    ci =1
                else:
                    raise ValueError('Error in the chebyshev derivative matrix formulation')
                
                if (j==0 or j==N):
                    cj = 2
                elif (j>0 or j<N):
                    cj =1
                else:
                    raise ValueError('Error in the chebyshev derivative matrix formulation')
                    
                D[i,j] = (ci/cj)*(-1)**(i+j)/(xi-xj)
                
    for i in range(0,N):
        tot_coeff = np.sum(D[i,:])
        D[i,i] = - tot_coeff #artifially fix the diagonal element, so that the sum of the elements in a generic row gives zero
    return D