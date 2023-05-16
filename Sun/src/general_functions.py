#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:22:40 2023
@author: F. Neri, TU Delft
"""
import numpy as np 


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
            #the points are considered watching at at indexes where i increase towards the top, and j increase towards the right.
            
            if (ii==0 and jj==0): #lower-left corner
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj]) / (Z[ii+1,jj]-Z[ii,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj]) / (R[ii,jj+1]-R[ii,jj])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj]) / (Z[ii+1,jj]-Z[ii,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj]) / (R[ii,jj+1]-R[ii,jj])
            elif (ii==Nz-1 and jj==0): #upper-left corner
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj]) / (Z[ii,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj]) / (R[ii,jj+1]-R[ii,jj])
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj]) / (Z[ii,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj]) / (R[ii,jj+1]-R[ii,jj])
            elif (ii==0 and jj==Nr-1): #bottom-right corner
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj]) / (Z[ii+1,jj]-Z[ii,jj])
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1]) / (R[ii,jj]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj]) / (Z[ii+1,jj]-Z[ii,jj])
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1]) / (R[ii,jj]-R[ii,jj-1])
            elif (ii==Nz-1 and jj==Nr-1): #upper-right corner
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj]) / (Z[ii,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1]) / (R[ii,jj]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj]) / (Z[ii,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1]) / (R[ii,jj]-R[ii,jj-1])
            elif (ii==0): #bottom side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii,jj])   / (Z[ii+1,jj]-Z[ii,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii,jj])   / (Z[ii+1,jj]-Z[ii,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
            elif (ii==Nz-1): #top side
                dxdz[ii,jj] = (X[ii,jj]-X[ii-1,jj])   / (Z[ii,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii,jj]-Y[ii-1,jj])   / (Z[ii,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
            elif (jj==0): #left side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj])   / (R[ii,jj+1]-R[ii,jj])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj])   / (R[ii,jj+1]-R[ii,jj])
            elif (jj==Nr-1): #right side
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj]-X[ii,jj-1])   / (R[ii,jj]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj]-Y[ii,jj-1])   / (R[ii,jj]-R[ii,jj-1])
            else: #all ohter internal points
                dxdz[ii,jj] = (X[ii+1,jj]-X[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dxdr[ii,jj] = (X[ii,jj+1]-X[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
                dydz[ii,jj] = (Y[ii+1,jj]-Y[ii-1,jj]) / (Z[ii+1,jj]-Z[ii-1,jj])
                dydr[ii,jj] = (Y[ii,jj+1]-Y[ii,jj-1]) / (R[ii,jj+1]-R[ii,jj-1])
    
    return dxdz, dxdr, dydz, dydr




def ChebyshevDerivativeMatrix(x):
    """
    Define the first order derivative Chebyshev matrix, where x is the array of Gauss-Lobatto points. Expression from 
    Peyret book, page 50
    """
    N = len(x) #dimension of the square matrix
    D = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            xi = np.cos(np.pi*i/(N-1))
            xj = np.cos(np.pi*j/(N-1))
            
            #select the right ci, cj
            if i==0 or i==N-1:
                ci = 2
            else:
                ci = 1
            if j==0 or j==N-1:
                cj = 2
            else:
                cj = 1
            
            #matrix coefficients
            if (i!=j):
                D[i,j] = (ci/cj) * (-1)**(i+j) / (xi-xj)
            elif (i==j and i>0 and i<N-1):
                D[i,j] = -xi/2/(1-xi**2)
            elif (i==0 and j==0):
                D[i,j] = (2*(N**2)+1)/6
            elif(i==N-1 and j==N-1):
                D[i,j] = -(2*(N**2)+1)/6
            else:
                raise ValueError('Some mistake in the computation of the matrix')
            
    return D

def ChebyshevDerivativeMatrixBayliss(x):
    """
    Define the first order derivative Chebyshev matrix, where x is the array of Gauss-Lobatto points. Expression from 
    Peyret book, page 50. Bayliss formulation for the diagonal term, as suggested by Peyret to fix the extremes
    """
    N = len(x) #dimension of the square matrix
    D = ChebyshevDerivativeMatrix(x) #basic
    for i in range(0,N):
        row_tot = np.sum(D[i,:]) - D[i,i] #sum of the terms out of the diagonal
        D[i,i] = -row_tot #to make the sum of the elements in a row = 0 for every row
    return D




def Refinement(x, add_points):
    """
    it returns a refined array, which has an additional number add_points of equispaced points in every interval of the original array.
    """
    refined_x = np.array(())
    n = len(x)
    for i in range(0,n - 1):
        refined_x = np.append(refined_x,x[i]) #insert the original point
        tmp_cord = np.linspace(x[i],x[i+1],add_points+2)
        tmp_cord = tmp_cord[1:-1]
        refined_x = np.append(refined_x,tmp_cord)
    refined_x = np.append(refined_x,x[-1])
    return refined_x



def GaussLobattoPoints(N):
    """
    it returns an array of N Gauss-Lobatto points, between -1 and +1
    """
    x = np.array(())
    for i in range(0,N):
        xnew = np.cos(i*np.pi/(N-1)) #gauss lobatto points
        x = np.append(x, xnew)
    x = np.flip(x)
    return x
        
      
for i in range(0,5):
    print(i)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        