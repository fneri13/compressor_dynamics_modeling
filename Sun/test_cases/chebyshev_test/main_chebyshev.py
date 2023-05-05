#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:59:17 2023
@author: F. Neri, TU Delft

script to learn and compute Chebyshev polynomials
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
from scipy.special import chebyt
import sys
sys.path.insert(1, '../../src/') #to add Classes folder


#show the basis functions
x = np.linspace(-1, 1, 1000)
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title(r'Chebyshev polynomials $T_n$')
for n in np.arange(1,5):
    ax.plot(x, chebyt(n)(x), label=rf'$T_n={n}$')
plt.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T_{n}(x)$')
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])






#%% gauss and gauss-lobatto points, where the basis functions become zero and maximum:
x_g = np.array(())
x_gl = np.array(())
N = 4 #choose the order of the chebyshev polynomial
for i in range(0,N+1):
    if i<N:
        xnew = np.cos((i+0.5)*np.pi/N) #where polynomials are zero
        x_g = np.append(x_g, xnew)
    
    xnew = np.cos(i*np.pi/N) #where polynomials peak
    x_gl = np.append(x_gl, xnew)
    
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title(r'Chebyshev polynomials - Gauss/Lobatto Points')
ax.plot(x, chebyt(N)(x))
ax.plot(x_g, chebyt(N)(x_g),'o', label=r'Gauss points')
ax.plot(x_gl, chebyt(N)(x_gl),'o', label=r'Gauss-Lobatto points')
plt.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T_{'+str(N)+'}(x)$')
ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])






#%% chebyshev interpolation
from numpy.polynomial.chebyshev import chebfit, chebval
x = np.linspace(1,5,1000)
y = (x**2+x-2*np.sin(10*x))/(1-6*x+3*x) #general oscillating non-periodic function

# Define the degree of the Chebyshev series. test three values
n = [10,15,20]

# Convert the function to its Chebyshev series representation
c1 = chebfit(x, y, n[0])
c2 = chebfit(x, y, n[1])
c3 = chebfit(x, y, n[2])

fig, ax = plt.subplots(figsize=(10,7))
ax.stem(np.abs(c3), 'o')
ax.set_title('Chebyshev Spectrum')
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\hat{y}$')
ax.set_yscale('log')
ax.set_xticks(np.linspace(0,n[2],5))
ax.legend()

# Test the Chebyshev series representation by evaluating it at a new set of points
x_test = np.linspace(np.min(x), np.max(x), 1000)
y_test1 = chebval(x_test, c1)
y_test2 = chebval(x_test, c2)
y_test3 = chebval(x_test, c3)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x, y, lw=1 ,label=r'y(x)')
ax.plot(x_test, y_test1, lw=1 ,label='order %2d' %(n[0]))
ax.plot(x_test, y_test2, lw=1 ,label='order %2d' %(n[1]))
ax.plot(x_test, y_test3, lw=1 ,label='order %2d' %(n[2]))
ax.set_title('Chebyshev interpolation')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()





#%% construct the matrix Dij for the derivatives. There is some conceptual issue here
from numpy.polynomial.chebyshev import chebfit, chebval

#analytical continuous function
x = np.linspace(-1,1,1000)
def nonPeriodicFunction(x):
    return (x**2+x-3*np.sin(15*x))/(x**2+1)
    # return (x**2)
y = nonPeriodicFunction(x) #general oscillating non-periodic function


# Define the degree of the Chebyshev series
N = 20 #order of polynomial. N+1 gauss lobatto points

#obtain the points, and function values at GL points
x_GL = np.array(())
for i in range(0,N+1):
    x_GL = np.append(x_GL, np.cos(np.pi*i/(N))) #gauss lobatto points of the higher order polynonial
y_GL = nonPeriodicFunction(x_GL)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x,y,label=r'$y(x)$')
ax.plot(x_GL, y_GL,'o', label='GL points N=%2d' %(N))
ax.set_title('function')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()

# Convert the function to its Chebyshev series representation
c = chebfit(x, y, N)
y_interp = chebval(x, c)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x,y,label=r'$y(x)$')
ax.plot(x,y_interp,'s', label='interpolation')
ax.set_title('chebyshev interpolation N=%2d' %(N))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend()


def ChebyshevDerivativeMatrix(x):
    """
    Define the first order derivative matrix, where x is the array of Gauss-Lobatto points
    """
    N = len(x) #dimension of the square matrix
    D = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            xi = np.cos(np.pi*i/N)
            xj = np.cos(np.pi*j/N)
            if (i==0 and j==0):
                D[i,j] = (2*N**2+1)/6
            elif (i==j and i>=1 and i<=N-1):
                D[i,j] = -(xi)/2/(1-xi**2)
            elif (i==j and i==N):
                D[i,j] = -(2*N**2+1)/6
            elif (i >=0 and j<=N and i!=j):
                if (i==0 or i==N):
                    ci = 2
                elif (i>0 or i <N):
                    ci =1
                else:
                    raise ValueError('Some mistake in the computation of the matrix')
                
                if (j==0 or j==N):
                    cj = 2
                elif (j>0 or j<N):
                    cj =1
                else:
                    raise ValueError('Some mistake in the computation of the matrix')
                
                D[i,j] = (ci/cj)*(-1)**(i+j)/(xi-xj)
            else:
                raise ValueError('Some mistake in the computation of the matrix')
    return D

def ChebyshevDerivativeMatrixBayliss(x):
    """
    Define the first order derivative matrix, where x is the array of Gauss-Lobatto points
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
                    raise ValueError('Some mistake in the computation of the matrix')
                
                if (j==0 or j==N):
                    cj = 2
                elif (j>0 or j<N):
                    cj =1
                else:
                    raise ValueError('Some mistake in the computation of the matrix')
                    
                D[i,j] = (ci/cj)*(-1)**(i+j)/(xi-xj)
    for i in range(0,N):
        tot_coeff = np.sum(D[i,:])
        D[i,i] = - tot_coeff
            
    return D

D = ChebyshevDerivativeMatrix(x_GL) #check the sum of the rows to be zero
D2 = ChebyshevDerivativeMatrixBayliss(x_GL) #check the sum of the rows to be zero


def ChebyshevFirstOrderDerivative(y,x):
    """
    Compute the derivative using GL method, where x and y are the interpolation cordinates and values in the GL points
    """
    D = ChebyshevDerivativeMatrix(x)
    dydx = np.matmul(D,y)
    return dydx

def ChebyshevFirstOrderDerivativeBayliss(y,x):
    """
    Compute the derivative using GL method, where x and y are the interpolation cordinates and values in the GL points
    """
    D = ChebyshevDerivativeMatrixBayliss(x)
    dydx = np.matmul(D,y)
    return dydx


dydx_GL = ChebyshevFirstOrderDerivative(y_GL, x_GL) #derivative values at GL points
dydx_GL2 = ChebyshevFirstOrderDerivativeBayliss(y_GL, x_GL) #derivative values at GL points


#Compare with second order finite difference (numpy.gradient())
x_FD = np.linspace(np.min(x),np.max(x),len(x_GL)) #same amount of points of the GL differentiation
y_FD = nonPeriodicFunction(x_FD) #value of the function

dydx_FD = np.gradient(y_FD, x_FD) #value of the finite difference values

dydx = np.gradient(y,x) #analytical (refined domain) derivative

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x, dydx, label=r'$dy/dx$')
ax.plot(x_GL, dydx_GL, '--o' ,lw=1 ,label=r'$dy/dx$ GL')
ax.plot(x_GL, dydx_GL2, '--^' ,lw=1 ,label=r'$dy/dx$ GLB')
ax.plot(x_FD, dydx_FD, '--s' ,lw=1 ,label=r'$dy/dx$ 2FD')
ax.set_title('derivative N=%2d' %(N))
ax.set_xlabel(r'$x$')
# ax.set_ylim([np.min(dydx_an), np.max(dydx_an)])
ax.set_ylabel(r'$\frac{dy}{dx}$')
ax.legend()























