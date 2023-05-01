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
ax.plot(x_g, chebyt(N)(x_g),'o', label=rf'Gauss points')
ax.plot(x_gl, chebyt(N)(x_gl),'o', label=rf'Gauss-Lobatto points')
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
x = np.linspace(1,6,100)
y = (x**2+x-2*np.sin(5*x))/(1-6*x+3*x) #general oscillating non-periodic function
# y = np.sin(x)

# Define the degree of the Chebyshev series
n = 10
# Convert the function to its Chebyshev series representation
c = chebfit(x, y, n)

x_test = np.linspace(np.min(x), np.max(x), 1000)
y_test = chebval(x_test, c)

def ChebyshevDerivativeMatrix(x):
    size = len(x) #dimension of the square matrix
    D = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            if i==j:
                D[i,j] = 0
            else:
                D[i,j] = (-1)**(i+j)/2/np.sin((x[i]-x[j])/2)
    return D

def ChebyshevFirstOrderDerivative(y,x):
    D = ChebyshevDerivativeMatrix(x)
    dydx = np.matmul(D,y)
    return dydx

dydx = ChebyshevFirstOrderDerivative(y_test, x_test)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x_test, y_test, lw=1 ,label=r'$y$')
# ax.plot(x_test, dydx, lw=1 ,label=r'$dy/dx$')
ax.set_title('Chebyshev derivative')
ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.legend()
























