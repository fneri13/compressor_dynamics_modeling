#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:21:25 2023

@author: fneri, TU Delft 

Construction of the matrices needed in the Spakozvsky models PhD thesis.
The n=0 modes are still missing!!!!!!
"""

import numpy as np
import scipy.integrate as integrate


#%% FUNCTIONS AND MATRICES FOR AXIAL DUCT
def Tax_n(x,s,theta,n,Vx,Vy):
    """
    Transmission matrix for the axial duct flow dynamics:
    |dVr|              |An|
    |dVtheta| = Trad_n*|Bn|
    |dp|               |Cn|
    x : non dimensional x-cordinate
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    Vx : non-dimensional background axial velocity
    Vy : non-dimensional background zimuthal velocity
    """
    Tax = np.zeros((3,3),dtype = complex)
    Tax[:,0] = np.array([1, 1j, -s/n - Vx -1j*Vy])*np.exp(n*x)*np.exp(1j*n*theta)
    Tax[:,1] = np.array([1, -1j, s/n - Vx +1j*Vy])*np.exp(-n*x)*np.exp(1j*n*theta)
    Tax[:,2] = np.array([1, -s*1j/(Vx*n) +Vy/Vx, 0])*np.exp(-s/Vx+1j*n*Vy/Vx)*np.exp(1j*n*theta)   
    return Tax



#%% FUNCTIONS AND MATRICES FOR SWIRLING FLOWS
def Rn(r,n,s,Q,GAMMA):
    """
    Integrals needed to construct the matrix for the whirling flow
    r : non dimensional radius
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    Q : source term of the swirling flow
    GAMMA : rotational term of the swirling flow
    """
    #quad wants a real argument, split the integrand betwee real and complex part
    result_real = integrate.quad(lambda x: (np.exp(-(1j*n*GAMMA*np.log(x)/Q) + 
                            s*(x**2)/(2*Q))*(r**n*x**(-n+1)-r**(-n)*x**(n+1))).real,
                            0, r)
    result_imag = integrate.quad(lambda x: (np.exp(-(1j*n*GAMMA*np.log(x)/Q) + 
                            s*(x**2)/(2*Q))*(r**n*x**(-n+1)-r**(-n)*x**(n+1))).imag,
                            0, r)
    return complex(result_real[0],result_imag[0])



def Rn_prime_r(r,n,s,Q,GAMMA):
    """
    First derivative OF Rn with respect to r, central difference scheme
    r : non dimensional radius
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    Q : source term of the swirling flow
    GAMMA : rotational term of the swirling flow
    """
    r_left = r*0.999
    r_right = r*1.001
    Rn_right = Rn(r_right,n,s,Q,GAMMA)
    Rn_left = Rn(r_left,n,s,Q,GAMMA)
    derivative = (Rn_right-Rn_left)/(2*(r_right-r_left))
    return derivative



def Rn_second_r(r,n,s,Q,GAMMA):
    """
    Second derivative of Rn with respect to r, central difference scheme
    r : non dimensional radius
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    Q : source term of the swirling flow
    GAMMA : rotational term of the swirling flow
    """
    r_left = r*0.999
    r_right = r*1.001
    Rn_right = Rn(r_right,n,s,Q,GAMMA)
    Rn_central = Rn(r,n,s,Q,GAMMA)
    Rn_left = Rn(r_left,n,s,Q,GAMMA)
    derivative = (Rn_right+Rn_left-2*Rn_central)/((r_right-r_left)**2)
    return derivative



def Trad_n(r,n,s,theta,Q,GAMMA):
    """
    Transmission matrix for the swirling flow dynamics:
    |dVr|          |An|
    |dVy| = Trad_n*|Bn|
    |dp|           |Cn|
    r : non dimensional radius
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    Q : source term of the swirling flow
    GAMMA : rotational term of the swirling flow
    """
    Trad = np.zeros((3,3),dtype = complex)
    Trad[:,0] = np.array([1j*n*r**(n-1),
                          -n*r**(n-1),
                          1j*Q*(1-n)*r**(n-2)+(GAMMA/r+s*r/(1j*n)-1j*Q/(r*n))*n*r**(n-1)])*np.exp(1j*n*theta)
    Trad[:,1] = np.array([1j*n*r**(-n-1),
                          n*r**(-n-1),
                          -1j*Q*(1+n)*r**(-n-2)-(GAMMA/r+s*r/(1j*n)+1j*Q/(r*n))*n*r**(-n-1)])*np.exp(1j*n*theta)
    Trad[:,2] = np.array([1j*n*Rn(r,n,s,Q,GAMMA)/r,
                          -Rn_prime_r(r,n,s,Q,GAMMA),
                          -(1j*Q/n)*Rn_second_r(r, n, s,Q,GAMMA)+(GAMMA/r+s*r/(1j*n)-1j*Q/(r*n))*Rn_prime_r(r, n, s,Q,GAMMA)])*np.exp(1j*n*theta)  
    return Trad



#%% FUNCTIONS AND MATRICES FOR AXIAL ROTOR AND STATOR ROWS
def Bsta_n(s,theta,n,Vx,Vy1,Vy2, alfa1, alfa2, lambda_s , dLs_dTana, tau_s=0):
    """
    Transmission matrix for the axial stator row dynamics:
    |dVx2|          |dVx1| 
    |dVy2| = Bsta_n*|dVy1|
    |dp2|           |dp1|
    n : circumferential harmonic number
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    Vx : non-dimensional background inlet axial velocity
    Vy1 : non-dimensional background inlet azimuthal velocity
    Vy2 : non-dimensional background outlet azimuthal velocity
    alfa1 : inlet absolute swirl angle
    alfa2 : outlet absolute swirl angle
    lambda_s : row inertia parameter
    tau_s : unsteadystator loss lag parameter
    dLs_dTana1 : loss derivative
    """
    Bsta = np.zeros((3,3),dtype = complex)
    Bsta[:,0] = np.array([1, np.tan(alfa2), -lambda_s*s+dLs_dTana*np.tan(alfa1)/
                          (Vx*(1+tau_s*s))-Vy2*np.tan(alfa2)])*np.exp(1j*n*theta)
    Bsta[:,1] = np.array([0,0,-dLs_dTana/(Vx*(1+tau_s*s))+Vy1])*np.exp(1j*n*theta)
    Bsta[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Bsta



def Brot_n(s,theta,n,Vx,Vy1,Vy2,alfa1,beta1,beta2,lambda_r,dLr_dTanb,tau_r=0):
    """
    Transmission matrix for the axial rotor row dynamics:
    |dVx2|          |dVx1| 
    |dVy2| = Brot_n*|dVy1|
    |dp2|           |dp1|
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    n : circumferential harmonic number
    Vx : non-dimensional background axial velocity
    Vy1 : non-dimensional background inlet azimuthal velocity
    Vy2 : non-dimensional background outlet azimuthal velocity
    alfa1 : inlet absolute swirl angle
    beta1 : inlet relative swirl angle
    beta2 : outlet relative swirl angle
    lambda_r : row inertia parameter
    tau_r : unsteady rotor loss lag parameter
    dLr_dTanb : loss derivative with respect to inlet beta at background operating point
    """
    Brot = np.zeros((3,3),dtype = complex)
    Brot[:,0] = np.array([1, np.tan(beta2), np.tan(beta2)-np.tan(alfa1)-lambda_r*(s+1j*n)+dLr_dTanb*
                          np.tan(beta1)/(Vx*(1+tau_r*(s+1j*n)))-Vy2*np.tan(beta2)])*np.exp(1j*n*theta)
    Brot[:,1] = np.array([0,0,-dLr_dTanb/(Vx*(1+tau_r*(s+1j*n)))+Vy1])*np.exp(1j*n*theta)
    Brot[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Brot



#%% FUNCTIONS AND MATRICES FOR RADIAL IMPELLER AND DIFFUSER
def Bimp_n(s,theta,n,Vx1,Vr2,Vy1,Vy2,alfa1,beta1,beta2,r1,r2,rho1,rho2,A1,A2,s_i,dLi_dTanb,tau_i=0):
    """
    Transmission matrix for radial impeller:
    |dVr2|          |dVx1| 
    |dVy2| = Bimp_n*|dVy1|
    |dp2|           |dp1|
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    n : circumferential harmonic number
    Vx1 : non-dimensional background inlet axial velocity
    Vr2 : non-dimensional background outlet radial velocity
    Vy1 : non-dimensional background inlet azimuthal velocity
    Vy2 : non-dimensional background outlet azimuthal velocity
    alfa1 : inlet absolute swirl angle
    beta1 : inlet relative swirl angle
    beta2 : outlet relative swirl angle
    r1,r2,rho1,rho2,A1,A2,s_i: radii and densities in order to compute AR aspect ratio and inertia parameter
    dLi_dTanb : loss derivative with respect to inlet beta at background operating point
    tau_i : unsteady impeller loss lag parameter
    """
    AR = rho2*A2/(rho1*A1) #aspect ratio
    lambda_i = s_i*AR*np.log(AR)/(AR-1) #inertia parameter
    Bimp = np.zeros((3,3),dtype = complex)
    Bimp[:,0] = np.array([1/AR, np.tan(beta2)/AR, (np.tan(beta2)-(r1/r2)*np.tan(alfa1)-lambda_i*(s+1j*n)-
                          Vr2-Vy2*np.tan(beta2))/AR+dLi_dTanb*np.tan(beta1)/(Vx1*(1+tau_i*(s+1j*n
                            )))+Vx1])*np.exp(1j*n*theta)
    Bimp[:,1] = np.array([0,0,-dLi_dTanb/(Vx1*(1+tau_i*(s+1j*n)))+Vy1])*np.exp(1j*n*theta)
    Bimp[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Bimp



def Bdif_n(s,theta,n,Vr1,Vr2,Vy1,Vy2,alfa1,beta1,alfa2,r1,r2,rho1,rho2,A1,A2,s_dif,dLd_dTana,tau_d=0):
    """
    Transmission matrix for bladed radial diffuser:
    |dVr2|          |dVr1| 
    |dVy2| = Bdif_n*|dVy1|
    |dp2|           |dp1|
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    n : circumferential harmonic number
    Vr1 : non-dimensional background inlet radial velocity
    Vr2 : non-dimensional background outlet radial velocity
    Vy1 : non-dimensional background inlet azimuthal velocity
    Vy2 : non-dimensional background outlet azimuthal velocity
    alfa1 : inlet absolute swirl angle
    beta1 : inlet relative swirl angle
    alfa2 : outlet absolute swirl angle
    r1,r2,rho1,rho2,A1,A2,s_dif: radii and densities in order to compute AR aspect ratio and inertia parameter
    dLd_dTana : loss derivative with respect to inlet alfa at background operating point
    tau_d : unsteady impeller loss lag parameter
    """
    AR = rho2*A2/(rho1*A1) #aspect ratio
    lambda_dif = s_dif*AR*np.log(AR)/(AR-1) #inertia parameter
    Bdif = np.zeros((3,3),dtype = complex)
    Bdif[:,0] = np.array([1/AR, np.tan(alfa2)/AR, -(lambda_dif*s+Vr2+Vy2*np.tan(alfa2))/AR+
                          Vr1+dLd_dTana*np.tan(alfa1)/(Vr1*(1+tau_d*s))])*np.exp(1j*n*theta)
    Bdif[:,1] = np.array([0,0,-dLd_dTana/(Vr1*(1+tau_d*s))+Vy1])*np.exp(1j*n*theta)
    Bdif[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Bdif



def Bvlsd_n(s,theta,n,r1,r2,Q,GAMMA):
    """
    Transmission matrix for vaneless diffuser:
    |dVr2|           |dVr1| 
    |dVy2| = Bvlsd_n*|dVy1|
    |dp2|            |dp1|
    s : Laplace variable s=sigma+j*omega
    theta : azimuthal cordinate
    n : circumferential harmonic number
    r1 : inlet non dimensional radius
    r2 : outlet non dimensional radius
    Q : non dimensional source term
    GAMMA : non dimensional rotational term of the potential flow
    """
    M_2 = Trad_n(r2, n, s, theta, Q, GAMMA)
    M_1 = np.linalg.inv(Trad_n(r1, n, s, theta, Q, GAMMA))
    Bvlsd = np.matmul(M_2,M_1)*np.exp(1j*n*theta)
    return Bvlsd



#%% Shot gun method for finding the roots of the determinants
def shot_gun_method(complex_function, s, R, N, i, mu=3):
    """
    Shot-gun method taken from Spakozvzski PhD thesis, needed to compute the complex zeros of a complex function.
    
    ARGUMENTS
    complex_function : is the complex function that we want to find the roots
    s : is the initial guess for the complex root, around which we will shoot many random points
    R : radius around s, where we will randomly shoot
    N : number of shots per round
    i : iterator
    mu : relaxation coefficient for the radius. 3 is the value sueggested in the thesis
    
    RETURN:
    CG : pole location (or None if no poles are found there)
    """
    i = i+1
    tol = 1e-3
    s0 = s #initial location for pole search
    R0 = R #initial radius for pole search
    N0 = N #initial number of shot points
    CG_err = np.abs(complex_function(s)) #error of the initial central location
    
    #Run the loop until we have 1 point, the radius is larger than zero, and the error is above a threshold
    while (N > 1 and R > 0 and CG_err>tol):
        s_points = np.zeros(N,dtype=complex)
        J_points = np.zeros(N)
        error_points = np.zeros(N)
        for kk in range(0,N):    
            r = np.random.uniform(0, R) #random distance from the shot point
            phi = np.random.uniform(0, 2*np.pi) #random phase angle from the shot point
            s_points[kk] = s+r*np.exp(1j*phi) #random points where determinante will be computed 
            error_points[kk] = np.abs(complex_function(s_points[kk]))
            J_points[kk] = (error_points[kk])**(-2) #errors associated to every random point
        CG = np.sum(s_points*J_points)/np.sum(J_points) #center of gravity of points
        min_pos = np.argmin(error_points) #index of the best point
        min_error = error_points[min_pos] #error of the best point
        CG_err = np.abs(complex_function(CG)) #error of the center of gravity
        s = CG #update central location for new loop
        R = R-mu*np.abs(CG-s_points[min_pos]) #update radius for new loop
        N = N-1 #reduce the number of shots
    
    #decide what to do when the previous while loop breaks out:
    if CG_err<tol and N>1 and R>0:
        print('Shot-gun method converged!')
        return CG
    else:
        print('Shot-gun method not converging.....')
        if i<100:
            CG = shot_gun_method(complex_function,s0, R0, N0, i)
            return CG
        else: #if it doesn't find a root after 100 attempts there is no root there. (Or you have been super unlucky)
            print('No roots in this zone!')
            CG = None
            R = None
            return CG





