#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:21:25 2023

@author: F. Neri, TU Delft 

Construction of the matrices needed in the Spakovszky models PhD thesis.
The n=0 mode is still missing!
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


#%% FUNCTIONS AND MATRICES FOR AXIAL DUCT AND GAP SPACE
def Tax_n(x,s,n,Vx,Vy,theta=0):
    """
    Transmission matrix for the axial duct flow dynamics:
    |dVr|              |An|
    |dVtheta| = Trad_n*|Bn|
    |dp|               |Cn|
    
    ARGUMENTS:
        x : non dimensional x-cordinate
        s : Laplace variable s=sigma-j*omega
        n : circumferential harmonic number
        Vx : non-dimensional background axial velocity
        Vy : non-dimensional background zimuthal velocity        
        theta : azimuthal cordinate (zero default)
    
    RETURN:
        Tax_n
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    Tax = np.zeros((3,3),dtype = complex)
    Tax[:,0] = np.array([1, 1j, -s/n - Vx -1j*Vy])*np.exp(n*x)*np.exp(1j*n*theta)
    Tax[:,1] = np.array([1, -1j, s/n - Vx +1j*Vy])*np.exp(-n*x)*np.exp(1j*n*theta)
    Tax[0,2] = np.exp(-(s/Vx + 1j*n*Vy/Vx)*x)*np.exp(1j*n*theta)
    Tax[1,2] = (-s*1j/Vx/n+Vy/Vx)*np.exp(-(s/Vx + 1j*n*Vy/Vx)*x)*np.exp(1j*n*theta)
    Tax[2,2] = 0
    return Tax



def Bgap_n(x1,x2,s,n,Vx,Vy,theta=0):
    """
    Transmission matrix for an axial gap between two blade rows:
    |dVr|              |An|
    |dVtheta| = Trad_n*|Bn|
    |dp|               |Cn|
    
    ARGUMENTS:
        x1 : non dimensional cordinate of the first row trailing edge
        x2 : non dimensional cordinate of the second row leading edge
        s : Laplace variable s=sigma-j*omega
        n : circumferential harmonic number
        Vx : non-dimensional background axial velocity in the gap space
        Vy : non-dimensional background azimuthal velocity in the gap space        
        theta : azimuthal cordinate (zero default)
    
    RETURN:
        Bgap_n
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    Bgap = np.zeros((3,3),dtype = complex)
    m1 = Tax_n(x2, s, n, Vx, Vy, theta=theta)
    m2 = np.linalg.inv(Tax_n(x1, s, n, Vx, Vy, theta=theta))
    Bgap = np.matmul(m1,m2)
    return Bgap



#%% FUNCTIONS AND MATRICES FOR SWIRLING FLOWS
def Rad_fun(r,r0,n,s,Q,GAMMA):
    """
    Radial functions needed to construct the matrix for the swirling flow
    ARGUMENTS;
        r : non dimensional radius
        r0 : non dimensional radius location where initial conditions are specified
        n : circumferential harmonic number
        s : Laplace variable s=sigma+j*omega
        Q : source term of the swirling flow
        GAMMA : rotational term of the swirling flow

    RETURNS:
        y[0] = Rn
        y[1] = dRn_dr
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    N = 5000
    x = np.linspace(r0,r,N+1)
    def fp(x):
        return np.exp(-1j*n*GAMMA/Q*np.log(x) - s/2/Q*x**2)*x**(+n+1)
    def fn(x):
        return np.exp(-1j*n*GAMMA/Q*np.log(x) - s/2/Q*x**2)*x**(-n+1)
    
    fn = fn(x) #positive integrand
    fp = fp(x) #negative integrand
    Fp = np.sum(fp[0:len(x)-1]+fp[1:len(x)])/2*np.sum((x[1:len(x)]-x[0:len(x)-1])) #positive intgrand integrated
    Fn = np.sum(fn[0:len(x)-1]+fn[1:len(x)])/2*np.sum((x[1:len(x)]-x[0:len(x)-1])) #positive intgrand integrated
    # dfp = (-1j*n*GAMMA/Q/r-r/Q*s)*fp[N] + (+n+1)*r**(+n)*np.exp(-1j*n*GAMMA/Q*np.log(r)-s/2/Q*r**2) #positive integrand derivative
    # dfn = (-1j*n*GAMMA/Q/r-r/Q*s)*fn[N] + (-n+1)*r**(-n)*np.exp(-1j*n*GAMMA/Q*np.log(r)-s/2/Q*r**2) #negative integrand derivative  
    Rn = r**n*Fn - r**(-n)*Fp
    Rn_prime = n*r**(n-1)*Fn + r**n*fn[N] + n*r**(-n-1)*Fp - r**(-n)*fp[N]
    # Rn_second = (n**2-n)*r**(n-2)*Fn + 2*n*r**(n-1)*fn[N] + r**n*dfn -(n**2+n)*r**(-n-2)*Fp + 2*n*r**(-n-1)*fp[N] - r**(-n)*dfp   
    return Rn, Rn_prime

def Rn_second(r,r0,n,s,Q,GAMMA):
    """
    Hard-coded, because the original version in Rad_fun doesn't convince me'
    """
    r_plus = r*1.001
    Rn_prime = Rad_fun(r, r0, n, s, Q, GAMMA)[1]
    Rn_prime_plus = Rad_fun(r_plus, r0, n, s, Q, GAMMA)[1]
    Rn_second = (Rn_prime_plus-Rn_prime)/(r_plus-r)
    return Rn_second

def Trad_n(r,r0,n,s,Q,GAMMA,theta=0):
    """
    Transmission matrix for the swirling flow dynamics:
    |dVr|          |An|
    |dVy| = Trad_n*|Bn|
    |dp|           |Cn|
    
    ARGUMENTS:
        r : non dimensional radius
        r0 : radius at which initial conditions are specified for the swirling flow
        n : circumferential harmonic number
        s : Laplace variable s=sigma+j*omega
        Q : source term of the swirling flow
        GAMMA : rotational term of the swirling flow
        theta : azimuthal cordinate
    
    RETURN:
        Trad_n
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    Trad = np.zeros((3,3),dtype = complex)
    # Trad[:,0] = np.array([1j*n*r**(n-1),
    #                       -n*r**(n-1),
    #                       1j*Q*(1-n)*r**(n-2)+(GAMMA/r+s*r/(1j*n)-1j*Q/(r*n))*n*r**(n-1)])*np.exp(1j*n*theta)
    # Trad[:,1] = np.array([1j*n*r**(-n-1),
    #                       n*r**(-n-1),
    #                       -1j*Q*(1+n)*r**(-n-2)-(GAMMA/r+s*r/(1j*n)+1j*Q/(r*n))*n*r**(-n-1)])*np.exp(1j*n*theta)
    # Trad[:,2] = np.array([1j*n*Rad_fun(r,r0,n,s,Q,GAMMA)[0]/r,
    #                       -Rad_fun(r,r0,n,s,Q,GAMMA)[1],
    #                       -(1j*Q/n)*Rad_fun(r,r0,n,s,Q,GAMMA)[2]+(GAMMA/r+s*r/(1j*n)-1j*Q/(r*n))*Rad_fun(r,r0,n,s,Q,GAMMA)[1]])*np.exp(1j*n*theta)  
    Trad[0,0] = 1j*n*r**(n-1)
    Trad[0,1] = 1j*n*r**(-n-1)
    Trad[0,2] = 1j*n*Rad_fun(r, r0, n, s, Q, GAMMA)[0]/r
    Trad[1,0] = -n*r**(n-1)
    Trad[1,1] = n*r**(-n-1)
    Trad[1,2] = -Rad_fun(r, r0, n, s, Q, GAMMA)[1]
    Trad[2,0] = 1j*Q*(1-n)*r**(n-2)+(GAMMA/r+s*r/1j/n-1j*Q/r/n)*n*r**(n-1)
    Trad[2,1] = -1j*Q*(1+n)*r**(-n-2)-(GAMMA/r+s*r/1j/n+1j*Q/r/n)*n*r**(-n-1)
    Trad[2,2] = (GAMMA/r+s*r/1j/n-1j*Q/r/n)*Rad_fun(r, r0, n, s, Q, GAMMA)[1]-1j*Q/n*Rn_second(r, r0, n, s, Q, GAMMA)
    return Trad*np.exp(1j*n*theta)



#%% FUNCTIONS AND MATRICES FOR AXIAL ROTOR AND STATOR ROWS
def Bsta_n(s,n,Vx,Vy1,Vy2,alfa1,alfa2,lambda_s,dLs_dTana,theta=0,tau_s=0):
    """
    Transmission matrix for the axial stator row dynamics:
    |dVx2|          |dVx1| 
    |dVy2| = Bsta_n*|dVy1|
    |dp2|           |dp1|
    
    ARGUMENTS:
        s : Laplace variable s=sigma-j*omega
        n : circumferential harmonic number
        Vx : non-dimensional background inlet axial velocity
        Vy1 : non-dimensional background inlet azimuthal velocity
        Vy2 : non-dimensional background outlet azimuthal velocity
        alfa1 : inlet absolute swirl angle
        alfa2 : outlet absolute swirl angle
        lambda_s : row inertia parameter
        dLs_dTana : loss derivative
        theta : azimuthal cordinate (zero default)     
        tau_s : unsteady stator loss lag parameter (zero default)
        
    RETURN:
        Bsta_n    
    """
    
    Bsta = np.zeros((3,3),dtype = complex)
    Bsta[:,0] = np.array([1, np.tan(alfa2), -lambda_s*s+dLs_dTana*np.tan(alfa1)/
                          (Vx*(1+tau_s*s))-Vy2*np.tan(alfa2)])*np.exp(1j*n*theta)
    Bsta[:,1] = np.array([0,0,-dLs_dTana/(Vx*(1+tau_s*s))+Vy1])*np.exp(1j*n*theta)
    Bsta[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Bsta



def Brot_n(s,n,Vx,Vy1,Vy2,alfa1,beta1,beta2,lambda_r,dLr_dTanb,theta=0,tau_r=0):
    """
    Transmission matrix for the axial rotor row dynamics:
    |dVx2|          |dVx1| 
    |dVy2| = Brot_n*|dVy1|
    |dp2|           |dp1|
    
    ARGUMENTS:
        s : Laplace variable s=sigma-j*omega
        n : circumferential harmonic number
        Vx : non-dimensional background axial velocity
        Vy1 : non-dimensional background inlet azimuthal velocity
        Vy2 : non-dimensional background outlet azimuthal velocity
        alfa1 : inlet absolute swirl angle
        beta1 : inlet relative swirl angle
        beta2 : outlet relative swirl angle
        lambda_r : row inertia parameter
        dLr_dTanb : loss derivative with respect to inlet beta at background operating point
        theta : azimuthal cordinate (zero default)
        tau_r : unsteady rotor loss lag parameter (zero default)
        
    RETURN:
        Brot_n
    """

    Brot = np.zeros((3,3),dtype = complex)
    Brot[:,0] = np.array([1, np.tan(beta2), np.tan(beta2)-np.tan(alfa1)-lambda_r*(s+1j*n)+dLr_dTanb*
                          np.tan(beta1)/(Vx*(1+tau_r*(s+1j*n)))-Vy2*np.tan(beta2)])*np.exp(1j*n*theta)
    Brot[:,1] = np.array([0,0,-dLr_dTanb/(Vx*(1+tau_r*(s+1j*n)))+Vy1])*np.exp(1j*n*theta)
    Brot[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Brot



#%% FUNCTIONS AND MATRICES FOR RADIAL IMPELLER AND DIFFUSER
def Bimp_n(s,n,Vx1,Vr2,Vy1,Vy2,alfa1,beta1,beta2,r1,r2,rho1,rho2,A1,A2,s_i,dLi_dTanb,theta=0,tau_i=0):
    """
    Transmission matrix for radial impeller:
    |dVr2|          |dVx1| 
    |dVy2| = Bimp_n*|dVy1|
    |dp2|           |dp1|
    
    ARGUMENTS:
        s : Laplace variable s=sigma-j*omega
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
        theta : azimuthal cordinate (zero default)
        tau_i : unsteady impeller loss lag parameter (zero default)

    RETURN:
        Bimp_n
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



def Bdif_n(s,n,Vr1,Vr2,Vy1,Vy2,alfa1,beta1,alfa2,r1,r2,rho1,rho2,A1,A2,s_dif,dLd_dTana,theta=0,tau_d=0):
    """
    Transmission matrix for bladed radial diffuser:
    |dVr2|          |dVr1| 
    |dVy2| = Bdif_n*|dVy1|
    |dp2|           |dp1|
    
    ARGUMENTS:
        s : Laplace variable s=sigma-j*omega
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
        theta : azimuthal cordinate (zero default)
        tau_d : unsteady impeller loss lag parameter (zero default)
    
    RETURN:
        Bdif_n
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    AR = rho2*A2/(rho1*A1) #aspect ratio
    lambda_dif = s_dif*AR*np.log(AR)/(AR-1) #inertia parameter
    Bdif = np.zeros((3,3),dtype = complex)
    Bdif[:,0] = np.array([1/AR, np.tan(alfa2)/AR, -(lambda_dif*s+Vr2+Vy2*np.tan(alfa2))/AR+
                          Vr1+dLd_dTana*np.tan(alfa1)/(Vr1*(1+tau_d*s))])*np.exp(1j*n*theta)
    Bdif[:,1] = np.array([0,0,-dLd_dTana/(Vr1*(1+tau_d*s))+Vy1])*np.exp(1j*n*theta)
    Bdif[:,2] = np.array([0,0,1])*np.exp(1j*n*theta)
    return Bdif



def Bvlsd_n(s,n,r1,r2,r0,Q,GAMMA,theta=0):
    """
    Transmission matrix for vaneless diffuser:
    |dVr2|           |dVr1| 
    |dVy2| = Bvlsd_n*|dVy1|
    |dp2|            |dp1|
    
    ARGUMENTS:
        s : Laplace variable s=sigma-j*omega
        n : circumferential harmonic number
        r1 : inlet non dimensional radius
        r2 : outlet non dimensional radius
        r0 : reference radius for Trad computation
        Q : non dimensional source term
        GAMMA : non dimensional rotational term of the potential flow
        theta : azimuthal cordinate (zero default)

    
    RETURN:
        Bvlsd_n
    """
    
    M_2 = Trad_n(r2, r0, n, s, Q, GAMMA, theta=theta)
    M_1 = np.linalg.inv(Trad_n(r1, r0, n, s, Q, GAMMA, theta=theta))
    Bvlsd = np.matmul(M_2,M_1)*np.exp(1j*n*theta)
    return Bvlsd



#%% Methods to find the complex roots of a complex function
def Shot_Gun(complex_function, domain, n_grid=[1,1], n=1, N=30, tol=1e-6, attempts=30):
    """
    Shot-gun method taken from Spakozvzski PhD thesis, needed to compute the complex zeros of a complex function.
    
    ARGUMENTS
        complex_function : pointer to the function that we want to find the roots
        domain : domain where we look for poles, in format [x_min, x_max, y_min, y_max]
        n_grid : number of intervals in x and y in the complex domain, in format [n_x, n_y]. Default [1,1]
        n : circumferential harmonic, needed from the complex function (1 as default)
        N : number of shots per round (default=30)
        tol : tolerance for the point to be a pole (default 1e-6)
        attempts : number of attempts in the same zone in order to find different poles in the same zone (default=30)
    
    RETURN:
    poles : array of poles (complex type)
    """
    if n==0:
        raise Exception("Sorry, the n=0 mode is still not implemented. Use n!=0")
    print('-----------------------------------------------------------------------')
    print('SHOT GUN METHOD CALLED')
    left_lim = domain[0] #left border of the domain
    right_lim = domain[1] #right border of the domain
    down_lim = domain[2] #lower border
    upper_lim = domain[3] #upper border
    lx = (right_lim-left_lim)/(n_grid[0]) #interval step in x direction
    ly = (upper_lim-down_lim)/(n_grid[1]) #interval step in y direction
    N0 = N #backup values
    lx0 = lx #backup value
    ly0 = ly #backup value
    s_real = np.linspace(left_lim+lx/2, right_lim-lx/2, n_grid[0]) #real value of points
    s_imag = np.linspace(down_lim+ly/2, upper_lim-ly/2, n_grid[1]) #imaginary value of points
    mu=3 #relaxation coefficient. It is equal to 3 in the thesis
    
    pole_list = [] #initialize pole list
    for ii in range(len(s_real)):
        for jj in range(len(s_imag)):
            print('-------------------------------------------')
            print('Zone centered in s = (' + str(s_real[ii]) + ','+str(s_imag[jj])+'j)')
            for rounds in range(0,attempts):    
                #for every round in the same zone initialize the parameters
                s = s_real[ii]+1j*s_imag[jj]
                N = N0
                lx = lx0
                ly = ly0
                while (N > 0):
                    s_points = np.zeros(N,dtype=complex)
                    J_points = np.zeros(N)
                    error_points = np.zeros(N)
                    for kk in range(0,N):    
                        dx = np.random.uniform(-lx/2, lx/2) #random deltaX from the shot point
                        dy = np.random.uniform(-ly/2, ly/2) #random deltaY from the shot point
                        s_points[kk] = s+dx+1j*dy #random points where determinante will be computed 
                        error_points[kk] = np.abs(complex_function(s_points[kk],n))
                        if error_points[kk]<1e-6:
                            pole_list.append(s_points[kk])
                            J_points[kk] = 1000
                        else:
                            J_points[kk] = (error_points[kk])**(-2) #errors associated to every random point
                    CG = np.sum(s_points*J_points)/np.sum(J_points) #center of gravity of points
                    min_pos = np.argmin(error_points) #index of the best point
                    min_error = error_points[min_pos] #error of the best point
                    s = s_points[min_pos] #update central location for new loop
                    lx = mu*np.abs((CG-s_points[min_pos]).real)
                    ly = mu*np.abs((CG-s_points[min_pos]).imag)
                    R = mu*np.abs(CG-s_points[min_pos]) #update radius for new loop
                    N = N-1 #reduce the number of points for the next round  
                
                if min_error < tol:
                    copy = False #assuming initially that this pole is not a copy, see if it is
                    for k in pole_list:
                        distance = np.abs(s-k)
                        if distance < tol:
                            copy = copy or True
                        else:
                            copy = copy or False
                    if copy==False:
                        pole_list.append(s)
                        print('append pole..')

    poles = np.array(pole_list)    
    print('-------------------------------------------')
    print('SHOT GUN EXIT SUCCESSFUL')
    print('-----------------------------------------------------------------------')
    return poles



