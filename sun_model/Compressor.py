#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:11:59 2023

@author: fneri
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Compressor:
    """
    Class of compressor, which contains everything useful for its data processing
    """
    
    def __init__(self, dataFile):
        """
        Read the unstructured data from an external file and save the value in the various attributes of the Compressor
        """
        
        my_data = pd.read_csv(dataFile, delimiter=',', encoding='utf-8')
        self.x = my_data['Points:0'].values
        self.y = my_data['Points:1'].values
        self.z = my_data['Points:2'].values
        self.r = np.sqrt(self.x**2 + self.y**2) #construct radial cordinate
        self.theta = np.arctan2(self.y, self.x)
        self.rho = my_data['Density'].values
        self.p1 = my_data['Momentum:0'].values
        self.p2 = my_data['Momentum:1'].values
        self.p3 = my_data['Momentum:2'].values
        self.p = my_data['Pressure'].values
        self.M = my_data['Mach'].values
        self.pr = self.p1*np.cos(self.theta) + self.p2*np.sin(self.theta)
        self.pt = -self.p1*np.sin(self.theta) + self.p2*np.cos(self.theta) 
        self.pz = self.p3
        self.ur = self.pr/self.rho
        self.ut = self.pt/self.rho
        self.uz = self.p3/self.rho
        
    def AddDataSetZone(self, dataFile):
        """
        It adds another dataset to the previously created one. For example you can add the inlet data zone to the impeller zone.
        It simply concatenates the arrays, it doesn't keep track of the zones'
        """
        my_data = pd.read_csv(dataFile, delimiter=',', encoding='utf-8')
        self.x = np.append(self.x, my_data['Points:0'].values)
        self.y = np.append(self.y, my_data['Points:1'].values)
        self.z = np.append(self.z, my_data['Points:2'].values)
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
        self.rho = np.append(self.rho, my_data['Density'].values)
        self.p1 = np.append(self.p1, my_data['Momentum:0'].values)
        self.p2 = np.append(self.p2, my_data['Momentum:1'].values)
        self.p3 = np.append(self.p3, my_data['Momentum:2'].values)
        self.p = np.append(self.p, my_data['Pressure'].values)
        self.M = np.append(self.M, my_data['Mach'].values)
        self.pr = self.p1*np.cos(self.theta) + self.p2*np.sin(self.theta)
        self.pt = -self.p1*np.sin(self.theta) + self.p2*np.cos(self.theta) 
        self.pz = self.p3
        self.ur = self.pr/self.rho
        self.ut = self.pt/self.rho
        self.uz = self.p3/self.rho
        
    def PrintInfo(self):
        """
        Function to print something important of the compressor. To be implemented later
        """
        print('Add important info in a second stage')
    
    def CircumferentialAverage(self, Nz, Nr):
        """
        It computes the circumferential average of the unstructured flow field. It projects the 3D data in a (z,r) plane and then
        average over rectangles in that domain. It should be improved later with Parablade methods, able to map the zone in the 
        (r,z) plane in a structured way (meridional, span).
        """
        z_min = np.min(self.z)
        z_max = np.max(self.z)
        r_min = np.min(self.r)
        r_max = np.max(self.r)
        z_k = np.linspace(z_min, z_max, Nz)
        r_k = np.linspace(r_min, r_max, Nr)
        z_avg = []
        r_avg = []
        rho_avg = []
        u_avg = []
        v_avg = []
        w_avg = []
        ur_avg = []
        ut_avg = []
        p_avg = []
        M_avg = []
        for ii in range(0,len(z_k)-1):
            for jj in range(0,len(r_k)-1):
                mesh_set = (self.r>=r_k[jj]) & (self.r<=r_k[jj+1]) & (self.z>=z_k[ii]) & (self.z<=z_k[ii+1]) #points in radial span
                z_group = self.z[mesh_set]
                if len(z_group != 0):
                    rho_group = self.rho[mesh_set]
                    p1_group = self.p1[mesh_set]
                    p2_group = self.p2[mesh_set]
                    p3_group = self.p3[mesh_set]
                    pr_group = self.pr[mesh_set]
                    pt_group = self.pt[mesh_set]
                    p_group = self.p[mesh_set]
                    M_group = self.M[mesh_set]
                    rho_cg = np.mean(rho_group)
                    u_cg = np.sum(p1_group)/np.sum(rho_group)
                    v_cg = np.sum(p2_group)/np.sum(rho_group)
                    ur_cg = np.sum(pr_group)/np.sum(rho_group)
                    ut_cg = np.sum(pt_group)/np.sum(rho_group)
                    w_cg = np.sum(p3_group)/np.sum(rho_group)
                    p_cg = np.mean(p_group)
                    M_cg = np.mean(M_group)
                    z_cg = (z_k[ii+1]+z_k[ii])/2
                    r_cg = (r_k[jj+1]+r_k[jj])/2
                    z_avg.append(z_cg)
                    r_avg.append(r_cg)
                    rho_avg.append(rho_cg)
                    u_avg.append(u_cg)
                    v_avg.append(v_cg)
                    ur_avg.append(ur_cg)
                    ut_avg.append(ut_cg)
                    w_avg.append(w_cg)
                    p_avg.append(p_cg)
                    M_avg.append(M_cg)
        self.z_avg = np.array(z_avg)
        self.r_avg = np.array(r_avg)
        self.rho_avg = np.array(rho_avg)
        self.u_avg = np.array(u_avg)
        self.v_avg = np.array(v_avg)
        self.ur_avg = np.array(ur_avg)
        self.ut_avg = np.array(ut_avg)
        self.w_avg = np.array(w_avg)
        self.p_avg = np.array(p_avg)
        self.M_avg = np.array(M_avg)

    def scatterPlot3D(self, field='default', formatFig=(10,6), size=1, slices=25):
        """
        Method for rendering 3D scatter plot of a certain field. If the field is not provided, it just shows the nodes
        """
        
        if (field=='theta' or field=='Theta' or field =='THETA'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.theta[::slices]*180/np.pi)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$\theta \ [deg]$')
        elif (field=='mach' or field=='Mach' or field =='MACH'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.M[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$M$')
        elif (field=='radial' or field=='Radial' or field =='RADIAL'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.ur[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$u_{r} \ \ [m/s]$')
        elif (field=='tangential' or field=='Tangential' or field =='TANGENTIAL'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.ut[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$u_{\theta} \ \ [m/s]$')
        elif (field=='axial' or field=='Axial' or field =='AXIAL'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.uz[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$u_{z} \ \ [m/s]$')
        elif (field=='pressure' or field=='Pressure' or field =='PRESSURE'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.p[::slices]/1e5)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$p \ \ [bar]$')
        elif (field=='density' or field=='Density' or field =='DENSITY'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size, c=self.rho[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$\rho \ \ [kg/m^3]$')
        else :
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.z[::slices]*1000, self.x[::slices]*1000, self.y[::slices]*1000, s=size)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
    
    def scatterPlot2D(self, field='default', formatFig=(10,6), size=1, slices=25):
        """
        method for rendering 2D scatter plot of a certain field. If the field is not provided, it just shows the nodes
        """
        if (field=='mach' or field=='Mach' or field =='MACH'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.M_avg)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$M$')
        elif (field=='density' or field=='Density' or field =='DENSITY'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.M_avg)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$\rho \ \ [kg/m^3]$')
        elif (field=='RADIAL' or field=='Radial' or field =='radial'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.ur_avg)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$u_{r} \ \ [m/s]$')
        elif (field=='tangential' or field=='Tangential' or field =='tangential'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.ut_avg)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$u_{\theta} \ \ [m/s]$')
        elif (field=='axial' or field=='Axial' or field =='AXIAL'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.w_avg)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$u_{z} \ \ [m/s]$')
        elif (field=='pressure' or field=='Pressure' or field =='PRESSURE'):
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size, c=self.p_avg/1e5)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
            cb = plt.colorbar()
            cb.set_label(r'$p \ \ [bar]$')
        else :
            plt.figure(figsize=formatFig)
            plt.scatter(self.z_avg*1e3, self.r_avg*1e3, s=size)
            plt.xlabel('Z [mm]')
            plt.ylabel('R [mm]')
        
        
    def scatterPlot3DFull(self, blades, totalBlades, field='default', formatFig=(10,6), size=1, slices=50):
        """
        method for rendering full machine 3D scatter plot of a certain field. If the field is not provided, it just shows the nodes.
        It doesn't save new data attributes, it just shows a representation of the flowfield for a given number of blades.
        """
        if (blades>totalBlades):
            blades = totalBlades
        z = self.z
        x = self.x
        y = self.y
        theta = self.theta
        p = self.p
        mach = self.M
        for n in range(1,blades):
            z = np.append(z, self.z)
            theta = np.append(theta, self.theta+2*np.pi*n/totalBlades)
            x = np.append(x, self.r*np.cos(self.theta+2*np.pi*n/totalBlades))
            y = np.append(y, self.r*np.sin(self.theta+2*np.pi*n/totalBlades))
            mach = np.append(mach, self.M)
            p = np.append(p, self.p)
            
        if (field=='theta' or field=='Theta' or field =='THETA'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(z[::slices]*1000, x[::slices]*1000, y[::slices]*1000, s=size, c=theta[::slices]*180/np.pi)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$\theta \ [deg]$')
        elif (field=='mach' or field=='Mach' or field =='MACH'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(z[::slices]*1000, x[::slices]*1000, y[::slices]*1000, s=size, c=mach[::slices])
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$M$')
        elif (field=='pressure' or field=='Pressure' or field =='PRESSURE'):
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(z[::slices]*1000, x[::slices]*1000, y[::slices]*1000, s=size, c=p[::slices]/1e5)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            cbar = fig.colorbar(sc)
            cbar.set_label(r'$p \ \ [bar]$')
        else :
            fig = plt.figure(figsize=formatFig)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(z[::slices]*1000, x[::slices]*1000, y[::slices]*1000, s=size)
            ax.set_xlabel('Z[mm]')
            ax.set_ylabel('X [mm]')
            ax.set_zlabel('Y [mm]')
            
