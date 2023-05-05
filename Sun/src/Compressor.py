#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:11:59 2023
@author: F. Neri, TU Delft

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Grid import Node

class Compressor:
    """
    Class of compressor, which contains everything useful for its data processing
    """
    
    def __init__(self, dataFile):
        """
        Read the unstructured data from an external file and save the value in the various attributes of the Compressor.
        
        ARGUMENTS:
            datafile : input file, on which the CFD data are stored
        
        """
        if dataFile[-3:]=='csv':
            #block to parse the data in a csv datafile containing the unstructured CFD results from a steady Paraview csv export
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
            
        else:
            raise TypeError('No compatible datafile has been provided. Provide a valid file containing the unstructured CFD results')
        
    def AddDataSetZone(self, dataFile):
        """
        It adds another dataset to the previously created one. For example you can add the inlet data zone to the impeller zone.
        It simply concatenates the arrays, it doesn't keep track of the different zone names.
        
        ARGUMENTS:
            datafile : input file, on which the CFD data are stored
                    
        """
        if dataFile[-3:]=='csv':
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
        else:
            raise TypeError('No compatible datafile has been provided. Provide a valid file containing the unstructured CFD results')
         
     
        
    def PrintInfo(self):
        """
        Function to print something important of the compressor. To be implemented..
        """
        print('Add important info in a second stage')
    
    
    
    def UnstructuredCircumferentialAverage(self, Nz, Nr):
        """
        It computes the circumferential average of the unstructured flow field. It projects the 3D data in a (z,r) plane and then
        average over rectangles in that domain. It should be improved later with Parablade methods, able to map the zone in the 
        (r,z) plane in a structured way (meridional, span).
        
        ARGUMENTS:
            Nz : number of nodes in the z-direction
            Nr : number of nodes in the r-direction
        """
        #rectangular grid, brutal
        z_min = np.min(self.z)
        z_max = np.max(self.z)
        r_min = np.min(self.r)
        r_max = np.max(self.r)
        z_k = np.linspace(z_min, z_max, Nz)
        r_k = np.linspace(r_min, r_max, Nr)
        
        #initialize the fields
        self.z_avg = np.array(())
        self.r_avg = np.array(())
        self.rho_avg = np.array(())
        self.u_avg = np.array(())
        self.v_avg = np.array(())
        self.w_avg = np.array(())
        self.ur_avg = np.array(())
        self.ut_avg = np.array(())
        self.p_avg = np.array(())
        self.M_avg = np.array(())
        
        #average over all the zones of the grid
        for ii in range(0,len(z_k)-1):
            for jj in range(0,len(r_k)-1):
                mesh_set = (self.r>=r_k[jj]) & (self.r<=r_k[jj+1]) & (self.z>=z_k[ii]) & (self.z<=z_k[ii+1]) #points in every grid zone
                z_group = self.z[mesh_set]
                if len(z_group != 0):
                    
                    #subgroup of data in every zone
                    rho_group = self.rho[mesh_set]
                    p1_group = self.p1[mesh_set]
                    p2_group = self.p2[mesh_set]
                    p3_group = self.p3[mesh_set]
                    pr_group = self.pr[mesh_set]
                    pt_group = self.pt[mesh_set]
                    p_group = self.p[mesh_set]
                    M_group = self.M[mesh_set]
                    
                    #perform the averages
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
                    
                    #write the results for every zone
                    self.z_avg = np.append(self.z_avg,z_cg)
                    self.r_avg = np.append(self.r_avg,r_cg)
                    self.rho_avg = np.append(self.rho_avg,rho_cg)
                    self.u_avg = np.append(self.u_avg,u_cg)
                    self.v_avg = np.append(self.v_avg,v_cg)
                    self.ur_avg = np.append(self.ur_avg,ur_cg)
                    self.ut_avg = np.append(self.ut_avg,ut_cg)
                    self.w_avg = np.append(self.w_avg,w_cg)
                    self.p_avg = np.append(self.p_avg,p_cg)
                    self.M_avg = np.append(self.M_avg,M_cg)



    def FindBorder(self):
        #for every z interval, take the min and the max value of the radial cordinate to extract upper and lower boundaries
        Nz = 100
        z_min = np.min(self.z)
        z_max = np.max(self.z)
        self.z_border = np.linspace(z_min, z_max, Nz)
        self.r_upper = np.array(())
        self.r_lower = np.array(())
        
        #find the upper and lower border for every z-interval
        for ii in range(0,len(self.z_border)):
            if ii<len(self.z_border)-1:
                mesh_set = (self.z>=self.z_border[ii]) & (self.z<=self.z_border[ii+1]) #points in every grid zone
            else:
                mesh_set = (self.z>=self.z_border[ii-1]) & (self.z<=self.z_border[ii]) #points in every grid zone
            r_group = self.r[mesh_set]
            self.r_upper = np.append(self.r_upper, np.max(r_group))
            self.r_lower = np.append(self.r_lower, np.min(r_group))
        print('DO NOT USE FindBorder() METHOD, IT IS NOT WELL IMPLEMENTED YET')
            
        

    def scatterPlot3D(self, field='default', formatFig=(10,6), size=1, slices=25):
        """
        Method for rendering 3D scatter plot of a certain field. If the field is not provided, it just shows the nodes
        
        ARGUMENTS:
            field : name of the field to plot
            formatFig : figsize of the plot. Default = (10,6)
            size = size of the dots in the scatter plot. Default = 1
            slices = slicing number of the full data array. Default = 25
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
        Method for rendering 2D scatter plot of a certain field. If the field is not provided, it just shows the nodes
        
        ARGUMENTS:
            field : name of the field to plot
            formatFig : figsize of the plot. Default = (10,6)
            size = size of the dots in the scatter plot. Default = 1
            slices = slicing number of the full data array. Default = 25
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
        
        ARGUMENTS:
            blades : number of blades to show
            totalBlades : total number of blades of the machine
            field : name of the field to plot
            formatFig : figsize of the plot. Default = (10,6)
            size = size of the dots in the scatter plot. Default = 1
            slices = slicing number of the full data array. Default = 50
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
            
