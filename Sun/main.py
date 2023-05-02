#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:58:06 2023

@author: fneri
read the data from paraview csv output file. Still need to understand how the geometry works, in order to do circumferential average
"""
#imports
from Compressor import Compressor


# Instantiate a compressor object with a compression ratio of 0.5
# my_compressor = Compressor(x,y,z,rho,p1/rho,p2/rho,p3/rho,p)

my_compressor = Compressor('data/eckardt_impeller.csv')
# my_compressor.AddDataSetZone('data/eckardt_inlet.csv')
# my_compressor.AddDataSetZone('data/eckardt_outlet.csv')


my_compressor.scatterPlot3D('theta')
my_compressor.scatterPlot3D('density')
my_compressor.scatterPlot3D('radial')
my_compressor.scatterPlot3D('tangential')
my_compressor.scatterPlot3D('axial')
my_compressor.scatterPlot3D('pressure')

my_compressor.scatterPlot3DFull(20, 20, field='pressure',slices=500)


my_compressor.UnstructuredCircumferentialAverage(50, 50)
my_compressor.scatterPlot2D('density', size=50)
my_compressor.scatterPlot2D('radial', size=50)
my_compressor.scatterPlot2D('tangential', size=50)
my_compressor.scatterPlot2D('axial', size=50)
my_compressor.scatterPlot2D('pressure', size=50)





