#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:01:15 2023
@author: F. Neri, TU Delft
"""
import numpy as np


# def refine_coordinates(x):
#     refined_x = []
#     n = len(x)
#     for i in range(n - 1):
#         refined_x.append(x[i])
#         interval = (x[i + 1] - x[i]) / 3  # Divide interval into three parts
#         mid_point1 = x[i] + interval
#         mid_point2 = x[i] + 2 * interval
#         refined_x.extend([mid_point1, mid_point2])
#     refined_x.append(x[-1])
#     return refined_x



def Refinement(x, add_points):
    refined_x = np.array(())
    n = len(x)
    for i in range(0,n - 1):
        refined_x = np.append(refined_x,x[i]) #insert the original point
        tmp_cord = np.linspace(x[i],x[i+1],add_points+2)
        tmp_cord = tmp_cord[1:-1]
        refined_x = np.append(refined_x,tmp_cord)
    refined_x = np.append(refined_x,x[-1])
    return refined_x

# Example usage
x = np.array([0, 1, 9, 10, 100, 500])  # Original x coordinates
refined_x = Refinement(x, 1)
print(refined_x)



