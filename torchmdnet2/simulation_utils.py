#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:47:21 2021

@author: schreibef98
"""

import numpy as np

def PT_temps(T_min, T_max, n):
    """
    Creates temperatures between T_min and T_max such that the inverse temperatures 1/T are 
    equidistantly spaced.
    
    Parameters
    ----------
    T_min (float) :     lowest simulation temperature.
    T_max (float) :     highest simulation temperature
    n (int) :           number of different simulation temperatures including T_min and T_max
    
    Output:
    ------
    T (np.array) :      array with the correctly spaced simulation temperatures         
    """
    # create correct temperature scaling
    f = lambda n, k: (1/T_max + (n -1 -k)*((1/T_min - 1/T_max)/(n-1)))**(-1)
    T = np.array([])
    for k in range(n):
        T = np.append(T, f(n,k))
        
    # test if \Delta 1/T = constant
    inv_spacing = []
    for i in range(1, n):
        inv_spacing = np.append(inv_spacing, 1/T[i]-1/T[i-1])
    
    assert np.allclose(inv_spacing - inv_spacing[0], np.zeros_like(inv_spacing)),\
        "no equidistant spacing of 1/T"
    return T