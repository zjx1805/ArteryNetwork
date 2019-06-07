import sys, os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.measure import label
from scipy.signal import convolve
from numpy.linalg import norm
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import networkx as nx
from scipy import interpolate
import scipy.spatial as sp
import logging
import traceback
import timeit
import time
import math
from ast import literal_eval as make_tuple
from skimage.measure import label
import subprocess
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # This import registers the 3D projection, but is otherwise unused.
import glob
import pickle
import copy
from scipy.stats import binned_statistic
from scipy.optimize import fsolve, fmin_tnc, least_squares, differential_evolution, minimize, fmin_l_bfgs_b

"""
This file contains a collection of frequently used auxiliary functions.
"""

def randomWalkBFS(G, initialVoxels, boundaryVoxels):
    """
    Traverse the graph from the specified beginning voxel(s) and bounded by `boundaryVoxels`.
    Note that this version works with voxel that is indexed by its coordinates. If the voxel is indexed otherwise, you 
    need to change the code accordingly.

    Parameters
    ----------
    G : NetworkX graph  
        The graph to be traversed.
    initialVoxels : list  
        A list of voxels to start with.
    boundaryVoxels : list  
        A list of voxels that mark the boundary.
    
    Returns
    -------
    G : NetworkX graph  
        The graph with traversal information added.
    voxelsVisited : list  
        A list of voxels that have been traversed.
    segmentIndexList : list  
        A list of segment indices that have been traversed.
    """
    initialVoxelsTuple = initialVoxels
    boundaryVoxelsTuple = boundaryVoxels
    initialVoxelsTuple = list(map(tuple, initialVoxels))
    boundaryVoxelsTuple = list(map(tuple, boundaryVoxels))
    numOfVoxelsVisited = len(initialVoxelsTuple)
    voxelsVisited = copy.deepcopy(initialVoxelsTuple) # make deep copy
    depthVoxel = 0 # increment by 1 after each voxel
    depthLevel = 0 # increment by 1 after each bifurcation
    for voxel in initialVoxelsTuple:
        G.node[voxel]['depthLevel'] = depthLevel
        G.node[voxel]['pathDistance'] = 0.0
    pool = copy.deepcopy(initialVoxelsTuple) # make deep copy
    segmentIndexList = [] # record the index of segments traversed
    while len(pool) != 0:
        poolTemp = []
        for currentVoxel in pool:
            G.node[currentVoxel]['depthVoxel'] = depthVoxel
            newVoxels = [voxel for voxel in G.neighbors(currentVoxel) if voxel not in boundaryVoxelsTuple and 'depthVoxel' not in G.node[voxel]]
            for newVoxel in newVoxels:
                # Increase depthLevel by 1 unless the newVoxel is a transition node (degree=2)
                G.node[newVoxel]['depthLevel'] = G.node[currentVoxel]['depthLevel'] if G.degree(newVoxel) == 2 else G.node[currentVoxel]['depthLevel'] + 1
                newLength = norm(np.array(newVoxel) - np.array(currentVoxel))
                G.node[newVoxel]['pathDistance'] = float(G.node[currentVoxel]['pathDistance'] + newLength)
                if G.degree(newVoxel) >= 3: # add the current segment after passing a bifurcation point
                    newSegmentIndex = G[currentVoxel][newVoxel]['segmentIndex']
                    segmentIndexList.append(newSegmentIndex)
                elif G.degree(newVoxel) == 1: # add the current segment when reaching the end of a segment
                    newSegmentIndex = G[currentVoxel][newVoxel]['segmentIndex']
                    segmentIndexList.append(newSegmentIndex)

            poolTemp += newVoxels
            numOfVoxelsVisited += len(newVoxels)
            voxelsVisited += newVoxels
        
        pool = poolTemp
        depthVoxel += 1
    
    print('{} nodes visited'.format(numOfVoxelsVisited))
    return G, voxelsVisited, segmentIndexList

def randomWalkBFS2(G, initialVoxels, boundaryVoxels):
    """
    Similar to `randomWalkBFS` except that this function does not modify the graph `G`. `G` should have `depthVoxel` 
    attribute for each voxel.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be traversed.
    initialVoxels : list
        A list of voxels to start with.
    boundaryVoxels : list
        A list of voxels that mark the boundary.
    
    Returns
    -------
    G : NetworkX graph
        The graph with traversal information added.
    voxelsVisitedList : list
        A list of voxels that have been traversed.
    segmentIndexVisitedList : list
        A list of segment indices that have been traversed.
    """
    initialVoxelsTuple = initialVoxels
    boundaryVoxelsTuple = boundaryVoxels
    initialVoxelsTuple = list(map(tuple, initialVoxels))
    boundaryVoxelsTuple = list(map(tuple, boundaryVoxels))
    numOfVoxelsVisited = len(initialVoxelsTuple)
    voxelsVisitedList = copy.deepcopy(initialVoxelsTuple) # make deep copy
    pool = copy.deepcopy(initialVoxelsTuple) # make deep copy
    segmentIndexVisitedList = [] # record the index of segments traversed
    while len(pool) != 0:
        poolTemp = []
        for currentVoxel in pool:
            depthVoxel = G.node[currentVoxel]['depthVoxel']
            newVoxels = [voxel for voxel in G.neighbors(currentVoxel) if voxel not in boundaryVoxelsTuple and 'depthVoxel' in G.node[voxel] and G.node[voxel]['depthVoxel'] > depthVoxel]
            for newVoxel in newVoxels:
                if G.degree(newVoxel) >= 3: # add the current segment after passing a bifurcation point
                    newSegmentIndex = G[currentVoxel][newVoxel]['segmentIndex']
                    segmentIndexVisitedList.append(newSegmentIndex)
                elif G.degree(newVoxel) == 1: # add the current segment when reaching the end of a segment
                    newSegmentIndex = G[currentVoxel][newVoxel]['segmentIndex']
                    segmentIndexVisitedList.append(newSegmentIndex)

            poolTemp += newVoxels
            numOfVoxelsVisited += len(newVoxels)
            voxelsVisitedList += newVoxels
        
        pool = poolTemp
    
    print('{} nodes visited'.format(numOfVoxelsVisited))
    return G, voxelsVisitedList, segmentIndexVisitedList

def generateColormap(data, maxValue=None):
    """
    Generate a color map based on `data` that resembles the `jet` colormap in Matlab.

    Parameters
    ----------
    data : list or 1d array
        The data with which to generate the colormap.
    maxValue : float, optional
        Manually set a upper bound for the colormap. If not provided, the max of `data` will be used.
    """
    data -= np.amin(data)
    if maxValue is None:
        maxValue = np.amax(data)
        
    colormap = np.full((len(data), 4), 0, dtype=np.float)
    colormap[:, 3] = 1
    
    # [64, 255, 191] looks like white, why?
    jet = np.array([[0,0,143],[0,0,159],[0,0,175],[0,0,191],[0,0,207],[0,0,223],[0,0,239],[0,0,255],[0,16,255],[0,32,255],[0,48,255],[0,64,255],[0,80,255],[0,96,255],[0,111,255],[0,128,255],[0,143,255],[0,159,255],
        [0,175,255],[0,191,255],[0,207,255],[0,223,255],[0,239,255],[0,255,255],[16,255,239],[32,255,223],[48,255,207],[64,255,191],[80,255,175],[96,255,159],[111,255,143],[128,255,128],[143,255,111],[159,255,96],
        [175,255,80],[191,255,64],[207,255,48],[223,255,32],[239,255,16],[255,255,0],[255,239,0],[255,223,0],[255,207,0],[255,191,0],[255,175,0],[255,159,0],[255,143,0],[255,128,0],[255,111,0],[255,96,0],[255,80,0],
        [255,64,0],[255,48,0],[255,32,0],[255,16,0],[255,0,0],[239,0,0],[223,0,0],[207,0,0],[191,0,0],[175,0,0],[159,0,0],[143,0,0],[128,0,0]])
    
    normalizedData = data/maxValue
    bins = np.linspace(0, 1, num=64)
    inds = np.digitize(normalizedData, bins) - 1
    colormap[:, 0:3] = jet[inds]/255.0

    return colormap

def splineInterpolation(coords, pointLoc, smoothing=None, return_derivative=False, k=3, w=None):
    '''
    Use spline curve to fit the coords and return the derivative at desired point.

    Paremeters
    ----------
    coords : array_like
        A sequence of coordinates to be fit with a spline.
    pointLoc : int
        The index of the point at which the interpolated value/derivative will be returned.
    smoothing : float, optional
        Controls the smoothness of the spline.
    return_derivative : bool, optional
        If True, return interpolated derivative at the specified location. Otherwise, return interpolated value.
    k : int, optional
        Degree of the spline.
    w : array_like
        A sequence of the same length as `coords` and will be treated as weights during interpolation.
    
    Returns
    -------
    tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    u : array
        An array of the values of the parameter.
    value : float
        Interpolated value at the specified location.
    derivative : float
        Interpolated 1st normalized derivative vector at the specified location.
    '''
    
    dataLength = len(coords)
    if smoothing is None:
        if dataLength <= 20:
            smoothing = 100 # 2 * dataLength  # dataLength + np.sqrt(2 * dataLength)
        else:
            smoothing = dataLength + np.sqrt(2 * dataLength)
    
    if len(coords) <= k:
        k = len(coords) - 1
    
    if w is None:
        w = np.ones((len(coords[:, 0]),))

    tck, u = interpolate.splprep([coords[:, 0], coords[:, 1], coords[:, 2]], s=smoothing, k=k, w=w)
    v1, v2, v3 = interpolate.splev(pointLoc, tck, der=0)
    if len(pointLoc) == 1:
        value = np.array([v1, v2, v3])
    else:
        value = np.hstack((v1.reshape(-1,1), v2.reshape(-1,1), v3.reshape(-1,1)))

    if return_derivative:
        d1, d2, d3 = interpolate.splev(pointLoc, tck, der=1)
        if len(pointLoc) == 1:
            derivative = np.array([d1, d2, d3])
            derivative /= norm(derivative)
        else:
            derivative = np.hstack((d1.reshape(-1,1), d2.reshape(-1,1), d3.reshape(-1,1)))
            normList = np.array(list(map(norm, derivative))).reshape(-1, 1)
            derivative = derivative / normList

        return tck, u, value, derivative
    else:
        return tck, u, value

def curvature_by_triangle(points):
    """
    Calculate the curvature using the formula: kappa = 4S/(abc), where a,b,c are three points and S is the area of the
    triangle formed by the three points. S = np.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))) / 4   # Heron's formula
    for triangle's surface (a>=b>=c) 
    [Credit: https://books.google.com/books?hl=en&id=7J52J4GrsJkC&pg=PA45#v=onepage&q&f=false]

    Parameters
    ----------
    points : array_like
        A sequence of three points with which to calculate the approximate curvature.
    
    Returns
    -------
    kappa : float
        Approximated curvature value.
    """
    A, B, C = np.array(points) # points is a 3*N array, where N is the dimension
    a, b, c = norm(A-B), norm(A-C), norm(B-C)
    c, b, a = np.sort([a, b, c]) # make sure a>=b>=c
    temp = (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))
    if temp < 0:
        S = 0
    else:
        S = np.sqrt(temp) / 4
        
    kappa = 4 * S / (a * b * c)

    return kappa