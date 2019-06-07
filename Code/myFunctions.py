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