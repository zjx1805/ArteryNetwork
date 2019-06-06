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
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from scipy import interpolate
import scipy.spatial as sp
import logging
import traceback
import timeit
import time
import math
from ast import literal_eval as make_tuple
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import pickle
from scipy.stats import ttest_ind
import copy
from operator import itemgetter
from os.path import join
import inspect
from partitionCompartmentGUIDetail import myVessel

def partitionCompartmentGUI():
    start_time = timeit.default_timer()
    functionName = inspect.currentframe().f_code.co_name
    directory = os.path.abspath(os.path.dirname(__file__))
    
    graphCleanedWithEdgeInfoFilePath = os.path.join(directory, 'graphRepresentationCleanedWithEdgeInfo.graphml') # returned by calculateBranchInfo()
    vesselVolumeMaskFilePath = join(directory, 'vesselVolumeMask.nii.gz')
    if not os.path.exists(graphCleanedWithEdgeInfoFilePath):
        print('Error! graphRepresentationCleanedWithEdgeInfo.graphml does not exist at {}.'.format(directory))
        return
    
    if not os.path.exists(vesselVolumeMaskFilePath):
        print('Error! vesselVolumeMask.nii.gz does not exist at {}.'.format(directory))
        return

    G = nx.read_graphml(graphCleanedWithEdgeInfoFilePath, node_type=make_tuple)
    vesselVolumeImg = nib.load(vesselVolumeMaskFilePath)
    vesselVolume = vesselVolumeImg.get_data()
    shape = vesselVolume.shape

    app = pg.QtGui.QApplication([])
    ex = myVessel(app=app)
    w = ex.plotwidget
    w.opts['distance'] = 800
    w.show()

    ex.addExtraInfo(directory=directory)
    w.addExtraInfo(directory=directory)
    offset = np.array(shape) / (-2.0)
    w.offset = offset # Don't forget this!
    G = nx.read_graphml(os.path.join(directory, 'graphRepresentationCleanedWithEdgeInfo.graphml'), node_type=make_tuple)
    segmentList = np.load(os.path.join(directory, 'segmentListCleaned.npz'))
    segmentList = list(map(tuple, segmentList['segmentList']))
    # G, segmentList = createDummyNetwork()
    itemCounter = 0

    allNodes = list(G.nodes())
    allNodesCoords = np.array(allNodes, dtype=np.int16)
    color = np.full((len(allNodesCoords), 4), 1, dtype=np.float)
    size = np.full((len(allNodesCoords), ), w._voxelNormalSize)
    aa = gl.GLScatterPlotItem(pos=allNodesCoords, size=size, color=color)
    aa.translate(*offset)
    w.addItem(aa)
    skeletonPlotItemIndex = itemCounter
    itemCounter += 1
    
    voxelSegmentIndexArray = np.full(shape, -1)
    segmentStartIndex = itemCounter
    for segmentIndex, segment in enumerate(segmentList):
        segmentCoords = np.array(segment, dtype=np.int16)
        voxelSegmentIndexArray[tuple(segmentCoords.T)] = segmentIndex
        aa = gl.GLLinePlotItem(pos=segmentCoords, width=2, color=pg.glColor('w'))
        aa.translate(*offset)
        w.addItem(aa)
        itemCounter += 1
    
    voxelIndexArray = np.full(shape, 0)
    voxelIndexArray[tuple(allNodesCoords.T)] = np.arange(len(allNodesCoords))
    # for idx, voxel in enumerate(allNodes):
    #     voxelIndexArray[voxel] = idx

    w.addExtraInfo(skeletonPlotItemIndex=skeletonPlotItemIndex, G=G, voxelIndexArray=voxelIndexArray, voxelSegmentIndexArray=voxelSegmentIndexArray, segmentList=segmentList, shape=shape, segmentStartIndex=segmentStartIndex)


    elapsed = timeit.default_timer() - start_time
    print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
    pg.QtGui.QApplication.exec_()
    # sys.exit(app.exec_())

if __name__ == "__main__":
    partitionCompartmentGUI()