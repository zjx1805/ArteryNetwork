import sys, os
from os.path import join
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
import glob
import pickle
from manualCorrectionGUIDetail import Vessel



def getBoundary(dataVolume, axis, flipAxis=False):
    """
    Obtain the boundary (the index of first nonzero element along an axis) of the data volume along a particular axis.

    Parameters
    ----------
    dataVolume : ndarray
        The data volume with which to find the boundary.
    axis : int
        Specify an axis along which to find the boundary.
    flipAxis : bool, optoinal
        If True, find the boundary from the other end.
    
    Returns
    -------
    val : ndarray
        The index of the first nonzero element along the given axis, dimension is one less than the data volume.
    """
    shape = np.array(dataVolume.shape)

    mask = dataVolume != 0
    if flipAxis:
        val = shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    else:
        val = np.argmax(mask, axis=axis)

    return val 

def mergeVolume(vessel150, vessel250, lowerBound, upperBound, axis):
    shape = vessel150.shape
    indexVolume = np.array([np.logical_and(ii >= lowerBound, ii <= upperBound) for ii in range(shape[axis])])
    print(np.count_nonzero(indexVolume), indexVolume.size)
    # print(indexVolume.shape, shape)
    vessel250[indexVolume] = vessel150[indexVolume]

    return indexVolume
    
def main():
    """
    Main function to manually correct the connections. The actual implementation of the GUI is defined in
    `manualCorrectionGUIDetail.py`
    """
    start_time = timeit.default_timer()
    baseFolder = os.path.abspath(os.path.dirname(__file__))
    resultFolder = os.path.abspath(os.path.dirname(__file__))
    vesselResultName = 'vesselResultFull'
    skeletonName = 'skeleton'
    labelInfoName = 'labelInfo'
    graphRepresentationName = 'GraphRepresentation'
    
    app = pg.QtGui.QApplication([])
    # w = gl.GLViewWidget()
    # w.opts['distance'] = 800
    # w.setGeometry(0, 110, 1920, 1080)
    ex = Vessel()
    w = ex.plotwidget
  
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

    directory = baseFolder
    
    vesselVolumeFilePath = os.path.join(baseFolder, 'vesselVolumeMask.nii.gz')
    vesselImg = nib.load(vesselVolumeFilePath)
    vessel = vesselImg.get_data()
    
    skeletonFilePath = os.path.join(baseFolder, 'skeleton.nii.gz')
    skeletonImg = nib.load(skeletonFilePath)
    skeleton = skeletonImg.get_data()
    shape = skeleton.shape
    offset = np.array(shape) / (-2.0)
    affineTransform = np.sign(np.diag(skeletonImg.affine)[:3])
    
    graphFilePath = os.path.join(baseFolder, 'graphRepresentation.graphml')
    G = nx.read_graphml(graphFilePath, node_type=make_tuple)
    
    plotItemCounter = 0
    #####
    skeletonCoords = np.array(np.where(skeleton != 0), dtype=np.int16).T
    skeletonCoordsView = (skeletonCoords + offset) * affineTransform
    #####
    d2 = np.empty(vessel.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = vessel * (255./(vessel.max()/1))
    d2[..., 1] = d2[..., 0]
    d2[..., 2] = d2[..., 0]
    d2[..., 3] = d2[..., 0]
    d2[..., 3] = 10#(d2[..., 3].astype(float) / 255.)**2 * 255
    glOptions = 'additive'
    v = gl.GLVolumeItem(d2, sliceDensity=1, smooth=True, glOptions=glOptions)
    v.translate(-d2.shape[0]/2, -d2.shape[1]/2, -d2.shape[2]/2)
    w.addItem(v)

    plotItemCounter += 1

    skeletonColor = np.full((len(skeletonCoords), 4), 1)
    skeletonColor[:, 1:3] = 0
    skeletonScatterPlot = gl.GLScatterPlotItem(pos=skeletonCoords, size=6, color=skeletonColor)
    skeletonScatterPlot.translate(-shape[0]/2, -shape[1]/2, -shape[2]/2)
    w.addItem(skeletonScatterPlot)
    skeletonNodesStartIndex = plotItemCounter
    plotItemCounter += 1
    
    segmentListFilePath = os.path.join(baseFolder, 'segmentList.npz')
    segmentList = np.load(segmentListFilePath)
    segmentList = list(segmentList['segmentList'])
    segmentStartIndex = plotItemCounter
    plotItemCounter += 1
    segmentCounter = 0
    indexVolume = np.full(shape, -1)
    segmentListDict = {}
    for segment in segmentList:
        segmentCoords = np.array(segment)
        aa = gl.GLLinePlotItem(pos=segmentCoords, color=pg.glColor('r'), width=3)
        aa.translate(-shape[0]/2, -shape[1]/2, -shape[2]/2)
        w.addItem(aa)
        indexVolume[tuple(segmentCoords.T)] = segmentCounter
        G.add_path(segment, segmentIndex=segmentCounter)
        segmentListDict[tuple(segment)] = segmentCounter
        w.segmentIndexUsed.append(segmentCounter)
        segmentCounter += 1
        plotItemCounter += 1
    
    print('There are {} segments'.format(len(segmentList)))
    
    eventListFilePath = os.path.join(baseFolder, 'eventList.pkl')
    removeListFilePath = os.path.join(baseFolder, 'removeList.npy')
    removeListAutoFilePath = os.path.join(baseFolder, 'removeListAuto.npy')
    if os.path.exists(eventListFilePath):
        with open(eventListFilePath, 'rb') as f:
            eventList = pickle.load(f)
        print('eventList loaded with {} events'.format(len(eventList)))
    elif os.path.exists(removeListFilePath):
        removeList = list(np.load(removeListFilePath))
        removeList = np.unique(removeList).tolist()
        print('removeList loaded with {} segments'.format(len(removeList)))
    elif os.path.exists(removeListAutoFilePath):
        removeList = list(np.load(removeListAutoFilePath))
        removeList = np.unique(removeList).tolist()
        print('removeListAuto loaded with {} segments'.format(len(removeList)))
    else:
        removeList = w.removeList
        print('Using empty removeList')
    
    cycle2SegmentDict = {}
    segment2CycleDict = {}

    w.show()
    w.addExtraInfo(skeletonNodesStartIndex=skeletonNodesStartIndex, segmentStartIndex=segmentStartIndex, indexVolume=indexVolume, 
                    affineTransform=affineTransform, offset=offset, cycle2SegmentDict=cycle2SegmentDict,
                    segment2CycleDict=segment2CycleDict, segmentList=segmentList, G=G, segmentListDict=segmentListDict,
                    resultFolder=directory)
    w.checkCycle()

    try:
        eventList
    except NameError:
        eventList = []
        for segmentIndex in removeList:
            event = {'type': 'remove', 'nodeIndex': 0, 'segmentIndex': segmentIndex}
            eventList.append(event)
    finally:
        for eventIndex, event in enumerate(eventList):
            print(eventIndex)
            success, event = w.processEvent(event)
            if success:
                w.eventList.append(event)

    # for segmentIndex in removeList:
    #     event = {'type': 'remove', 'nodeIndex': 0, 'segmentIndex': segmentIndex}
    #     success, event = w.processEvent(event)
    #     if success:
    #         w.eventList.append(event)
    
    w.checkCycle()
    w.onLoading = False
    # w.reportCycleInfo()

    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))
    pg.QtGui.QApplication.exec_()
    
    removeList = w.removeList
    removeListFilePath = os.path.join(baseFolder, 'removeList.npy')
    np.save(removeListFilePath, removeList)
    print('removeList saved with {} segments marked for deletion (ID = {})'.format(len(removeList), removeList))

    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))


if __name__ == "__main__":
    main()

