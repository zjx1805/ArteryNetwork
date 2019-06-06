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
import inspect
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
    
def manualCorrectionGUI():
    """
    Main function to manually correct the connections. The actual implementation of the GUI is defined in
    `manualCorrectionGUIDetail.py`
    """
    start_time = timeit.default_timer()
    baseFolder = os.path.abspath(os.path.dirname(__file__))
    
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

def calculateBranchInfo(segmentList1, segmentList2, vesselVolume, directory):
    '''
    This function tries to
    -- recover radius information for segments in segmentList2 based on segmentList1.
    -- calculate basic attribute (tuotorsity, length, meanRadius) of each branch.

    Note that since it is possible that some segments in segmentList2 are completely new and does not exist 
    in segmentList1, and thus the radius estimation may not be accurate. *Needs to find a better way in the 
    future.*
    
    Parameters
    ----------
    segmentList1 : list
        Comes directly from skeletonization.
    segmentList2 : list
        Cleaned version of segmentList1 (i.e., with loops removed by `manualCorrectionGUI`)
    vesselVolume : ndarray
        Segmented vessel volume array.
    directory : str
        The folder path where the distance transform array (associated with `vesselVolume`) is located at.
    
    Returns
    -------
    G : NetworkX graph
        A graph corresponding to the segments in segmentList2 and corresponding branch properties.
    
    '''
    
    vesselVolumeDistanceTransformFilePath = os.path.join(directory, 'vesselVolumeDistanceTransform.npz')
    if os.path.exists(vesselVolumeDistanceTransformFilePath):
        distanceTransform = np.load(vesselVolumeDistanceTransformFilePath)
        distanceTransform = distanceTransform['distanceTransform']
    else:
        distanceTransform = ndi.morphology.distance_transform_edt(vesselVolume)
        np.savez_compressed(vesselVolumeDistanceTransformFilePath, distanceTransform=distanceTransform) 
    
    shape = vesselVolume.shape
    # creat index volume based on segmentList1
    indexVolume = np.full(shape, 0, dtype=np.int16)
    for segment1Index, segment1 in enumerate(segmentList1):
        segment1Coords = np.array(segment1, dtype=np.int16)
        indexVolume[tuple(segment1Coords.T)] = segment1Index + 1 # first segment start with 1
    
    # create graph based on segmentList2
    G = nx.Graph()
    for segment2 in segmentList2:
        G.add_path(segment2)
    
    shortSegments2 = [] # segments of length 2 are processed later
    newSegments2 = [] # completely new segments from segmentList2 (not present in segmentList1) are processed later
    numOfSegments2 = len(segmentList2)
    normalSegments2Counter = 0 # segment2 that is exactly the same as in segmentList1 are put here
    segmentInfoDict = {} # stores the same info as the attributes in graph edges
    for segment2Index, segment2 in enumerate(segmentList2):
        if len(segment2) == 2:
            shortSegments2.append([segment2Index, segment2])
        else:
            linkVoxels = [voxel for voxel in segment2 if G.degree(voxel) == 2 and indexVolume[voxel] != 0]
            if len(linkVoxels) != 0:
                linkVoxelsCoords = np.array(linkVoxels, dtype=np.int16)
                linkVoxelsIndices = indexVolume[tuple(linkVoxelsCoords.T)]
                uniqueIndices = np.unique(linkVoxelsIndices).tolist()
                radiusList = distanceTransform[tuple(linkVoxelsCoords.T)]
                if len(uniqueIndices) == 1:
                    meanRadius = np.mean(radiusList)
                    sigma = np.std(radiusList)
                    normalSegments2Counter += 1
                else:
                    meanRadius = np.mean(radiusList)
                    sigma = np.std(radiusList)
                
                if meanRadius == 0:
                    temp = distanceTransform[tuple(np.array(segment2, dtype=np.int16).T)]
                    nonZeroRadiusList = temp[temp != 0]
                    if len(nonZeroRadiusList) != 0:
                        meanRadius = np.mean(nonZeroRadiusList)
                        sigma = np.std(nonZeroRadiusList)
                    else:
                        print('temp={}'.format(temp))
                        print('Error! Mean radius = 0, radiusList={}'.format(radiusList))
                
                lengthList = [norm(np.array(segment2[ii + 1]) - np.array(segment2[ii])) for ii in range(len(segment2) - 1)]
                pathLength = np.sum(lengthList)
                eculideanLength = norm(np.array(segment2[0]) - np.array(segment2[-1]))
                tortuosity = pathLength / eculideanLength
                voxelLength = len(segment2)
                # nx.write_graphml only accepts native python types as edge attributes
                pathLength = float(pathLength)
                eculideanLength = float(eculideanLength)
                tortuosity = float(tortuosity)
                meanRadius = float(meanRadius)
                sigma = float(sigma)
                segment2Index = int(segment2Index)
                G.add_path(segment2, pathLength=pathLength, eculideanLength=eculideanLength, tortuosity=tortuosity, voxelLength=voxelLength, 
                    meanRadius=meanRadius, sigma=sigma, segmentIndex=segment2Index)
                segmentInfoDict[tuple(segment2)] = {'pathLength': pathLength, 'eculideanLength': eculideanLength, 'tortuosity': tortuosity, 'voxelLength': voxelLength, 
                    'meanRadius': meanRadius, 'sigma': sigma, 'segmentIndex': segment2Index}
            else:
                newSegments2.append([segment2Index, segment2])
    
    for segment2Index, segment2 in shortSegments2:
        segment2Head, segment2Tail = segment2
        segment2HeadRadiusList = [G[segment2Head][voxel]['meanRadius'] for voxel in G.neighbors(segment2Head) if voxel != segment2Tail and 'meanRadius' in G[segment2Head][voxel]]
        segment2HeadMeanRadius = np.mean(segment2HeadRadiusList) if len(segment2HeadRadiusList) != 0 else 0
        segment2TailRadiusList = [G[segment2Tail][voxel]['meanRadius'] for voxel in G.neighbors(segment2Tail) if voxel != segment2Head and 'meanRadius' in G[segment2Tail][voxel]]
        segment2TailMeanRadius = np.mean(segment2TailRadiusList) if len(segment2TailRadiusList) != 0 else 0
        if segment2HeadMeanRadius != 0 and segment2TailMeanRadius != 0:
            meanRadius = (segment2HeadMeanRadius + segment2TailMeanRadius) / 2
        elif segment2HeadMeanRadius != 0 and segment2TailMeanRadius == 0:
            meanRadius = segment2HeadMeanRadius
        elif segment2HeadMeanRadius == 0 and segment2TailMeanRadius != 0:
            meanRadius = segment2TailMeanRadius
        else:
            meanRadius = 0
            print('Degree of head = {}, degree of tail = {}'.format(G.degree(segment2Head), G.degree(segment2Tail)))
            print('mean radius at both head and tail are zero')
        pathLength = norm(np.array(segment2Head) - np.array(segment2Tail))
        eculideanLength = pathLength
        tortuosity = pathLength / eculideanLength
        voxelLength = len(segment2)
        # nx.write_graphml only accepts native python types as edge attributes
        pathLength = float(pathLength)
        eculideanLength = float(eculideanLength)
        tortuosity = float(tortuosity)
        meanRadius = float(meanRadius)
        segment2Index = int(segment2Index)
        G.add_path(segment2, pathLength=pathLength, eculideanLength=eculideanLength, tortuosity=tortuosity, voxelLength=voxelLength, 
            meanRadius=meanRadius, segmentIndex=segment2Index) # short segments do not have sigma
        segmentInfoDict[tuple(segment2)] = {'pathLength': pathLength, 'eculideanLength': eculideanLength, 'tortuosity': tortuosity, 'voxelLength': voxelLength, 
            'meanRadius': meanRadius, 'segmentIndex': segment2Index}
    
    for segment2Index, segment2 in newSegments2:
        segment2Head, segment2Tail = segment2[0], segment2[-1]
        segment2HeadRadiusList = [G[segment2Head][voxel]['meanRadius'] for voxel in G.neighbors(segment2Head) if voxel != segment2Tail and 'meanRadius' in G[segment2Head][voxel]]
        segment2HeadMeanRadius = np.mean(segment2HeadRadiusList) if len(segment2HeadRadiusList) != 0 else 0
        segment2TailRadiusList = [G[segment2Tail][voxel]['meanRadius'] for voxel in G.neighbors(segment2Tail) if voxel != segment2Head and 'meanRadius' in G[segment2Tail][voxel]]
        segment2TailMeanRadius = np.mean(segment2TailRadiusList) if len(segment2TailRadiusList) != 0 else 0
        if segment2HeadMeanRadius != 0 and segment2TailMeanRadius != 0:
            meanRadius = (segment2HeadMeanRadius + segment2TailMeanRadius) / 2
        elif segment2HeadMeanRadius != 0 and segment2TailMeanRadius == 0:
            meanRadius = segment2HeadMeanRadius
        elif segment2HeadMeanRadius == 0 and segment2TailMeanRadius != 0:
            meanRadius = segment2TailMeanRadius
        else:
            meanRadius = 0
            print('mean radius at both head and tail are zero')
        pathLength = norm(np.array(segment2Head) - np.array(segment2Tail))
        eculideanLength = pathLength
        tortuosity = pathLength / eculideanLength
        voxelLength = len(segment2)
        # nx.write_graphml only accepts native python types as edge attributes
        pathLength = float(pathLength)
        eculideanLength = float(eculideanLength)
        tortuosity = float(tortuosity)
        meanRadius = float(meanRadius)
        segment2Index = int(segment2Index)
        G.add_path(segment2, pathLength=pathLength, eculideanLength=eculideanLength, tortuosity=tortuosity, voxelLength=voxelLength, 
            meanRadius=meanRadius, segmentIndex=segment2Index) # new segments do not have sigma
        segmentInfoDict[tuple(segment2)] = {'pathLength': pathLength, 'eculideanLength': eculideanLength, 'tortuosity': tortuosity, 'voxelLength': voxelLength, 
            'meanRadius': meanRadius, 'segmentIndex': segment2Index}
    
    for node in G.nodes():
        G.node[node]['radius'] = float(distanceTransform[node])
    
    print('normal segments: {}, shortSegments: {}, newSegments: {}'.format(normalSegments2Counter, len(shortSegments2), len(newSegments2)))
    # nx.write_graphml(G, os.path.join(directory, 'graphRepresentationCleanedWithEdgeInfo.graphml'))
    # print('graphRepresentationCleanedWithEdgeInfo.graphml saved')
    # with open(os.path.join(directory, 'segmentInfoDictBasic.pkl'), 'wb') as f:
    #     pickle.dump(segmentInfoDict, f, 2)
    #     print('segmentInfoDictBasic.pkl saved')
    return G

def updateGraph():
    """
    Update the graph corresponding to `segmentList2`.
    """
    start_time = timeit.default_timer()
    functionName = inspect.currentframe().f_code.co_name
    directory = os.path.abspath(os.path.dirname(__file__))
    
    segmentList1 = np.load(os.path.join(directory, 'segmentList.npz'))
    segmentList1 = segmentList1['segmentList']
    segmentList1 = list(map(tuple, segmentList1))
    segmentList2 = np.load(os.path.join(directory, 'segmentListCleaned.npz'))
    segmentList2 = segmentList2['segmentList']
    segmentList2 = list(map(tuple, segmentList2))

    vesselVolumeMaskFilePath = join(directory, 'vesselVolumeMask.nii.gz')
    if not os.path.exists(vesselVolumeMaskFilePath):
        print('Error! vesselVolumeMask.nii.gz does not exist at {}.'.format(directory))
        return
    
    vesselVolumeImg = nib.load(os.path.join(directory, 'vesselVolumeMask.nii.gz'))
    vesselVolume = vesselVolumeImg.get_data()
    shape = vesselVolume.shape
    G = calculateBranchInfo(segmentList1, segmentList2, vesselVolume, directory)
    nx.write_graphml(G, os.path.join(directory, 'graphRepresentationCleanedWithEdgeInfo.graphml'))
    print('graphRepresentationCleanedWithEdgeInfo.graphml saved to {}.'.format(directory))
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

if __name__ == "__main__":
    manualCorrectionGUI()
    updateGraph()

