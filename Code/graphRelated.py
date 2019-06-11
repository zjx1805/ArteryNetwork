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
import myFunctions as mf
from scipy.stats import ttest_ind
import copy
from operator import itemgetter
from os.path import join
import inspect



def calculateProperty(G, segmentList, spacing=0.00025, skipUncategorizedVoxels=False):
    '''
    This function calculates various morphological properties described in our paper. Note that this function should be
    used after calling `calculateBranchInfo` in `manualCorrectionGUI.py` (because radius needs to be recovered first).
    Also note that `nodeInfoDict` only contains voxels that have degree = 3.

    Parameters
    ----------
    G : NetworkX graph
        The graph containing all the necessary information about the vessel network.
    segmentList : list
        A list containing all the segments (such that each of them is a simple branch).
    spacing : float
        Unit is meter/voxel. Used to convert voxel distance into physical distance.
    skipUncategorizedVoxels : bool, optional
        If True, skip all the voxels that are not partitioned into any compartments.
    
    Returns
    -------
    nodeInfoDict: dict  
        A dictionary containing all the information about the bifurcation voxels indexed by their coordinates.  
    segmentInfoDict : dict  
        A dictionary containing all the information about the segments in `segmentList` indexed by its position in
        `segmentList`.  
    '''
    G2 = nx.Graph() # only to record what voxels are used
    segmentInfoDict = {}
    nodeInfoDict = {}
    for segmentIndex, segment in enumerate(segmentList):
        if segment[0] != segment[-1]:
            lengthList = [norm(np.array(segment[ii + 1]) - np.array(segment[ii])) for ii in range(len(segment) - 1)]
            pathLength = G[segment[0]][segment[1]]['pathLength']
            eculideanLength = G[segment[0]][segment[1]]['eculideanLength']
            tortuosity = G[segment[0]][segment[1]]['tortuosity']
            voxelLength = G[segment[0]][segment[1]]['voxelLength']
            meanRadius = G[segment[0]][segment[1]]['meanRadius']
            segmentInfoDict[segmentIndex] = {'pathLength': pathLength, 'eculideanLength': eculideanLength, 'tortuosity': tortuosity, 'voxelLength': voxelLength, 'meanRadius': meanRadius}
            # partitionName and segmentLevel only exist in categorized voxels
            if 'partitionName' in G[segment[0]][segment[1]]:
                segmentInfoDict[segmentIndex]['partitionName'] = G[segment[0]][segment[1]]['partitionName']
            if 'segmentLevel' in G[segment[0]][segment[1]]:
                segmentInfoDict[segmentIndex]['segmentLevel'] = G[segment[0]][segment[1]]['segmentLevel']
            # sigma only exists in normal AND categorized segments (definition of normal see calculateBranchInfo())
            if 'sigma' in G[segment[0]][segment[1]]:
                segmentInfoDict[segmentIndex]['sigma'] = G[segment[0]][segment[1]]['sigma']
            # add info to indicate whether this segment is terminating or bifurcating
            if G.degree(segment[0]) == 1 or G.degree(segment[-1]) == 1:
                segmentInfoDict[segmentIndex]['type'] = 'terminating'
            elif G.degree(segment[0]) >= 3 or G.degree(segment[-1]) >= 3:
                segmentInfoDict[segmentIndex]['type'] = 'bifurcating'
            # add aspect ratio (length/radius)
            segmentInfoDict[segmentIndex]['aspectRatio'] = pathLength / meanRadius
            # add voxels to G2
            G2.add_path(segment)

        else:
            print('Segment {} has same head and tail'.format(segment))
    
    if len(G.nodes()) <= 50:
        print('Statistics calculation aborted because there are less than 50 nodes in this connected component!')
        return
    
    for node in G2.nodes():
        nodeInfoDict[node] = {}
        # The following applies to all nodes (not only bifurcation nodes)
        # add depthVoxel/depthLevel/pathDistance if possible
        if 'depthVoxel' in G.node[node]:
            nodeInfoDict[node]['depthVoxel'] = G.node[node]['depthVoxel']
        if 'depthLevel' in G.node[node]:
            nodeInfoDict[node]['depthLevel'] = G.node[node]['depthLevel'] 
        if 'pathDistance' in G.node[node]:
            nodeInfoDict[node]['pathDistance'] = G.node[node]['pathDistance']
        
        # add partitionName if possible
        if 'partitionName' in G.node[node]:
            nodeInfoDict[node]['partitionName'] = G.node[node]['partitionName']
        # add info indicating whether this voxel is terminating or bifurcating
        if G.degree(node) == 1:
            nodeInfoDict[node]['type'] = 'terminating'
        elif G.degree(node) >= 3:
            nodeInfoDict[node]['type'] = 'bifurcating'
        # every node has radius info
        nodeInfoDict[node]['radius'] = G.node[node]['radius']

        if G.degree(node) == 3:
            segmentsInfo = []
            # nodeInfoDict[node] = {}
            for segmentIndex, segment in enumerate(segmentList):
                if segment[0] == node and len(segment) >= 3:
                    segmentsInfo.append([segmentIndex, segment])
                elif segment[-1] == node and len(segment) >= 3:
                    segmentsInfo.append([segmentIndex, segment[::-1]])
                else:
                    # print('Error! Desired segment not in segmentList!')
                    continue
            
            if len(segmentsInfo) != 3:
                continue
            
            # radiusList = np.array([segmentInfoDict[segmentIndex]['meanRadius'] for segmentIndex, _ in segmentsInfo])
            # segmentsInfo = [segmentsInfo[ii] for ii in radiusList.argsort()]
            
            # vec1 and vec2 are child branches and vec3 is parent branch
            segmentsInterpolated = []
            segmentsDerivative = []
            hasDepthInfoList = []
            for segmentIndex, segment in segmentsInfo:
                segmentLength = len(segment)
                segmentCoords = np.array(segment)
                weights = np.ones((segmentLength,))
                weights[[0, -1]] = 20
                _, _, value, derivative = mf.splineInterpolation(segmentCoords, np.linspace(0, 1, segmentLength), return_derivative=True, w=weights)
                segmentsInterpolated.append(value)
                segmentsDerivative.append(derivative)
                hasDepthInfo = True if 'depthVoxel' in G.node[segment[1]] else False
                hasDepthInfoList.append(hasDepthInfo)
            
            useOtherWayToDetermineOrder = False
            if 'depthVoxel' in G.node[node] and np.all(hasDepthInfoList):
                depthList = [G.node[segmentsInfo[0][1][1]]['depthVoxel'], G.node[segmentsInfo[1][1][1]]['depthVoxel'], 
                    G.node[segmentsInfo[2][1][1]]['depthVoxel'], G.node[node]['depthVoxel']]
                sortedIndex = np.argsort(depthList)
                nodeDepthLoc = np.nonzero(sortedIndex == 3)[0][0]    
                if nodeDepthLoc == 1:
                    order = [sortedIndex[2], sortedIndex[3], sortedIndex[0]]
                else:
                    useOtherWayToDetermineOrder = True
                    # print('depthList = {}'.format(depthList))
            else:
                useOtherWayToDetermineOrder = True
                # The following applies to all nodes (not only bifurcation nodes)
                # add depthVoxel/depthLevel/pathDistance if possible
                if 'depthVoxel' in G.node[node]:
                    nodeInfoDict[node]['depthVoxel'] = G.node[node]['depthVoxel']
                if 'depthLevel' in G.node[node]:
                    nodeInfoDict[node]['depthLevel'] = G.node[node]['depthLevel'] 
                if 'pathDistance' in G.node[node]:
                    nodeInfoDict[node]['pathDistance'] = G.node[node]['pathDistance']
                
                # add partitionName if possible
                if 'partitionName' in G.node[node]:
                    nodeInfoDict[node]['partitionName'] = G.node[node]['partitionName']
                # add info indicating whether this voxel is terminating or bifurcating
                if G.degree(node) == 1:
                    nodeInfoDict[node]['type'] = 'terminating'
                elif G.degree(node) >= 3:
                    nodeInfoDict[node]['type'] = 'bifurcating'
                # every node has radius info
                nodeInfoDict[node]['radius'] = G.node[node]['radius']
                # print('G.node[{}] has depthVoxel: {}, hasDepthInfoList = {}'.format(node, 'depthVoxel' in G.node[node], hasDepthInfoList))

            if useOtherWayToDetermineOrder:
                if skipUncategorizedVoxels:
                    continue

                maxCosineValue = -10
                for ii in range(3):
                    if ii != 2:
                        vec1 = segmentsDerivative[ii][0]
                        vec2 = segmentsDerivative[ii + 1][0]
                    else:
                        vec1 = segmentsDerivative[ii][0]
                        vec2 = segmentsDerivative[0][0]

                    currentCosineValue = np.dot(vec1, vec2)
                    if currentCosineValue > maxCosineValue:
                        maxCosineValue = currentCosineValue
                        if ii == 0:
                            order = [0,1,2]
                        elif ii == 1:
                            order = [1,2,0]
                        else:
                            order = [2,0,1]
            
            segmentsInfo = [segmentsInfo[ii] for ii in order]
            segmentsInterpolated = [segmentsInterpolated[ii] for ii in order]
            segmentsDerivative = [segmentsDerivative[ii] for ii in order]
            # local bifurcation amplitude 
            ###
            # vec1Local = np.array(segmentsInfo[0][1][1]) - np.array(node)
            # vec2Local = np.array(segmentsInfo[1][1][1]) - np.array(node)
            # vec1LocalNorm = norm(vec1Local)
            # vec2LocalNorm = norm(vec2Local)
            ###
            vec1Local = segmentsDerivative[0][0]
            vec2Local = segmentsDerivative[1][0]
            vec1LocalNorm = norm(vec1Local)
            vec2LocalNorm = norm(vec2Local)
            cosineValue = np.dot(vec1Local, vec2Local) / (vec1LocalNorm * vec2LocalNorm)
            if cosineValue > 1:
                cosineValue = 1
            elif cosineValue < -1:
                cosineValue = -1

            localBifurcationAmplitude = np.arccos(cosineValue) / np.pi * 180
            nodeInfoDict[node]['localBifurcationAmplitude'] = localBifurcationAmplitude
            # remote bifurcation amplitude
            vec1Remote = np.array(segmentsInfo[0][1][-1]) - np.array(node)
            vec2Remote = np.array(segmentsInfo[1][1][-1]) - np.array(node)
            vec1RemoteNorm = norm(vec1Remote)
            vec2RemoteNorm = norm(vec2Remote)
            cosineValue = np.dot(vec1Remote, vec2Remote) / (vec1RemoteNorm * vec2RemoteNorm)
            if cosineValue > 1:
                cosineValue = 1
            elif cosineValue < -1:
                cosineValue = -1

            remoteBifurcationAmplitude = np.arccos(cosineValue) / np.pi * 180
            nodeInfoDict[node]['remoteBifurcationAmplitude'] = remoteBifurcationAmplitude
            # local bifurcation tilt
            vecHalfAngle = vec1Local / vec1LocalNorm + vec2Local / vec2LocalNorm
            vecHalfAngleNorm = norm(vecHalfAngle)
            if vecHalfAngleNorm > 10**-4:
                # vecParent = np.array(segmentsInfo[2][1][0]) - np.array(segmentsInfo[2][1][-1])
                # vecParentNorm = norm(vecParent)
                vecParent = -segmentsDerivative[2][0]
                vecParentNorm = norm(vecParent)
                cosineValue = np.dot(vecHalfAngle, vecParent) / (vecHalfAngleNorm * vecParentNorm)
                localBifurcationTilt = np.arccos(cosineValue) / np.pi * 180
                if np.isnan(localBifurcationTilt):
                    print(vecHalfAngle, vecParent, vec1Local, vec2Local)
                nodeInfoDict[node]['localBifurcationTilt'] = localBifurcationTilt
            # remote bifurcation tilt
            vecHalfAngle = vec1Remote / vec1RemoteNorm + vec2Remote / vec2RemoteNorm
            vecHalfAngleNorm = norm(vecHalfAngle)
            if vecHalfAngleNorm > 10**-4:
                cosineValue = np.dot(vecHalfAngle, vecParent) / (vecHalfAngleNorm * vecParentNorm)
                if np.isnan(cosineValue):
                    print(vecHalfAngleNorm, vecParentNorm, vec1Remote, vec2Remote)
                remoteBifurcationTilt = np.arccos(cosineValue) / np.pi * 180
                nodeInfoDict[node]['remoteBifurcationTilt'] = remoteBifurcationTilt
            # check if r1^n + r2^n ~= r3^n for n = 2, 3 holds
            r1 = segmentInfoDict[segmentsInfo[0][0]]['meanRadius']
            r2 = segmentInfoDict[segmentsInfo[1][0]]['meanRadius']
            r3 = segmentInfoDict[segmentsInfo[2][0]]['meanRadius']
            nodeInfoDict[node]['cubicLawResult'] = (r1**3 + r2**3) / r3**3
            nodeInfoDict[node]['squareLawResult'] = (r1**2 + r2**2) / r3**2
            nodeInfoDict[node]['radiusList'] = [r1, r2, r3]
            nodeInfoDict[node]['minRadius'] = min([r1, r2, r3])
            nodeInfoDict[node]['minRadiusRatio'] = min([r1, r2]) / r3
            nodeInfoDict[node]['maxRadiusRatio'] = max([r1, r2]) / r3
            # length ratio (min child branch length/parent branch length)
            l1 = segmentInfoDict[segmentsInfo[0][0]]['pathLength']
            l2 = segmentInfoDict[segmentsInfo[1][0]]['pathLength']
            l3 = segmentInfoDict[segmentsInfo[2][0]]['pathLength']
            nodeInfoDict[node]['lengthRatio'] = min([l1, l2]) / l3
            
            # normal vector
            normalVector = np.cross(vec1Local, vec2Local)
            normalVector /= norm(normalVector)
            nodeInfoDict[node]['normalVector'] = normalVector
        
        # The following applies to all nodes (not only bifurcation nodes)
        # add depthVoxel/depthLevel/pathDistance if possible
        if 'depthVoxel' in G.node[node]:
            nodeInfoDict[node]['depthVoxel'] = G.node[node]['depthVoxel']
        if 'depthLevel' in G.node[node]:
            nodeInfoDict[node]['depthLevel'] = G.node[node]['depthLevel'] 
        if 'pathDistance' in G.node[node]:
            nodeInfoDict[node]['pathDistance'] = G.node[node]['pathDistance']
        
        # add partitionName if possible
        if 'partitionName' in G.node[node]:
            nodeInfoDict[node]['partitionName'] = G.node[node]['partitionName']
        # add info indicating whether this voxel is terminating or bifurcating
        if G.degree(node) == 1:
            nodeInfoDict[node]['type'] = 'terminating'
        elif G.degree(node) >= 3:
            nodeInfoDict[node]['type'] = 'bifurcating'
        # every node has radius info
        nodeInfoDict[node]['radius'] = G.node[node]['radius']
        
    # local bifurcation torque
    for segmentIndex, segment in enumerate(segmentList):
        edgeHead = segment[0]
        edgeTail = segment[-1]
        if G.degree(edgeHead) == 3 and G.degree(edgeTail) == 3 and 'normalVector' in nodeInfoDict[edgeHead] and 'normalVector' in nodeInfoDict[edgeTail]:
            edgeHeadNormalVector = nodeInfoDict[edgeHead]['normalVector']
            edgeTailNormalVector = nodeInfoDict[edgeTail]['normalVector']
            edgeHeadNormalVectorNorm = norm(edgeHeadNormalVector)
            edgeTailNormalVectorNorm = norm(edgeTailNormalVector)
            cosineValue = np.dot(edgeHeadNormalVector, edgeTailNormalVector) / (edgeHeadNormalVectorNorm * edgeTailNormalVectorNorm)
            if cosineValue > 1:
                cosineValue = 1
            elif cosineValue < -1:
                cosineValue = -1

            localBifurcationTorque = np.arccos(cosineValue) / np.pi * 180
            if localBifurcationTorque > 90:
                localBifurcationTorque = 180 - localBifurcationTorque
            segmentInfoDict[segmentIndex]['localBifurcationTorque'] = localBifurcationTorque

    
    # Report the result

    quantity = 'meanRadius'
    quantityList = np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) > 2 and G.degree(segmentList[segmentIndex][-1]) > 2])
    print(quantity + '(bifurcating, voxel) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'meanRadius'
    quantityList = np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) == 1 or G.degree(segmentList[segmentIndex][-1]) == 1])
    print(quantity + '(terminating, voxel) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'pathLength'
    quantityList = spacing * 1000 * np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) > 2 and G.degree(segmentList[segmentIndex][-1]) > 2])
    print(quantity + '(bifurcating, mm) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'pathLength'
    quantityList = spacing * 1000 * np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) == 1 or G.degree(segmentList[segmentIndex][-1]) == 1])
    print(quantity + '(terminating, mm) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))
    
    # local bifurcating tortuosity
    quantity = 'tortuosity'
    quantityList = np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) > 2 and G.degree(segmentList[segmentIndex][-1]) > 2])
    print(quantity + '(local bifurcating) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    # local terminating tortuosity
    quantity = 'tortuosity'
    quantityList = np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo and G.degree(segmentList[segmentIndex][0]) == 1 or G.degree(segmentList[segmentIndex][-1]) == 1])
    print(quantity + '(local terminating) mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'localBifurcationTorque'
    quantityList = np.array([segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo])
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'localBifurcationAmplitude'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    # print(quantityList)
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'remoteBifurcationAmplitude'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'localBifurcationTilt'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    # print(quantityList)
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'remoteBifurcationTilt'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'cubicLawResult'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    quantity = 'squareLawResult'
    quantityList = np.array([nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo])
    print(quantity + ' mean+-SD = {} +- {} ({}-{})'.format(np.mean(quantityList), np.std(quantityList), np.amin(quantityList), np.amax(quantityList)))

    print('Total # branches: {}'.format(len(segmentInfoDict)))
    quantity = 'pathLength'
    pathLengthList = [segInfo[quantity] for segmentIndex, segInfo in segmentInfoDict.items() if quantity in segInfo]
    print('Total length (mm): {}'.format(np.sum(pathLengthList) * spacing * 1000))
    numOfBifurcatingNode = len([node for node, nodeInfo in nodeInfoDict.items() if 'type' in nodeInfo and nodeInfo['type'] == 'bifurcating'])
    numOfTerminatingNode = len([node for node, nodeInfo in nodeInfoDict.items() if 'type' in nodeInfo and nodeInfo['type'] == 'terminating'])
    print('# bifurcating nodes = {}, # of terminating nodes = {}'.format(numOfBifurcatingNode, numOfTerminatingNode))
    quantity = 'depthLevel'
    branchOrderList = [nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo]
    print('Max branch order: {}'.format(np.max(branchOrderList)))
    quantity = 'pathDistance'
    pathDistanceList = [nodeInfo[quantity] for node, nodeInfo in nodeInfoDict.items() if quantity in nodeInfo]
    print('Max path distance (mm): {}'.format(np.max(pathDistanceList) * spacing * 1000))

    return nodeInfoDict, segmentInfoDict

def generateInfoDict():
    '''
    Generate segmentInfoDict/nodeInfoDict for the entire graph (only categorized voxels).

    The keys of `segmentInfoDict` are the segment indices. For example, key=10 refers to the 10th segment in
    `segmentListCleaned.npz`. Each segment has various properties as attributes: length, radius, tortuosity, etc., See
    function `calculateProperty` for details.

    The keys of `nodeInfoDict` are the voxel coordinates of the bifurcation voxels. Simiarly, it contains various
    properties as attributes: depth information, bifurcation angles, Murray's law ratio, etc., See function
    `calculateProperty` for details.
    '''
    start_time = timeit.default_timer()
    functionName = inspect.currentframe().f_code.co_name
    directory = os.path.abspath(os.path.dirname(__file__))
    spacing = 0.00040 # meter/voxel

    G = nx.read_graphml(os.path.join(directory, 'graphRepresentationCleanedWithAdvancedInfo.graphml'), node_type=make_tuple)
    segmentList = np.load(os.path.join(directory, 'segmentListCleaned.npz'))
    segmentList = list(map(tuple, segmentList['segmentList']))
    nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentList, spacing=spacing, skipUncategorizedVoxels=True)
    with open(os.path.join(directory, 'segmentInfoDict.pkl'), 'wb') as f:
        pickle.dump(segmentInfoDict, f, 2)
        print('segmentInfoDict.pkl saved')
    with open(os.path.join(directory, 'nodeInfoDict.pkl'), 'wb') as f:
        pickle.dump(nodeInfoDict, f, 2)
        print('nodeInfoDict.pkl saved')
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

def loadBasicFiles(directory=None):
    """
    Load files that will be used for most analysis.

    Parameters
    ----------
    directory : str, optional
        The folder path under which data files are saved.

    Returns
    -------
    result : dict
        A dictionary containing all the loaded data files.
    """
    if directory is None:
        directory = os.path.abspath(os.path.dirname(__file__))
    print('Start loading files...')
    
    filepath = join(directory, 'graphRepresentationCleanedWithAdvancedInfo.graphml')
    if os.path.exists(filepath):
        G = nx.read_graphml(filepath, node_type=make_tuple)
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}
    
    filepath = join(directory, 'segmentListCleaned.npz')
    if os.path.exists(filepath):
        segmentList = np.load(os.path.join(directory, 'segmentListCleaned.npz'))
        segmentList = list(map(tuple, segmentList['segmentList']))
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}
    
    filepath = join(directory, 'segmentInfoDict.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            segmentInfoDict = pickle.load(f)
            print('segmentInfoDict.pkl loaded from {}.'.format(directory))
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}
    
    filepath = join(directory, 'nodeInfoDict.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            nodeInfoDict = pickle.load(f)
            print('nodeInfoDict.pkl loaded from {}.'.format(directory))
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}
    
    filepath = join(directory, 'chosenVoxelsForPartition.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            chosenVoxels = pickle.load(f)
            print('chosenVoxelsForPartition.pkl loaded from {}.'.format(directory))
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}
    
    filepath = join(directory, 'partitionInfo.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            partitionInfo = pickle.load(f)
            print('partitionInfo.pkl loaded from {}.'.format(directory))
    else:
        print('Error! {} does not exist.'.format(filepath))
        return {}

    ADANFolder = os.path.abspath(os.path.join(directory, '../../../'))
    filepath = join(ADANFolder, 'ADAN-Web/resultADANDict.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            resultADANDict = pickle.load(f)
            print('resultADANDict.pkl loaded from {}.'.format(ADANFolder))
    else:
        print('Warning! {} does not exist.'.format(filepath))
        resultADANDict = {}
    
    result = {'G': G, 'segmentList': segmentList, 'segmentInfoDict': segmentInfoDict, 'nodeInfoDict': nodeInfoDict, 
              'chosenVoxels': chosenVoxels, 'partitionInfo': partitionInfo, 'resultADANDict': resultADANDict}

    return result

def calculateCurvature():
    """
    Calculate the curvature for each segment and save the result to `segmentInfoDict`.
    """
    start_time = timeit.default_timer()
    functionName = inspect.currentframe().f_code.co_name
    directory = os.path.abspath(os.path.dirname(__file__))
    spacingFactor = 0.40 # voxel->mm
    
    # Load files #
    result = loadBasicFiles(directory=directory)
    G, segmentList, segmentInfoDict, nodeInfoDict = itemgetter('G', 'segmentList', 'segmentInfoDict', 'nodeInfoDict')(result)
    chosenVoxels, partitionInfo, resultADANDict = itemgetter('chosenVoxels', 'partitionInfo', 'resultADANDict')(result)

    nodeInfoLocal: dict = {}
    edgeInfoLocal: dict = {}
    for partitionName, partitionInfo in partitionInfo.items():
        nodeInfoLocal[partitionName] = {}
        rootNodes = chosenVoxels[partitionName]['initialVoxels']
        boundaryNodes = chosenVoxels[partitionName]['boundaryVoxels']
        visitedNodes = partitionInfo['visitedVoxels']
        segmentIndexUsed: list = partitionInfo['segmentIndexList']
        GSub = G.subgraph(visitedNodes)
        for node in visitedNodes:
            nodeInfoLocal[partitionName][node] = {'curvatureWeight': 0} # used for determing weight for spline interpolation

        terminatingNodes: list = [node for node in visitedNodes if (G.degree(node) == 1) and node not in rootNodes]
        for terminatingNode in terminatingNodes:
            rootNodeFound = False
            for rootNodeCandidate in rootNodes:
                if nx.has_path(GSub, rootNodeCandidate, terminatingNode) == 1:
                    rootNode = rootNodeCandidate
                    rootNodeFound = True
                    path: list = nx.shortest_path(GSub, rootNode, terminatingNode)
                    pathSegmentIndexList: list = [GSub[path[ii]][path[ii + 1]]['segmentIndex'] for ii in range(len(path) - 1)]
                    uniquePathSegmentIndexList: list = np.unique(pathSegmentIndexList)
                    assert len(uniquePathSegmentIndexList) != 0
                    segmentLengthList: list = [segmentInfoDict[segmentIndex]['pathLength'] for segmentIndex in uniquePathSegmentIndexList]
                    segmentLengthCumsum = np.cumsum(segmentLengthList)
                    segmentLengthCumsum = np.insert(segmentLengthCumsum, 0, 0) # prepend 0
                    # save path/segmentLengthCumsum
                    nodeInfoLocal[partitionName][terminatingNode]['path'] = path
                    nodeInfoLocal[partitionName][terminatingNode]['uniquePathSegmentIndexList'] = uniquePathSegmentIndexList
                    nodeInfoLocal[partitionName][terminatingNode]['segmentLengthCumsum'] = segmentLengthCumsum
                    # add one to the curvature weight of each node in the path
                    for node in path:
                        nodeInfoLocal[partitionName][node]['curvatureWeight'] += 1
                    break
            
            if not rootNodeFound:
                print('Error! Root node not found for node {}.'.format(terminatingNode))
                return segmentInfoDict
        
        # Spline interpolation #
        for terminatingNode in terminatingNodes:
            # Retrieve info #
            path = nodeInfoLocal[partitionName][terminatingNode]['path']
            uniquePathSegmentIndexList = nodeInfoLocal[partitionName][terminatingNode]['uniquePathSegmentIndexList']
            segmentLengthCumsum = nodeInfoLocal[partitionName][terminatingNode]['segmentLengthCumsum']
            weightList = [nodeInfoLocal[partitionName][node]['curvatureWeight'] for node in path]

            # Init #
            for segmentIndex in uniquePathSegmentIndexList:
                edgeInfoLocal[segmentIndex] = {'maxCurvatureList': [], 'meanCurvatureList': []}
            
            # Calculate spline #
            coords = np.array(path) * spacingFactor # Unit: mm
            pointLoc = segmentLengthCumsum / np.amax(segmentLengthCumsum)
            tck, u, values = mf.splineInterpolation(coords, pointLoc, w=weightList)

            # Insert more sampling points into the spline of each branch so that the distance between two consecutive points is at most 0.125 mm (0.5 voxels)
            for ii, segmentIndex in enumerate(uniquePathSegmentIndexList):
                curvatureList = []
                uHead, uTail = pointLoc[ii], pointLoc[ii + 1]
                pathLength, voxelLength = itemgetter('pathLength', 'voxelLength')(segmentInfoDict[segmentIndex]) # Unit: voxel
                numOfNodesNeeded = np.ceil(pathLength / 0.5) + 1
                uList = np.linspace(uHead, uTail, numOfNodesNeeded)
                v1, v2, v3 = interpolate.splev(uList, tck, der=0)
                if len(uList) == 1:
                    values = np.array([v1, v2, v3])
                else:
                    values = np.hstack((v1.reshape(-1,1), v2.reshape(-1,1), v3.reshape(-1,1)))
                
                for jj in range(len(values) - 2):
                    splineCoords = np.array(values[jj:jj+3])
                    curvature = mf.curvature_by_triangle(splineCoords)
                    curvatureList.append(curvature)
                
                edgeInfoLocal[segmentIndex]['maxCurvatureList'].append(np.amax(curvatureList))
                edgeInfoLocal[segmentIndex]['meanCurvatureList'].append(np.mean(curvatureList))
        
        for segmentIndex in edgeInfoLocal.keys():
            maxCurvatureAveraged = np.mean(edgeInfoLocal[segmentIndex]['maxCurvatureList'])
            meanCurvatureAveraged = np.mean(edgeInfoLocal[segmentIndex]['meanCurvatureList'])
            segmentInfoDict[segmentIndex]['maxCurvatureAveragedInmm'] = maxCurvatureAveraged # Unit: mm
            segmentInfoDict[segmentIndex]['meanCurvatureAveragedInmm'] = meanCurvatureAveraged # Unit: mm
    
    with open(join(directory, 'segmentInfoDict.pkl'), 'wb') as f:
        pickle.dump(segmentInfoDict, f, 2)
        print('segmentInfoDict.pkl saved to {}.'.format(directory))
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

def reduceGraph(G, segmentList, segmentIndexList):
    """
    Reduce the graph such that the node is either terminating or bifurcating point. Essentially, each segment indexed by
    `segmentIndexList` w.r.t `segmentList` will be reduced to an edge in the new graph. All of the edge and node
    properties in the old graph `G` will be kept.

    Parameters
    ----------
    G : NetworkX graph
        The graph corresponding to `segmentList`.
    segmentList : list
        A list containing all the segments.
    segmentIndexList : list
        A list of indices of segments w.r.t `segmentList` that are present in the graph `G`.
    
    Returns
    -------
    DG : NetworkX graph
        The reduced graph with all the node and edge properties kept.
    """
    DG = nx.DiGraph()
    for segmentIndex in segmentIndexList:
        segment = segmentList[segmentIndex]
        head, tail, secondNode = segment[0], segment[-1], segment[1]
        headLevel, tailLevel = G.node[head]['depthLevel'], G.node[tail]['depthLevel']
        if headLevel > tailLevel:
            head, tail, secondNode = tail, head, segment[-2]
            headLevel, tailLevel = tailLevel, headLevel
        
        DG.add_path([head, tail])
        for key, value in G[head][secondNode].items():
            DG[head][tail][key] = value
        
        for key, value in G.node[head].items():
            DG.node[head][key] = value
        
        for key, value in G.node[tail].items():
            DG.node[tail][key] = value

    return DG

def statisticsPerPartition():
    """
    Output morphological properties per compartment.
    """
    print('Running statisticsPerPartition...')
    start_time = timeit.default_timer()
    directory = os.path.abspath(os.path.dirname(__file__))
    spacing = 0.00040 # meter/voxel

    # Load files #
    result = loadBasicFiles(directory=directory)
    G, segmentList, partitionInfo = itemgetter('G', 'segmentList', 'partitionInfo')(result)

    print('Overall statistics:')
    segmentListOverall = []
    for partitionName, info in partitionInfo.items():
        segmentListOverall += [segmentList[ii] for ii in info['segmentIndexList']]

    nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentListOverall, spacing=spacing, skipUncategorizedVoxels=True)
    
    for partitionName, info in partitionInfo.items():
        print('Partition {}'.format(partitionName))
        segmentListPartition = [segmentList[ii] for ii in info['segmentIndexList']]
        nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentListPartition, spacing=spacing, skipUncategorizedVoxels=True)
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

def statisticsPerPartition2():
    '''
    Output morphological properties of three partitions: PCA (LPCA+RPCA), MCA (LMCA+RMCA), ACA
    '''
    print('Running statisticsPerPartition2...')
    start_time = timeit.default_timer()
    directory = os.path.abspath(os.path.dirname(__file__))
    spacing = 0.00040 # meter/voxel

    # Load files #
    result = loadBasicFiles(directory=directory)
    G, segmentList, partitionInfo = itemgetter('G', 'segmentList', 'partitionInfo')(result)

    with open(join(directory, 'partitionInfo.pkl'), 'rb') as f:
        partitionInfo = pickle.load(f)
    
    segmentListACA = [segmentList[ii] for ii in (list(partitionInfo['LPCA']['segmentIndexList']) +  list(partitionInfo['RPCA']['segmentIndexList']))]
    print('Partition PCA:')
    print((list(partitionInfo['LPCA']['segmentIndexList']) +  list(partitionInfo['RPCA']['segmentIndexList'])))
    nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentListACA, spacing=spacing, skipUncategorizedVoxels=True)

    segmentListMCA = [segmentList[ii] for ii in (list(partitionInfo['LMCA']['segmentIndexList']) +  list(partitionInfo['RMCA']['segmentIndexList']))]
    print('Partition MCA:')
    print((list(partitionInfo['LMCA']['segmentIndexList']) +  list(partitionInfo['RMCA']['segmentIndexList'])))
    nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentListMCA, spacing=spacing, skipUncategorizedVoxels=True)

    segmentListPCA = [segmentList[ii] for ii in list(partitionInfo['ACA']['segmentIndexList'])]
    print('Partition ACA:')
    print(list(partitionInfo['ACA']['segmentIndexList']))
    nodeInfoDict, segmentInfoDict = calculateProperty(G, segmentListPCA, spacing=spacing, skipUncategorizedVoxels=True)

    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

def createPlots():
    """
    Create plots that are used in our paper.
    """
    start_time = timeit.default_timer()
    # Load files #
    directory = os.path.abspath(os.path.dirname(__file__))
    result = loadBasicFiles(directory=directory)
    G, segmentList, segmentInfoDict, nodeInfoDict = itemgetter('G', 'segmentList', 'segmentInfoDict', 'nodeInfoDict')(result)
    chosenVoxels, partitionInfo, resultADANDict = itemgetter('chosenVoxels', 'partitionInfo', 'resultADANDict')(result)
    
    fig1(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig2(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig3(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig4(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig5(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig6(segmentInfoDict, nodeInfoDict, isLastFigure=False)
    fig11(segmentInfoDict, nodeInfoDict, isLastFigure=False) # radius vs graph level # GBM_Radius vs Graph level_Compartment (4)
    fig11b(segmentInfoDict, nodeInfoDict, isLastFigure=False) # radius vs graph level # GBM_Radius vs Graph level_Compartment (5)
    fig12(segmentInfoDict, nodeInfoDict, isLastFigure=False) # curvature distribution
    fig13(segmentInfoDict, nodeInfoDict, isLastFigure=False) # max curvature vs graph level
    fig18(segmentInfoDict, nodeInfoDict, isLastFigure=False) # mean curvature vs branch length

    plt.show()
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

def fig1(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Figure S1, subplot 1-8
    """
    partitionNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    actualNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    fig = plt.figure(1, figsize=(15, 8))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)

    subplotCounter = 1
    # Path length distribution per partition
    valuesList = []
    bincentersList, yList = [], []
    for partitionName in partitionNames:
        segmentsInfoList = [segmentInfo for _, segmentInfo in segmentInfoDict.items() if 'partitionName' in segmentInfo and segmentInfo['partitionName'] == partitionName]
        values = [segmentInfo['pathLength']*0.25 for segmentInfo in segmentsInfoList] # mm
        valuesList.append(values)
        weights = (np.zeros_like(values) + 1. / len(values)).tolist()
        y,binEdges = np.histogram(values, weights=weights)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        bincentersList.append(bincenters)
        yList.append(y)
    
    ax = fig.add_subplot(2, 4, 1)
    # ax.hist(valuesList, label=actualNames)
    for ii in range(len(yList)):
        bincenters, y = bincentersList[ii], yList[ii]
        ax.plot(bincenters, y, 'o-', label=actualNames[ii])
    ax.legend(loc='upper right')
    ax.set_xlabel('Branch Length (mm)')
    ax.set_ylabel('Frequency')
    subplotCounter += 1

    # graph level distribution per partition
    valuesList = []
    bincentersList, yList = [], []
    for partitionName in partitionNames:
        nodesInfoList = [nodeInfo for _, nodeInfo in nodeInfoDict.items() if 'partitionName' in nodeInfo and 'depthLevel' in nodeInfo and nodeInfo['partitionName'] == partitionName]
        values = [nodeInfo['depthLevel'] for nodeInfo in nodesInfoList]
        valuesList.append(values)
        weights = (np.zeros_like(values) + 1. / len(values)).tolist()
        y,binEdges = np.histogram(values, weights=weights)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        bincentersList.append(bincenters)
        yList.append(y)
    
    ax = fig.add_subplot(2, 4, 2)
    # ax.hist(valuesList, label=partitionNames)
    for ii in range(len(yList)):
        bincenters, y = bincentersList[ii], yList[ii]
        ax.plot(bincenters, y, 'o-', label=actualNames[ii])
    ax.legend(loc='upper right')
    ax.set_xlabel('Graph Level')
    ax.set_ylabel('Frequency')

    # number of bifurcations vs graph level per partition
    attribute1, attribute2 = 'depthLevel', 'type'
    attribute1ValuesList, attribute2ValuesList = [], []
    for partitionName in partitionNames:
        nodesInfoList = [nodeInfo for _, nodeInfo in nodeInfoDict.items() if 'partitionName' in nodeInfo and attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo['partitionName'] == partitionName]
        attribute1Values = [nodeInfo[attribute1] for nodeInfo in nodesInfoList]
        attribute2Values = [nodeInfo[attribute2] for nodeInfo in nodesInfoList]
        attribute1ValuesList.append(attribute1Values)
        attribute2ValuesList.append(attribute2Values)
    
    ax = fig.add_subplot(2, 4, 3)
    mf.linePlot(attribute1ValuesList, attribute2ValuesList, ax, bins='auto', integerBinning=True, statistic='count', xlabel='Graph Level', ylabel='# of nodes', legendLabelList=actualNames)

    # number of bifurcations vs graph level (left/right)
    names = [['LMCA', 'LPCA'], ['RMCA', 'RPCA']]
    attribute1, attribute2 = 'depthLevel', 'type'
    attribute1ValuesList, attribute2ValuesList = [], []
    for partitionName in names:
        nodesInfoList = [nodeInfo for _, nodeInfo in nodeInfoDict.items() if 'partitionName' in nodeInfo and attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo['partitionName'] in partitionName]
        attribute1Values = [nodeInfo[attribute1] for nodeInfo in nodesInfoList]
        attribute2Values = [nodeInfo[attribute2] for nodeInfo in nodesInfoList]
        attribute1ValuesList.append(attribute1Values)
        attribute2ValuesList.append(attribute2Values)
    
    ax = fig.add_subplot(2, 4, 4)
    mf.linePlot(attribute1ValuesList, attribute2ValuesList, ax, bins='auto', integerBinning=True, statistic='count', xlabel='Graph Level', ylabel='# of nodes', legendLabelList=['Left', 'Right'])

    # voxel level distribution per partition
    valuesList = []
    bincentersList, yList = [], []
    for partitionName in partitionNames:
        nodesInfoList = [nodeInfo for _, nodeInfo in nodeInfoDict.items() if 'partitionName' in nodeInfo and 'depthVoxel' in nodeInfo and nodeInfo['partitionName'] == partitionName]
        values = [nodeInfo['depthVoxel'] for nodeInfo in nodesInfoList]
        valuesList.append(values)
        weights = (np.zeros_like(values) + 1. / len(values)).tolist()
        y,binEdges = np.histogram(values, weights=weights)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        bincentersList.append(bincenters)
        yList.append(y)
    
    ax = fig.add_subplot(2, 4, 5)
    # ax.hist(valuesList, label=partitionNames)
    for ii in range(len(yList)):
        bincenters, y = bincentersList[ii], yList[ii]
        ax.plot(bincenters, y, 'o-', label=actualNames[ii])
    ax.legend(loc='upper right')
    ax.set_xlabel('Voxel Level')
    ax.set_ylabel('Frequency')
    
    # number of bifurcations vs graph level (terminating/bifurcating)
    attribute1, attribute2 = 'depthLevel', 'type'
    attribute1ValuesList, attribute2ValuesList = [], []
    typeList = ['terminating', 'bifurcating']
    for typeName in typeList:
        nodesInfoList = [nodeInfo for _, nodeInfo in nodeInfoDict.items() if attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo[attribute2] == typeName]
        attribute1Values = [nodeInfo[attribute1] for nodeInfo in nodesInfoList]
        attribute2Values = [nodeInfo[attribute2] for nodeInfo in nodesInfoList]
        attribute1ValuesList.append(attribute1Values)
        attribute2ValuesList.append(attribute2Values)
    
    ax = fig.add_subplot(2, 4, 6)
    mf.linePlot(attribute1ValuesList, attribute2ValuesList, ax, bins='auto', integerBinning=True, statistic='count', xlabel='Graph Level', ylabel='# of nodes', legendLabelList=['Terminating', 'Bifurcating'])
    # aa = [node for node, nodeInfo in nodeInfoDict.items() if attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo[attribute2] == 'bifurcating']
    # bb = [node for node, nodeInfo in nodeInfoDict.items() if attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo[attribute2] == 'terminating']
    # print(len(aa), len(bb))

    # mean radius distribution per partition
    valuesList = []
    attribute = 'meanRadius'
    dictUsed = segmentInfoDict
    bincentersList, yList = [], []
    for partitionName in partitionNames:
        infoList = [info for _, info in dictUsed.items() if 'partitionName' in info and attribute in info and info['partitionName'] == partitionName]
        values = [info[attribute]*0.25 for info in infoList] # mm
        valuesList.append(values)
        weights = (np.zeros_like(values) + 1. / len(values)).tolist()
        y,binEdges = np.histogram(values, weights=weights)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        bincentersList.append(bincenters)
        yList.append(y)
    
    ax = fig.add_subplot(2, 4, 7)
    # ax.hist(valuesList, label=partitionNames)
    for ii in range(len(yList)):
        bincenters, y = bincentersList[ii], yList[ii]
        ax.plot(bincenters, y, 'o-', label=actualNames[ii])
    ax.legend(loc='upper right')
    ax.set_xlabel('Mean radius (mm)')
    ax.set_ylabel('Frequency')
    
    # mean radius distribution for left/right brain
    valuesList = []
    attribute = 'meanRadius'
    names = [['LMCA', 'LPCA'], ['RMCA', 'RPCA']]
    dictUsed = segmentInfoDict
    weightsList = []
    for partitionName in names:
        infoList = [info for _, info in dictUsed.items() if 'partitionName' in info and attribute in info and info['partitionName'] in partitionName]
        values = [info[attribute]*0.25 for info in infoList] # mm
        valuesList.append(values)
        weightsList.append((np.zeros_like(values) + 1. / len(values)).tolist())
    
    ax = fig.add_subplot(2, 4, 8)
    n, bins, _ = ax.hist(valuesList, weights=weightsList, label=['Left', 'Right'])
    # print(n)
    ax.legend(loc='upper right')
    ax.set_xlabel('Mean radius (mm)')
    ax.set_ylabel('Frequency')

    if isLastFigure:
        plt.show()

def fig2(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Figure S1B
    """
    fig = plt.figure(2, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
    # branch length vs graph level
    attribute1, attribute2 = 'segmentLevel', 'pathLength'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info] # mm
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Branch length (mm)', xTickLabelSize=7)

    # branch length (terminating) vs graph level
    attribute1, attribute2, attribute3 = 'segmentLevel', 'pathLength', 'type'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating']
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating'] # mm
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Terminating branch length (mm)', xTickLabelSize=7)
    pathLengthTerminating = attribute2List
    
    # branch length (bifurcating) vs graph level (2D/3D)
    attribute1, attribute2, attribute3 = 'segmentLevel', 'pathLength', 'type'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating']
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating'] # mm
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Bifurcating branch length (mm)', xTickLabelSize=7)
    pathLengthBifurcating = attribute2List

    # voxel level vs gaph level
    attribute1, attribute2 = 'depthLevel', 'depthVoxel'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Voxel level', xTickLabelSize=7)

    # one sided T-test between branch length of bifurcating/terminating vessels (t < 0 means less than relationship)
    # ttest_ind performs two-sided T test and the resulting p value needs to be halfed
    tValue, pValue = ttest_ind(pathLengthBifurcating, pathLengthTerminating)
    meanPathLengthTerminating = np.mean(pathLengthTerminating)
    meanPathLengthBifurcating = np.mean(pathLengthBifurcating)
    factor = (meanPathLengthTerminating - meanPathLengthBifurcating) / meanPathLengthBifurcating
    print('Path length between bifurcating and terminating branches: t = {}, p = {} (t < 0 means less than relationship), factor = {}.'.format(tValue, pValue/2, factor))

    if isLastFigure:
        plt.show()

def fig3(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Figure S1C
    """
    fig = plt.figure(3, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
    # tortuosity (terminating) vs graph level
    attribute1, attribute2, attribute3 = 'segmentLevel', 'tortuosity', 'type'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating']
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating']
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Terminating tortuosity', xTickLabelSize=7)

    # tortuosity (bifurcating) vs graph level
    attribute1, attribute2, attribute3 = 'segmentLevel', 'tortuosity', 'type'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating']
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating']
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Bifurcating tortuosity', xTickLabelSize=7)
    tortuosityBifurcating = attribute2List

    # voxel level (terminating) vs graph level
    attribute1, attribute2, attribute3 = 'depthLevel', 'pathDistance', 'type'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating']
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'terminating']
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Voxel level', ylabel='Terminating path distance (mm)', xTickLabelSize=7)
    tortuosityTerminating = attribute2List

    # voxel level (bifurcating) vs graph level
    attribute1, attribute2, attribute3 = 'depthLevel', 'pathDistance', 'type'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating']
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == 'bifurcating']
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Voxel level', ylabel='Bifurcating path distance (mm)', xTickLabelSize=7)

    # one sided T-test between branch length of bifurcating/terminating vessels (t < 0 means less than relationship)
    # ttest_ind performs two-sided T test and the resulting p value needs to be halfed
    tValue, pValue = ttest_ind(tortuosityBifurcating, tortuosityTerminating)
    meanTortuosityBifurcating = np.mean(tortuosityBifurcating)
    meanTortuosityTerminating = np.mean(tortuosityTerminating)
    factor = (meanTortuosityTerminating - meanTortuosityBifurcating) / meanTortuosityBifurcating
    print('Tortuosity between bifurcating and terminating branches: t = {}, p = {} (t < 0 means less than relationship), factor = {}'.format(tValue, pValue/2, factor))

    if isLastFigure:
        plt.show()

def fig4(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Figure S1D
    """
    fig = plt.figure(3, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
    # localBifurcationAmplitude vs graph level
    attribute1, attribute2 = 'depthLevel', 'localBifurcationAmplitude'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Local bifurcation amplitude (deg)', xTickLabelSize=7)
    localBifurcationAmplitudes = attribute2List

    # remoteBifurcationAmplitude vs graph level
    attribute1, attribute2 = 'depthLevel', 'remoteBifurcationAmplitude'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Remote bifurcation amplitude (deg)', xTickLabelSize=7)
    remoteBifurcationAmplitudes = attribute2List

    # local bifurcation tilt vs graph level
    attribute1, attribute2 = 'depthLevel', 'localBifurcationTilt'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Local bifurcation tilt (deg)', xTickLabelSize=7)

    # remote bifurcation tilt vs graph level
    attribute1, attribute2 = 'depthLevel', 'remoteBifurcationTilt'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Remote bifurcation tilt (deg)', xTickLabelSize=7)
    
    # one sided T-test between local/remote bifurcation amplitude (t < 0 means less than relationship)
    # ttest_ind performs two-sided T test and the resulting p value needs to be halfed
    tValue, pValue = ttest_ind(localBifurcationAmplitudes, remoteBifurcationAmplitudes)
    meanLocalBifurcationAmplitude = np.mean(localBifurcationAmplitudes)
    meanRemoteBifurcationAmplitude = np.mean(remoteBifurcationAmplitudes)
    factor = (meanRemoteBifurcationAmplitude - meanLocalBifurcationAmplitude) / meanLocalBifurcationAmplitude
    print('Local/Remote bifurcating amplitude: t = {}, p = {} (t < 0 means less than relationship), factor = {}'.format(tValue, pValue/2, factor))

    if isLastFigure:
        plt.show()

def fig5(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Not used in our paper.
    """
    fig = plt.figure(3, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
    # localBifurcationTorque vs graph level
    attribute1, attribute2 = 'segmentLevel', 'localBifurcationTorque'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Local bifurcation torque (deg)', xTickLabelSize=7)

    # aspect ratio vs graph level
    attribute1, attribute2 = 'segmentLevel', 'aspectRatio'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) 

    ax = fig.add_subplot(1, 4, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Aspect ratio', xTickLabelSize=7)

    # length ratio vs graph level
    attribute1, attribute2 = 'depthLevel', 'lengthRatio'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Length ratio', xTickLabelSize=7)

    # min radius ratio vs graph level
    attribute1, attribute2 = 'depthLevel', 'minRadiusRatio'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Min radius ratio', xTickLabelSize=7)
    
    if isLastFigure:
        plt.show()

def fig6(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Figure S1E
    """
    fig = plt.figure(3, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
    # max radius ratio vs graph level
    attribute1, attribute2 = 'depthLevel', 'maxRadiusRatio'
    dictUsed = nodeInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Max radius ratio', xTickLabelSize=7)

    # mean radius vs branch length
    attribute1, attribute2 = 'pathLength', 'meanRadius'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    # y, binEdges = np.histogram(attribute1List, bins=np.linspace(0, 1.05*np.max(attribute1List), num=8))
    binEdges = np.ceil(np.linspace(0, 1.01*np.max(attribute1List), num=10))
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    binIndices = np.digitize(attribute1List, bins=binEdges)
    # positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for binIndex in np.unique(binIndices):
        locs = np.nonzero(binIndices == binIndex)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=bincenters, whis='range', xlabel='Branch length (mm)', ylabel='Mean radius (mm)', xTickLabelSize=7)
    ax.set_xlim([0,1.01*np.max(attribute1List)])
    # ax.tick_params(axis='x', rotation=70)

    # sigma vs branch length
    attribute1, attribute2 = 'pathLength', 'sigma'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    binEdges = np.ceil(np.linspace(0, 1.01*np.max(attribute1List), num=10))
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    binIndices = np.digitize(attribute1List, bins=binEdges)
    # positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for binIndex in np.unique(binIndices):
        locs = np.nonzero(binIndices == binIndex)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=bincenters, whis='range', xlabel='Branch length (mm)', ylabel='Mean branch radius sigma (mm)', xTickLabelSize=7)
    ax.set_xlim([0,1.01*np.max(attribute1List)])
    # ax.tick_params(axis='x', rotation=70)

    # mean radius vs sigma
    attribute1, attribute2 = 'meanRadius', 'sigma'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    attribute2List = [info[attribute2]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info]
    binEdges = np.linspace(0, np.max(attribute1List), num=10)
    bincenters = np.round(0.5*(binEdges[1:]+binEdges[:-1]), 2)
    binIndices = np.digitize(attribute1List, bins=binEdges)
    # positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for binIndex in np.unique(binIndices):
        locs = np.nonzero(binIndices == binIndex)[0]
        values.append((attribute2Array[locs]).tolist()) # mm

    ax = fig.add_subplot(1, 4, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=bincenters, whis='range', xlabel='Mean branch radius (mm)', ylabel='Mean branch radius sigma (mm)', xTickLabelSize=7)
    ax.set_xlim([0,1.05*np.max(attribute1List)])
    # ax.tick_params(axis='x', rotation=70)

    if isLastFigure:
        plt.show()

def fig11(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Radius vs graph level per compartment (4)
    """
    fig = plt.figure(11, figsize=(10, 6))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.08, wspace=0.3, hspace=0.4)

    # mean radius vs branch length
    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'LMCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.25*attribute2Array[locs]).tolist()) # Unit: mm for radius
    ax = fig.add_subplot(2, 2, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    ax.set_ylim(0.2, 1.6)
    ax.set_title('LMCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'RMCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.25*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(2, 2, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    ax.set_ylim(0.2, 1.6)
    ax.set_title('RMCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'LPCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.25*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(2, 2, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    ax.set_ylim(0.2, 1.6)
    ax.set_title('LPCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'RPCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.25*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(2, 2, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    ax.set_ylim(0.2, 1.6)
    ax.set_title('RPCA')

    if isLastFigure:
        plt.show()

def fig11b(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Radius vs graph level per compartment (5)
    """
    fig = plt.figure(11, figsize=(15, 3))
    plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)

    # mean radius vs branch length
    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'LMCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.4*attribute2Array[locs]).tolist()) # Unit: mm for radius
    ax = fig.add_subplot(1, 5, 1)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    # ax.set_ylim(0.2,2.5)
    ax.set_title('LMCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'RMCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.4*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(1, 5, 2)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    # ax.set_ylim(0.2,2.5)
    ax.set_title('RMCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'LPCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.4*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(1, 5, 3)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    # ax.set_ylim(0.2,2.5)
    ax.set_title('LPCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'RPCA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.4*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(1, 5, 4)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    # ax.set_ylim(0.2,2.5)
    ax.set_title('RPCA')

    attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
    partitionName = 'ACA'
    dictUsed = segmentInfoDict
    attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
    positions = np.sort(np.unique(attribute1List))
    values = []
    attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
    for segmentLevel in positions:
        locs = np.nonzero(attribute1Array == segmentLevel)[0]
        values.append((0.4*attribute2Array[locs]).tolist())
    ax = fig.add_subplot(1, 5, 5)
    mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean radius (mm)')
    # ax.set_ylim(0.2,2.5)
    ax.set_title('ACA')

    if isLastFigure:
        plt.show()

def fig12(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Curvature distribution
    """
    datasetName = 'Ron'
    ## Curvature ##
    fig = plt.figure(12, figsize=(15, 8))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
    partitionNames = ['LMCA', 'RMCA', 'LPCA', 'RPCA', 'ACA']
    subplotIndex = 1

    ## curvature distribution per partition ##
    for partitionName in partitionNames:
        attribute1, attribute2 = 'meanCurvatureAveragedInmm', 'partitionName'
        dictUsed = segmentInfoDict
        attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and info[attribute2] == partitionName] # 
        ax = fig.add_subplot(2, 4, subplotIndex)
        ax.hist(attribute1List)
        ax.set_xlabel('Mean curvature (mm^-1)')
        ax.set_ylabel('Count')
        ax.hist(attribute1List)
        ax.set_title('{}, {}'.format(partitionName, datasetName))

        subplotIndex += 1
    
    if isLastFigure:
        plt.show()

def fig13(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Max curvature vs graph level
    """
    datasetName = 'Ron'
    ## Curvature ##
    fig = plt.figure(13, figsize=(15, 8))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
    partitionNames = ['LMCA', 'RMCA', 'LPCA', 'RPCA', 'ACA']
    subplotIndex = 1

    ## curvature vs graph level per partition ##
    for partitionName in partitionNames:
        attribute1, attribute2, attribute3 = 'segmentLevel', 'meanCurvatureAveragedInmm', 'partitionName'
        dictUsed = segmentInfoDict
        attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName]
        attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName] # 
        positions = np.sort(np.unique(attribute1List))
        values = []
        attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
        for segmentLevel in positions:
            locs = np.nonzero(attribute1Array == segmentLevel)[0]
            values.append((attribute2Array[locs]).tolist())

        ax = fig.add_subplot(2, 4, subplotIndex)
        mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Mean curvature (mm^-1)')
        ax.set_title('{}, {}'.format(partitionName, datasetName))

        subplotIndex += 1
    
    if isLastFigure:
        plt.show()

def fig18(segmentInfoDict, nodeInfoDict, isLastFigure=True):
    """
    Mean curvature vs graph level
    """
    datasetName = 'Ron'
    ## Curvature ##
    fig = plt.figure(18, figsize=(15, 8))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
    partitionNames = ['LMCA', 'RMCA', 'LPCA', 'RPCA', 'ACA']
    subplotIndex = 1

    ## curvature vs graph level per partition ##
    for partitionName in partitionNames:
        attribute1, attribute2, attribute3 = 'pathLength', 'meanCurvatureAveragedInmm', 'partitionName'
        dictUsed = segmentInfoDict
        attribute1List = [info[attribute1]*0.25 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName] # mm
        attribute2List = [info[attribute2] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] == partitionName] # 
        positions = np.sort(np.unique(attribute1List))

        ax = fig.add_subplot(2, 4, subplotIndex)
        ax.plot(attribute1List, attribute2List, 'bo')
        ax.set_xlabel('Branch length (mm)')
        ax.set_ylabel('Mean curvature (mm^-1)')
        ax.set_title('{}, {}'.format(partitionName, datasetName))

        subplotIndex += 1
    
    if isLastFigure:
        plt.show()

def plotNetwork(G, infoDict, figIndex=1, isLastFigure=True, hideColorbar=False):
    """
    Plot the graph G in a tree structure. The color of the nodes and edges reflects corresponding values.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be plot.
    infoDict : dict
        A dictionary containing necessary information for plotting.
    figIndex : int, optional
        The figure index.
    isLastFigure : bool, optional
        If True, `plt.show()` will be executed.
    hideColorbar : bool, optional
        If True, the colorbars will be hidden.
    """
    ## Unpack infoDict ##
    nodeLabelDict, nodeValueList = itemgetter('nodeLabelDict', 'nodeValueList')(infoDict)
    edgeLabelDict, edgeValueList = itemgetter('edgeLabelDict', 'edgeValueList')(infoDict)
    figTitle, nodeColorbarLabel, edgeColorbarLabel = itemgetter('figTitle', 'nodeColorbarLabel', 'edgeColorbarLabel')(infoDict)

    ## Calculate extra info ##
    if 'vmin' not in infoDict or 'vmax' not in infoDict:
        vmin, vmax = np.amin(nodeValueList), np.amax(nodeValueList)
    else:
        vmin, vmax = itemgetter('vmin', 'vmax')(infoDict)

    if 'edge_vmin' not in infoDict or 'edge_vmax' not in infoDict:
        edge_vmin, edge_vmax = np.amin(edgeValueList), np.amax(edgeValueList)
    else:
        edge_vmin, edge_vmax = itemgetter('edge_vmin', 'edge_vmax')(infoDict)

    ## Plot ##
    fig = plt.figure(figIndex, figsize=(15, 8))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
    
    pos = graphviz_layout(G, prog='dot')
    ax = fig.add_axes([0.05, 0.05, 0.7, 0.9])
    ax.set_title(figTitle)
    ax.set_axis_off()
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, node_color=nodeValueList, cmap=plt.cm.get_cmap('jet'), vmin=vmin, vmax=vmax)
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=10, edge_color=edgeValueList, edge_cmap=plt.cm.get_cmap('jet'), edge_vmin=edge_vmin, edge_vmax=edge_vmax, width=2)
    if len(nodeLabelDict) != 0:
        nx.draw_networkx_labels(G, pos, labels=nodeLabelDict, font_size=8)
    
    if len(edgeLabelDict) != 0:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabelDict, font_size=8)
    
    # node colorbar
    if len(nodeColorbarLabel) != 0 and not hideColorbar:
        # plt.colorbar(nodes, cmap=plt.cm.jet, label=nodeColorbarLabel) 
        ax1 = fig.add_axes([0.8, 0.05, 0.03, 0.9])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=plt.cm.get_cmap('jet'), norm=norm, orientation='vertical')
        cb1.set_label(nodeColorbarLabel, size=10)
        cb1.ax.tick_params(labelsize=10)
    # edge colorbar
    if len(edgeColorbarLabel) != 0 and not hideColorbar:
        ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.9])
        norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.cm.get_cmap('jet'), norm=norm, orientation='vertical')
        cb2.set_label(edgeColorbarLabel, size=10)
        cb2.ax.tick_params(labelsize=10)
    
    if isLastFigure:
        plt.show()

def graphPlotPerPartition():
    """
    An example using the function `plotNetwork`.
    """
    start_time = timeit.default_timer()
    spacing = 0.00040 # meter/voxel
    # Load files #
    directory = os.path.abspath(os.path.dirname(__file__))
    result = loadBasicFiles(directory=directory)
    G, segmentList, segmentInfoDict, nodeInfoDict = itemgetter('G', 'segmentList', 'segmentInfoDict', 'nodeInfoDict')(result)
    chosenVoxels, partitionInfo, resultADANDict = itemgetter('chosenVoxels', 'partitionInfo', 'resultADANDict')(result)

    subplotCounter = 1
    datasetName = 'GBM'
    for partitionName in partitionInfo.keys():
        entryPoints = chosenVoxels[partitionName]['initialVoxels']
        boundaryPoints = chosenVoxels[partitionName]['boundaryVoxels']
        allVoxels = partitionInfo[partitionName]['visitedVoxels']
        segmentIndexList = partitionInfo[partitionName]['segmentIndexList']
    
        GSub = G.subgraph(allVoxels)
        DGSub = reduceGraph(GSub, segmentList, segmentIndexList)
        nodeLabelDict = {}
        nodeValueList = [nodeInfoDict[node]['cubicLawResult'] if 'cubicLawResult' in nodeInfoDict[node] else 0 for node in DGSub.nodes()]
        edgeLabelDict = {}
        edgeValueList = [segmentInfoDict[DGSub[edge[0]][edge[1]]['segmentIndex']]['meanRadius'] * spacing * 1000 for edge in DGSub.edges()] # mm
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Murray\'s law ratio',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Radius (mm)',
                    'figTitle': '{}, {}'.format(partitionName, datasetName)}
        
        graphPlot(DGSub, infoDict, figIndex=subplotCounter)
        subplotCounter += 1
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))
    plt.show()

def graphPlotPerPartition2():
    """
    An example using the function `plotNetwork`. Show Murray's law ratio and radius for all five compartments that share one colorbar
    """
    start_time = timeit.default_timer()
    directory = os.path.abspath(os.path.dirname(__file__))
    datasetName = 'GBM'
    ## Load ##
    result = loadBasicFiles()
    G, segmentList, segmentInfoDict, nodeInfoDict = itemgetter('G', 'segmentList', 'segmentInfoDict', 'nodeInfoDict')(result)
    chosenVoxels, partitionInfo, resultADANDict = itemgetter('chosenVoxels', 'partitionInfo', 'resultADANDict')(result)

    subplotCounter = 1
    infoDictList = []
    nodeValueListList, edgeValueListList = [], []
    allVoxelsList, DGSubList = [], []
    partitionNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    actualNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    for partitionName, actualName in zip(partitionNames, actualNames):
        entryPoints = chosenVoxels[partitionName]['initialVoxels']
        boundaryPoints = chosenVoxels[partitionName]['boundaryVoxels']
        allVoxels = partitionInfo[partitionName]['visitedVoxels']
        segmentIndexList = partitionInfo[partitionName]['segmentIndexList']
        allVoxelsList += allVoxels
    
        GSub = G.subgraph(allVoxels)
        DGSub = reduceGraph(GSub, segmentList, segmentIndexList)
        DGSubList.append(DGSub)
        nodeLabelDict = {}
        nodeValueList = [nodeInfoDict[node]['cubicLawResult'] if 'cubicLawResult' in nodeInfoDict[node] else 0 for node in DGSub.nodes()]
        edgeLabelDict = {}
        edgeValueList = [segmentInfoDict[DGSub[edge[0]][edge[1]]['segmentIndex']]['meanRadius'] * 0.4 for edge in DGSub.edges()] # mm
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Murray\'s law ratio',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Radius (mm)',
                    'figTitle': '{}, {}'.format(actualName, datasetName)}
        
        infoDictList.append(infoDict)
        nodeValueListList += nodeValueList
        edgeValueListList += edgeValueList
    
    vmin, vmax = np.amin(nodeValueListList), np.amax(nodeValueListList)
    edge_vmin, edge_vmax = np.amin(edgeValueListList), np.amax(edgeValueListList)
    
    for infoDict in infoDictList:
        tempDict = {'vmin': vmin, 'vmax': vmax, 'edge_vmin': edge_vmin, 'edge_vmax': edge_vmax}
        infoDict.update(tempDict)

    hideColorbarList = [True, True, True, True, True]
    for ii in range(len(partitionNames)):
        infoDict = infoDictList[ii]
        DGSub = DGSubList[ii]
        hideColorbar = hideColorbarList[ii]
        plotNetwork(DGSub, infoDict, hideColorbar=hideColorbar, figIndex=ii+1)
    
    # Standalone color bar
    fig = plt.figure(10, figsize=(12, 8))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.94, bottom=0.06, wspace=0.3, hspace=0.9)

    ax1 = fig.add_axes([0.15, 0.9, 0.7, 0.04])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=mpl.cm.jet, norm=norm, orientation='horizontal')
    cb1.set_label('Murray\'s law ratio', size=18)
    cb1.ax.tick_params(labelsize=18)

    ax2 = fig.add_axes([0.15, 0.75, 0.7, 0.04])
    norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=mpl.cm.jet, norm=norm, orientation='horizontal')
    cb2.set_label('Mean branch radius (mm)', size=18)
    cb2.ax.tick_params(labelsize=18)

    plt.show()


if __name__ == "__main__":
    generateInfoDict()
    calculateCurvature()
    statisticsPerPartition()
    statisticsPerPartition2()
    createPlots()
    graphPlotPerPartition()
    graphPlotPerPartition2()