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


def fig1():
    """
    Figure S1, subplot 1-8
    """
    # Load files #
    directory = os.path.abspath(os.path.dirname(__file__))
    result = loadBasicFiles(directory=directory)
    G, segmentList, segmentInfoDict, nodeInfoDict = itemgetter('G', 'segmentList', 'segmentInfoDict', 'nodeInfoDict')(result)
    chosenVoxels, partitionInfo, resultADANDict = itemgetter('chosenVoxels', 'partitionInfo', 'resultADANDict')(result)
    
    partitionNames = ['LCA', 'RCA', 'PCA', 'LACA', 'RACA']
    partitionNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    actualNames = ['LMCA', 'RMCA', 'ACA', 'LPCA', 'RPCA']
    fig = plt.figure(1, figsize=(15, 8))
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
    names = [['LCA', 'LACA'], ['RCA', 'RACA']]
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
    aa = [node for node, nodeInfo in nodeInfoDict.items() if attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo[attribute2] == 'bifurcating']
    bb = [node for node, nodeInfo in nodeInfoDict.items() if attribute1 in nodeInfo and attribute2 in nodeInfo and nodeInfo[attribute2] == 'terminating']
    print(len(aa), len(bb))

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
    names = [['LCA', 'LACA'], ['RCA', 'RACA']]
    dictUsed = segmentInfoDict
    weightsList = []
    for partitionName in names:
        infoList = [info for _, info in dictUsed.items() if 'partitionName' in info and attribute in info and info['partitionName'] in partitionName]
        values = [info[attribute]*0.25 for info in infoList] # mm
        valuesList.append(values)
        weightsList.append((np.zeros_like(values) + 1. / len(values)).tolist())
    
    ax = fig.add_subplot(2, 4, 8)
    n, bins, _ = ax.hist(valuesList, weights=weightsList, label=['Left', 'Right'])
    print(n)
    ax.legend(loc='upper right')
    ax.set_xlabel('Mean radius (mm)')
    ax.set_ylabel('Frequency')
    

    plt.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)