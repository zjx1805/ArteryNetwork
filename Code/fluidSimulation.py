import sys, os
import numpy as np
from numpy.linalg import norm
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import logging
import traceback
import timeit
import time
import math
from ast import literal_eval as make_tuple
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import glob
import pickle
import myFunctions as mf
import copy
from operator import itemgetter
from os.path import join
import inspect
from scipy.optimize import fsolve, fmin_tnc, least_squares, differential_evolution, minimize, fmin_l_bfgs_b, basinhopping
import myFunctions as mf
from scipy import stats

class FluidNetwork(object):
    """
    Unified framework for doing the fluid simulation. At this stage, the graph used has already been reduced, i.e., each
    edge represens a segment in `segmentList` and each node represents a bifurcation. Previously, each segment may be
    consisted of one or more edges. To reduce the previous graph, use the function `reduceGraph`. Also, for the sake of
    consistency, the original `segmentInfoDict` has been renamed to `edgeInfoDict`, `segmentList` to `edgeList`, but
    `nodeInfoDict` remains the same. Besides, the nodes and edges are now indexed by integers starting from zero for
    simplicity. Use the function `convertGraph` to do the conversion.

    So the `fundemental staff` that you need to have are: `edgeList`, `edgeInfoDict`, `nodeInfoDict`, `G`. These are 
    necessary to all of the subsquent analysis. To perform a blood flow simulation, you need to do the following steps:

    1. Get the graph and the `fundemental staff` by either creating one or loading an existing one.
    2. Set c and k (used in H-W equation) for each edge by `setNetwork`.
    3. Set the terminating pressures by `setTerminatingPressure`.
    4. Generate H-W equations for each edge and flow conservation equations for each node by `setupFluidEquations`.
    5. Solve the equations by optimization and use `computerNetworkDetail` as objective function.

    The fluid simulation tries to solve the network by finding a set of pressures for each node and a set of flow rates 
    for each edges such that H-W equations and flow conservation equations are satisfied with the given set of 
    terminating pressures. For a binary tree structure without merges, a solution is guaranteed to exist no matter what 
    the terminating pressures look like. However, for the network with merges (e.g., the GBM network with CoW), it is 
    possible that a solution does not exist for the given set of terminating pressures. Therefore, for these cases, we 
    need to check the optimization result and check whether the error in each equations are within a acceptable range.

    Note that not all the functions in this class should be used. Some are just for experimental purposes!
    """
    def __init__(self):
        self.directory = os.path.abspath(os.path.dirname(__file__))
        self.edgeList = []
        self.edgeIndexList = []
        self.G = nx.Graph()
        self.rootNode = 0
        self.edgeInfoDict = {}
        self.nodeInfoDict = {}
        self.nodeIndex = 0 
        self.edgeIndex = 0
        self.spacing = 0.00040 # meter/voxel
        self.eqnInfoDictList = []
        self.velocityPressure = []
        self.velocityPressureGroundTruth = []
        self.distributeFlowEqnDict = {}
        self.nodeInfoDictBefore = {}
        self.nodeInfoDictAfter = {}
        self.edgeInfoDictBefore = {}
        self.edgeInfoDictAfter = {}

    def generateNetwork(self, maxDepth=10, allowMerge=False):
        """
        Generate a binary tree with random edge and node properties.

        Parameters
        ----------
        maxDepth : int, optional
            Maximum depth of the graph (depth start from zero).
        allowMerge : bool, optional
            If True, there will be 30% change that two edges at the same depth will merge together.
        """
        G = nx.Graph()
        nodeDepth, edgeDepth = 0, 0
        G.add_node(0, depth=nodeDepth, depthLevel=nodeDepth, nodeIndex=self.nodeIndex, isEntryNode=True) # first node
        self.nodeIndex += 1
        while nodeDepth <= maxDepth - 1:
            nodesAtCurrentDepth = [node for node in G.nodes() if G.node[node]['depth'] == nodeDepth]
            if len(nodesAtCurrentDepth) > 2:
                # Determine if merge would happen
                if allowMerge:
                    mergeAtCurrentDepth = (np.random.rand() <= 0.3) # 30% probability TODO: this should be controlled by function arguments
                else:
                    mergeAtCurrentDepth = False
                
                # Merge nodes if allowed 
                if mergeAtCurrentDepth:
                    numOfMerges = 1 # TODO: this should be controlled by function arguments
                    nodesToMerge = np.random.choice(nodesAtCurrentDepth, 2, replace=False)
                    newNode = self.nodeIndex
                    newEdgeIndex1, newEdgeIndex2 = self.edgeIndex, self.edgeIndex + 1 # TODO: allow >2 edge merge?
                    G.add_edge(nodesToMerge[0], newNode, depth=edgeDepth, segmentLevel=edgeDepth, edgeIndex=self.edgeIndex, segmentIndex=self.edgeIndex)
                    G.add_edge(nodesToMerge[1], newNode, depth=edgeDepth, segmentLevel=edgeDepth, edgeIndex=self.edgeIndex + 1, segmentIndex=self.edgeIndex + 1)
                    G.node[newNode]['depth'] = nodeDepth + 1
                    G.node[newNode]['depthLevel'] = nodeDepth + 1
                    G.node[newNode]['nodeIndex'] = self.nodeIndex
                    G.node[newNode]['isEntryNode'] = False

                    self.nodeIndex += 1
                    self.edgeIndex += 2
            
            for currentNode in nodesAtCurrentDepth:
                numOfChildEdges = len([node for node in G[currentNode].keys() if G.node[node]['depth'] > nodeDepth])
                numOfNewEdges = 2 - numOfChildEdges # TODO: allow for more child edges?
                for ii in range(numOfNewEdges):
                    newNode = self.nodeIndex
                    G.add_edge(currentNode, newNode, depth=edgeDepth, segmentLevel=edgeDepth, edgeIndex=self.edgeIndex, segmentIndex=self.edgeIndex)
                    G.node[newNode]['depth'] = nodeDepth + 1
                    G.node[newNode]['depthLevel'] = nodeDepth + 1
                    G.node[newNode]['nodeIndex'] = self.nodeIndex
                    G.node[newNode]['isEntryNode'] = False

                    self.nodeIndex += 1
                    self.edgeIndex += 1
            
            nodeDepth += 1
            edgeDepth += 1
        
        # Gather data
        edgeList = [0] * self.edgeIndex
        for edge in G.edges():
            edgeIndex = G[edge[0]][edge[1]]['edgeIndex']
            edgeList[edgeIndex] = edge

        nodeIndexList = [G.node[node]['nodeIndex'] for node in G.nodes()]
        edgeIndexList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in edgeList]
        nodeInfoDict, edgeInfoDict = {}, {}
        for node in G.nodes():
            nodeInfoDict[node] = G.node[node]
            nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            nodeInfoDict[node]['coord'] = []
        
        for edge in G.edges():
            edgeIndex = G[edge[0]][edge[1]]['edgeIndex']
            edgeInfoDict[edgeIndex] = G[edge[0]][edge[1]]
            edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset

        # Save
        self.G = G
        self.edgeList = edgeList
        self.nodeIndexList = nodeIndexList
        self.edgeIndexList = edgeIndexList
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
    
    def loadNetwork(self, version=2, year=2013):
        """
        Load old version of data files (that needs to be converted).
        """
        directory = self.directory
        if version == 1:
            filename = 'basicFilesForStructureWithCoW(year={}).pkl'.format(year)
        elif version == 2:
            filename = 'basicFilesForStructureWithCoW2(year={}).pkl'.format(year)
        elif version == 3:
            filename = 'basicFilesForStructureWithCoW3(year={}).pkl'.format(year)
        elif version == 4:
            filename = 'basicFilesForStructureWithCoW4(year={}).pkl'.format(year)

        with open(join(directory, filename), 'rb') as f:
            resultDict = pickle.load(f)
        
        with open(join(directory, 'partitionInfo.pkl'), 'rb') as f:
            partitionInfo = pickle.load(f)
        
        with open(join(directory, 'chosenVoxelsForPartition.pkl'), 'rb') as f:
            chosenVoxels = pickle.load(f)
        
        ADANFolder = os.path.abspath(join(directory, '../../../../'))
        with open(join(ADANFolder, 'ADAN-Web/resultADANDict.pkl'), 'rb') as f:
            resultADANDict = pickle.load(f)
        
        resultDict['resultADANDict'] = resultADANDict
        resultDict['partitionInfo'] = partitionInfo
        resultDict['chosenVoxels'] = chosenVoxels

        self.loadedNetwork = resultDict
    
    def reduceGraph(self, G, segmentList, segmentIndexList):
        """
        Reduce the graph such that the node is either terminating or bifurcating point.

        Parameters
        ----------
        G : NetworkX graph
            The graph representation of the network.
        segmentList : list
            A list of segments in which each segment is a simple branch.
        segmentIndexList : list
            A list of segment indices referring to the segments actually be used in `segmentList`.
        
        Returns
        -------
        DG : NetworkX graph
            The reduced graph (each edge refers to a segment).
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

    def convertNetowrk(self):
        """
        Convert the old version of data files into the new version used here.
        """
        resultDict = self.loadedNetwork
        GOld, segmentList, partitionInfo, chosenVoxels, segmentInfoDictOld, nodeInfoDictOld, resultADANDict = itemgetter('G', 'segmentList', 'partitionInfo', 'chosenVoxels', 'segmentInfoDict', 'nodeInfoDict', 'resultADANDict')(resultDict)
        segmentIndexList = list(segmentInfoDictOld.keys())
        heartLoc = (255, 251, 26) # change as needed
        DG = self.reduceGraph(GOld, segmentList, segmentIndexList)

        G = nx.Graph()
        nodeInfoDict, edgeInfoDict = {}, {}
        nodeIndex, edgeIndex = 0, 0
        maxNodeDepth = np.max([DG.node[node]['depthLevel'] for node in DG.nodes()])
        for currentDepth in range(maxNodeDepth + 1):
            nodesAtCurrentDepth = [node for node in DG.nodes() if DG.node[node]['depthLevel'] == currentDepth]
            for node in nodesAtCurrentDepth:
                G.add_node(nodeIndex, depth=DG.node[node]['depthLevel'], nodeIndex=nodeIndex, coord=node)
                DG.node[node]['nodeIndexHere'] = nodeIndex
                if node == heartLoc:
                    G.node[nodeIndex]['isEntryNode'] = True
                    rootNode = nodeIndex
                else:
                    G.node[nodeIndex]['isEntryNode'] = False
                
                nodeIndex += 1
        
        for edge in DG.edges():
            depth = np.min([DG.node[edge[0]]['depthLevel'], DG.node[edge[1]]['depthLevel']])
            DG[edge[0]][edge[1]]['depth'] = depth
        
        maxEdgeDepth = np.max([DG[edge[0]][edge[1]]['depth'] for edge in DG.edges()])
        for currentDepth in range(maxEdgeDepth + 1):
            edgesAtCurrentDepth = [edge for edge in DG.edges() if DG[edge[0]][edge[1]]['depth'] == currentDepth]
            for edge in edgesAtCurrentDepth:
                G.add_edge(DG.node[edge[0]]['nodeIndexHere'], DG.node[edge[1]]['nodeIndexHere'], depth=currentDepth, edgeIndex=edgeIndex)
                edgeIndex += 1
        
        currentNodeIndex = nodeIndex
        currentEdgeIndex = edgeIndex

        edgeList = [[]] * edgeIndex
        for edge in G.edges():
            edgeIndex = G[edge[0]][edge[1]]['edgeIndex']
            edgeList[edgeIndex] = edge

        nodeIndexList = [G.node[node]['nodeIndex'] for node in G.nodes()]
        edgeIndexList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in edgeList]

        for node in DG.nodes():
            nodeIndexHere = DG.node[node]['nodeIndexHere']
            nodeInfoDict[nodeIndexHere] = DG.node[node]
            nodeInfoDict[nodeIndexHere]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            nodeInfoDict[nodeIndexHere]['coord'] = []
        
        for edge in DG.edges():
            edgeIndex = G[DG.node[edge[0]]['nodeIndexHere']][DG.node[edge[1]]['nodeIndexHere']]['edgeIndex']
            segmentIndex = DG[edge[0]][edge[1]]['segmentIndex']
            edgeInfoDict[edgeIndex] = DG[edge[0]][edge[1]]
            edgeInfoDict[edgeIndex]['length'] = DG[edge[0]][edge[1]]['pathLength'] # backward compatibility
            edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        # Sync between G and nodeInfoDict
        for node in G.nodes():
            for key, value in G.node[node].items():
                nodeInfoDict[node][key] = value

        # Save
        self.G = G
        self.edgeIndex = currentEdgeIndex
        self.nodeIndex = currentNodeIndex
        self.edgeList = edgeList
        self.nodeIndexList = nodeIndexList
        self.edgeIndexList = edgeIndexList
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
        self.rootNode = rootNode
    
    def adjustNetwork(self):
        """
        If the network changes, recheck the correspondence between branch name and edgeIndex!
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        
        # LICA(Pre)
        edgeInfoDict[0]['meanRadius'] = 3.3 / (spacing * 1000) # mm->voxel
        edgeInfoDict[0]['length'] = 1.5 / (spacing * 1000) # mm->voxel

        # LICA(Post)
        edgeInfoDict[3]['meanRadius'] = 3.3 / (spacing * 1000) # mm->voxel
        edgeInfoDict[3]['length'] = 1.5 / (spacing * 1000) # mm->voxel

        # RICA(Pre)
        edgeInfoDict[2]['meanRadius'] = 3.3 / (spacing * 1000) # mm->voxel
        edgeInfoDict[2]['length'] = 1.5 / (spacing * 1000) # mm->voxel

        # RICA(Post)
        edgeInfoDict[7]['meanRadius'] = 3.3 / (spacing * 1000) # mm->voxel
        edgeInfoDict[7]['length'] = 1.5 / (spacing * 1000) # mm->voxel

        # VA
        # edgeInfoDict[1]['meanRadius'] = 2.0 / (spacing * 1000) # mm->voxel
        edgeInfoDict[1]['length'] = 28 / (spacing * 1000) # mm->voxel

        # RPCAComm
        edgeInfoDict[4]['length'] = 16 / (spacing * 1000) # mm->voxel

        # RMCA(first segment)
        # edgeInfoDict[12]['length'] = 8 / (spacing * 1000) # mm->voxel
        
        # Save
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
    
    def setNetwork(self, option=1, extraInfo=None):
        """
        Set c and k (and possibly radius and length) for each branch
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        # Use BraVa data to set the radius and ADAN result to set the c and k 
        if option == 1:
            minSetLength, maxSetLength = 1, 70 # mm
            # Params used to fit radius to edgeLevel using the BraVa data. radius (mm) = a * np.exp(-b * edgeLevel) + c
            fitResultDict = {'LMCA': {'param': [0.5569, 0.4199, 0.469]}, 'RMCA': {'param': [0.6636, 0.3115, 0.3666]}, 'LPCA': {'param': [0.6571, 0.3252, 0.2949]}, 'RPCA': {'param': [0.7103, 0.5587, 0.3815]}, 'ACA': {'param': [0.3604, 1.0538, 0.4714]}} # new names
            # fitResultDict = {'LCA': {'param': [0.5569, 0.4199, 0.469]}, 'RCA': {'param': [0.6636, 0.3115, 0.3666]}, 'LACA': {'param': [0.6571, 0.3252, 0.2949]}, 'RACA': {'param': [0.7103, 0.5587, 0.3815]}, 'PCA': {'param': [0.3604, 1.0538, 0.4714]}} # old names
            a, b, c = fitResultDict['LMCA']['param']
            
            for edgeIndex in edgeIndexList:
                edgeLevel = edgeInfoDict[edgeIndex]['depth']
                radius = (a * np.exp(-b * edgeLevel) + c) / (spacing * 1000) # voxel
                edgeInfoDict[edgeIndex]['meanRadius'] = radius # voxel
                length = (np.random.rand() * (maxSetLength - minSetLength) + minSetLength) / (spacing * 1000) # voxel
                edgeInfoDict[edgeIndex]['pathLength'] = length # for backward compatibility
                edgeInfoDict[edgeIndex]['length'] = length # voxel

            ADANFolder = os.path.abspath(join(directory, '../../../../'))
            with open(join(ADANFolder, 'ADAN-Web/resultADANDict.pkl'), 'rb') as f:
                resultADANDict = pickle.load(f)
                print('resultADANDict.pkl loaded from {}'.format(ADANFolder))
            
            slopeCRadius, interceptCRadius = resultADANDict['slopeCRadius'], resultADANDict['interceptCRadius']
            radiusThresholds, CKCandidates, numOfCCategory = resultADANDict['radiusThresholds'], resultADANDict['CKCandidates'], resultADANDict['numOfCCategory']
            minRadius, maxRadius = np.min(radiusThresholds), np.max(radiusThresholds) # meter
            slopePressureRadius, interceptPressureRadius = resultADANDict['slopePressureRadius'], resultADANDict['interceptPressureRadius']
            for edgeIndex in edgeIndexList:
                edge = edgeList[edgeIndex]
                radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing
                if radius > minRadius and radius < maxRadius:
                    binIndex = np.digitize([radius], radiusThresholds)[0] - 1
                    c, k = CKCandidates[binIndex], CKCandidates[-1] # assuming c is different for each branch and k is the same
                    edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k'] = c, k
                else:
                    c = np.poly1d([slopeCRadius, interceptCRadius])(radius) # extrapolate
                    k = CKCandidates[-1] # assuming c is different for each branch and k is the same
                    c = c if c > 0 else 0.1
                    edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k'] = c, k
        
        # Only set c and k using ADAN result
        elif option == 2:
            ADANFolder = os.path.abspath(join(directory, '../../../../'))
            with open(join(ADANFolder, 'ADAN-Web/resultADANDict.pkl'), 'rb') as f:
                resultADANDict = pickle.load(f)
                print('resultADANDict.pkl loaded from {}'.format(ADANFolder))
            
            if extraInfo is not None:
                excludedEdgeIndex = itemgetter('excludedEdgeIndex')(extraInfo)
            
            slopeCRadius, interceptCRadius = resultADANDict['slopeCRadius'], resultADANDict['interceptCRadius']
            # print('slopeCRadius={}, interceptCRadius={}'.format(slopeCRadius, interceptCRadius))
            radiusThresholds, CKCandidates, numOfCCategory = resultADANDict['radiusThresholds'], resultADANDict['CKCandidates'], resultADANDict['numOfCCategory']
            minRadius, maxRadius = np.min(radiusThresholds), np.max(radiusThresholds) # meter
            slopePressureRadius, interceptPressureRadius = resultADANDict['slopePressureRadius'], resultADANDict['interceptPressureRadius']
            # if extraInfo is not None:
            #     edgeIndexListToUse = [edgeIndex for edgeIndex in edgeIndexList if edgeIndex not in excludedEdgeIndex]
            # else:
            #     edgeIndexListToUse = edgeIndexList
            edgeIndexListToUse = edgeIndexList

            for edgeIndex in edgeIndexListToUse:
                edge = edgeList[edgeIndex]
                radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing
                if radius > minRadius and radius < maxRadius:
                    binIndex = np.digitize([radius], radiusThresholds)[0] - 1
                    c, k = CKCandidates[binIndex], CKCandidates[-1] # assuming c is different for each branch and k is the same
                    c = np.poly1d([slopeCRadius, interceptCRadius])(radius) # extrapolate
                    edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k'] = c, k
                else:
                    c = np.poly1d([slopeCRadius, interceptCRadius])(radius) # extrapolate
                    k = CKCandidates[-1] # assuming c is different for each branch and k is the same
                    # c = c if c > 0 else 0.1
                    if radius * 1000 >= 1.5 and radius * 1000 <= 2.5:
                        c = 1
                    else:
                        if c < 0:
                            c = 0.1
                    edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k'] = c, k
        

        # Save
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
    
    def showFlowInfo(self):
        """
        Print out flow rates for selected edges and pressure for selected nodes.
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        for edgeIndex in range(16):
            flow = edgeInfoDict[edgeIndex]['simulationData']['flow']
            radius, length, c, k = itemgetter('meanRadius', 'length', 'c', 'k')(edgeInfoDict[edgeIndex])
            if flow is not None:
                flow *= 10**6 # convert to cm^3/s
            else:
                flow = -1 # 
            radius *= (spacing * 100) # convert to cm
            length *= (spacing * 100) # convert to cm
            print('Edge {}: flow={:.3f} cm^3/s, radius={:.4f} cm, length={:.4f} cm, c={:.4f}, k={:.4f}'.format(edgeIndex, flow, radius, length, c, k))
        
        print('\n')
        for node in range(16):
            flow, pressure = itemgetter('flow', 'pressure')(nodeInfoDict[node]['simulationData'])
            if flow is not None:
                flow *= 10**6 # convert to cm^3/s
            else:
                flow = -1

            if pressure is not None:
                pressure /= (13560*9.8/1000) # convert to mmHg
            else:
                pressure = -1

            print('Node {}: flow={:.3f} cm^3/s, pressure={:.3f} mmHg'.format(node, flow, pressure))

    def getFlowInfoFromDeltaPressure(self, edgeIndex, deltaPressure):
        """
        Calculate the required flow/velocity in order to achieve the given pressure drop for the specific edge.

        Parameters
        ----------
        edgeIndex : int
            The index of the edge.
        deltaPressure : float
            The desired pressure drop with a unit of Pascal.
        
        Returns
        -------
        flow : float
            The required flow rate to achieve the desired pressure drop with a unit of cm^3/s.
        velocity : float
            The velocity in that edge corresponding to the required flow rate.
        """
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
        length = edgeInfoDict[edgeIndex]['length'] * spacing # meter
        c, k = itemgetter('c', 'k')(edgeInfoDict[edgeIndex])
        flow = np.power(deltaPressure * c**k * (2*radius)**4.8704 / 10.67 / length, 1/k) # m^3/s
        velocity = flow / (np.pi * radius**2) # m/s

        return flow, velocity
    
    def getDeltaPressureFromFlow(self, edgeIndex, flow):
        """
        Calculate the required pressure drop in order to achieve the given flow for the specific edge.

        Parameters
        ----------
        edgeIndex : int
            The index of the edge.
        flow : float
            The desired flow rate of the edge with a unit of cm^3/s.
        
        Returns
        -------
        deltaPressure : float
            The required pressure drop in the edge to achieve the desired flow rate with a unit of Pascal.
        """
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
        length = edgeInfoDict[edgeIndex]['length'] * spacing # meter
        c, k = itemgetter('c', 'k')(edgeInfoDict[edgeIndex])
        deltaPressure = 10.67 * flow**k * length / c**k / (2*radius)**4.8704

        return deltaPressure

    def createGroundTruth(self, seed=None, option=1):
        """
        Manually set the velocity and pressure for all edges/nodes in order to check whether the solver is correct.
        Option 1: each child branch randomly takes ~1/N (with some random fluctuation) of the parent flow.
        Option 2: flow is split proportional to the cross sectional area of the child branches.
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        success = False

        # Set argsIndex (index of pressure/velocity unknowns in the fluid simulation)
        argsIndex = 0
        for edgeIndex in edgeIndexList:
            edgeInfoDict[edgeIndex]['argsIndex'] = argsIndex
            argsIndex += 1

        for node in G.nodes():
            nodeInfoDict[node]['isBifurcatingNode'] = False

        nodeList = [node for node in G.nodes() if node != 0 and G.degree(node) != 1]
        for node in nodeList:
            nodeInfoDict[node]['argsIndex'] = argsIndex
            nodeInfoDict[node]['isBifurcatingNode'] = True
            argsIndex += 1

        minSetVelocity, maxSetVelocity = 0.01, 3 # m/s
        inletPressure = 13560 * 9.8 * 0.12 # Pascal
        inletVelocity = 1.5 # m/s
        inletFlow = 754/60/10**6 # m^3/s
        minSplitAmout, maxSplitAmout = 0.4, 0.6
        maxDepth = np.max([info['depth'] for node, info in nodeInfoDict.items()])
        for currentDepth in range(maxDepth):
            ## first deal with the nodes whose child edge merges
            nodesAtNextDepth = [node for node in G.nodes() if nodeInfoDict[node]['depth'] == currentDepth + 1]
            for nodeAtNextDepth in nodesAtNextDepth:
                parentNodes = [node for node in G[nodeAtNextDepth].keys() if nodeInfoDict[node]['depth'] == currentDepth]
                # parentNodes = [node for node in G[nodeAtNextDepth].keys() if nodeInfoDict[node]['depth'] < nodeInfoDict[nodeAtNextDepth]['depth']]
                if len(parentNodes) > 1:
                    # print('Node {} merge into {}'.format(parentNodes, nodeAtNextDepth))
                    flowAtParentNodes = [nodeInfoDict[node]['simulationData']['flow'] for node in parentNodes] # m^3/s
                    degreeAtParentNodes = [G.degree(node) for node in parentNodes]
                    pressureAtParentNodes = [nodeInfoDict[node]['simulationData']['pressure'] for node in parentNodes]
                    parentEdgeIndexList = [G[nodeAtNextDepth][node]['edgeIndex'] for node in parentNodes]
                    parentEdgeDeltaPressureList = [self.getDeltaPressureFromFlow(edgeIndex, flow) for edgeIndex, flow in zip(parentEdgeIndexList, flowAtParentNodes)]
                    nodeMinPressureList = [headPressure - deltaPressure for headPressure, deltaPressure in zip(pressureAtParentNodes, parentEdgeDeltaPressureList)]
                    if degreeAtParentNodes[0] == 2 and degreeAtParentNodes[1] > 2:
                        loc1, loc2 = 0, 1
                        isEdge1StraightPipe, isEdge2StraightPipe = True, False
                    elif degreeAtParentNodes[0] > 2 and degreeAtParentNodes[1] == 2:
                        loc1, loc2 = 1, 0
                        isEdge1StraightPipe, isEdge2StraightPipe = True, False
                    elif degreeAtParentNodes[0] == 2 and degreeAtParentNodes[1] == 2:
                        loc1, loc2 = 0, 1
                        isEdge1StraightPipe, isEdge2StraightPipe = True, True
                        if nodeMinPressureList[0] != nodeMinPressureList[1]:
                            success = False
                            print('Error! Two straight edges cannot achieve the same end pressure')
                            return success
                        
                        print('Warning! Two straight edges merge into one node')
                    else:
                        if nodeMinPressureList[0] > nodeMinPressureList[1]:
                            loc1, loc2 = 0, 1
                        else:
                            loc1, loc2 = 1, 0
                        
                        isEdge1StraightPipe, isEdge2StraightPipe = False, False
                    
                    edgeIndex1, edgeIndex2 = parentEdgeIndexList[loc1], parentEdgeIndexList[loc2]
                    parentNode1, parentNode2 = parentNodes[loc1], parentNodes[loc2]
                    parentPressure1, parentPressure2 = pressureAtParentNodes[loc1], pressureAtParentNodes[loc2]
                    parentFlow1, parentFlow2 = flowAtParentNodes[loc1], flowAtParentNodes[loc2]
                    radius1, radius2 = edgeInfoDict[edgeIndex1]['meanRadius'] * spacing, edgeInfoDict[edgeIndex2]['meanRadius'] * spacing
                    length1, length2 = edgeInfoDict[edgeIndex1]['length'] * spacing, edgeInfoDict[edgeIndex2]['length'] * spacing
                    c1, c2 = edgeInfoDict[edgeIndex1]['c'], edgeInfoDict[edgeIndex2]['c']
                    k1, k2 = edgeInfoDict[edgeIndex1]['k'], edgeInfoDict[edgeIndex2]['k']
                    
                    flowCounter = 0
                    # for the first edge
                    maxPossibleFlow = parentFlow1
                    minDeltaPressure = np.max([0, pressureAtParentNodes[loc1] - pressureAtParentNodes[loc2]])
                    minPossibleFlow, _ = self.getFlowInfoFromDeltaPressure(parentEdgeIndexList[loc1], minDeltaPressure)
                    if minPossibleFlow > maxPossibleFlow:
                        success = False
                        print('Error while merging node {} to node {}, minPossibleFlow ({}) is larger than maxPossibleFlow ({})'.format(parentNodes, nodeAtNextDepth, minPossibleFlow, maxPossibleFlow))
                        return success
                    
                    if isEdge1StraightPipe:
                        flow1 = parentFlow1
                        if flow1 >= minPossibleFlow and flow1 <= maxPossibleFlow:
                            pass
                        else:
                            print('Edge {} wants to use all flow={} from node {}, but possible range is [{}, {}]'.format(edgeIndex1, flow1, parentNode1, minPossibleFlow, maxPossibleFlow))
                    else:
                        # flow1 = np.random.rand() * (maxPossibleFlow - minPossibleFlow) + minPossibleFlow
                        flow1 = (maxPossibleFlow + minPossibleFlow) / 2
                    
                    ## Manual manipulation !!! ##
                    if nodeAtNextDepth == 10:
                        if edgeIndex1 == 9:
                            flow1 = maxPossibleFlow * 0.15 # used to be 0.3
                            print('Edge {} gets flow={} cm^3/s'.format(edgeIndex1, flow1*10**6))
                        elif edgeIndex1 == 11:
                            flow1 = maxPossibleFlow * 0.15 # used to be 0.3
                            print('Edge {} gets flow={} cm^3/s'.format(edgeIndex1, flow1*10**6))
                        # radius8, radius9 = edgeInfoDict[8]['meanRadius'], edgeInfoDict[9]['meanRadius']
                        # flow9 = maxPossibleFlow * radius9**2 / (radius8**2 + radius9**2)
                        # print('Edge {} get flow={}'.format(edgeIndex1, flow1))

                    velocity1 = flow1 / (np.pi * radius1**2) # m/s
                    edgeInfoDict[edgeIndex1]['simulationData']['velocity'] = velocity1
                    edgeInfoDict[edgeIndex1]['simulationData']['flow'] = flow1
                    deltaPressure1 = 10.67 * flow1**k1 * length1 / c1**k1 / (2*radius1)**4.8704
                    tailPressure = parentPressure1 - deltaPressure1 # pressure at the merging node
                    nodeInfoDict[nodeAtNextDepth]['simulationData']['pressure'] = tailPressure
                    flowCounter += flow1

                    # the other edge
                    deltaPressure2 = parentPressure2 - tailPressure
                    flow2 = np.power(deltaPressure2 / 10.67 / length2 * c2**k2 * (2*radius2)**4.8704, 1/k2)
                    velocity2 = flow2 / (np.pi * radius2**2) # m/s
                    edgeInfoDict[edgeIndex2]['simulationData']['velocity'] = velocity2
                    edgeInfoDict[edgeIndex2]['simulationData']['flow'] = flow2
                    flowCounter += flow2
                    nodeInfoDict[nodeAtNextDepth]['simulationData']['flow'] = flowCounter
                    if flow2 > parentFlow2:
                        print('Node {}: the flow ({}) in other edge is larger than provided ({})'.format(nodeAtNextDepth, flow2, parentFlow2))
                        print('edgeIndex1={}, edgeIndex2={}, flow1={}, flow2={}'.format(edgeIndex1, edgeIndex2, flow1, flow2))
                        print(nodeInfoDict[1]['simulationData']['pressure']/13560/9.8*1000, nodeInfoDict[3]['simulationData']['pressure']/13560/9.8*1000, nodeInfoDict[2]['simulationData']['pressure']/13560/9.8*1000)
            
            ## Now deal with remaining nodes
            nodesAtCurrentDepth = [node for node in G.nodes() if nodeInfoDict[node]['depth'] == currentDepth]
            for currentNode in nodesAtCurrentDepth:
                if currentDepth == 0:
                    nodeInfoDict[currentNode]['simulationData']['pressure'] = inletPressure
                    nodeInfoDict[currentNode]['simulationData']['flow'] = inletFlow
                    flowIn = inletFlow
                    pressureIn = inletPressure
                    # print('inletPressure={} mmHg, inletFlow={} cm^3/s, currentDepth={}'.format(inletPressure/13560/9.8*1000, inletFlow*10**6, currentDepth))
                else:
                    flowIn = nodeInfoDict[currentNode]['simulationData']['flow']
                    if flowIn is None:
                        print('Node {} has flow=None, nodesAtCurrentDepth={}'.format(currentNode, nodesAtCurrentDepth))
                    pressureIn = nodeInfoDict[currentNode]['simulationData']['pressure']
                    
                edgeIndexAtNextDepth = [G[currentNode][neighborNode]['edgeIndex'] for neighborNode in G[currentNode].keys() if nodeInfoDict[neighborNode]['depth'] > currentDepth]
                edgeIndexToProcess = [edgeIndex for edgeIndex in edgeIndexAtNextDepth if edgeInfoDict[edgeIndex]['simulationData']['flow'] is None]
                edgeIndexCompleted = [edgeIndex for edgeIndex in edgeIndexAtNextDepth if edgeInfoDict[edgeIndex]['simulationData']['flow'] is not None]
                edgeCounter = len(edgeIndexToProcess)
                flowAvailable = nodeInfoDict[currentNode]['simulationData']['flow']
                for edgeIndex in edgeIndexCompleted:
                    flowAvailable -= edgeInfoDict[edgeIndex]['simulationData']['flow']
                
                if flowAvailable < 0 - np.finfo(float).eps:
                    flowIn = nodeInfoDict[currentNode]['simulationData']['flow']
                    flowUsed = ['Edge {}: {}'.format(edgeIndex, edgeInfoDict[edgeIndex]['simulationData']['flow']) for edgeIndex in edgeIndexCompleted]
                    print('Error! Node {}: flowIn={}, flowUsed={}, flowAvailable={}'.format(currentNode, flowIn, flowUsed, flowAvailable))
                
                flowAmount = []
                # Random split the flow (within a range)
                if option == 1:
                    while edgeCounter >= 1:
                        if edgeCounter > 1:
                            basePercentage = 100 / edgeCounter
                            fluctuationPercentage = basePercentage / 3.0
                            actualPercentage = basePercentage - fluctuationPercentage/2 + np.random.rand() * fluctuationPercentage
                            # actualPercentage = (np.random.rand() * 0.8 + 0.1) * 100
                            flow = flowAvailable * actualPercentage / 100
                            if flow < 0:
                                print('Node {}: flow < 0, actualPercentage={}, flowAvailable={}'.format(currentNode, actualPercentage, flowAvailable))
                            flowAmount.append(flow)
                            flowAvailable -= flow
                            if flowAvailable < 0:
                                print('Node {}: flowAvailable < 0, actualPercentage={}'.format(currentNode, actualPercentage))
                        else:
                            flowAmount.append(flowAvailable)
                        
                        edgeCounter -= 1
                
                elif option == 2:
                    radiusList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexToProcess]
                    radiusSqList = [radius**2 for radius in radiusList]
                    sumOfRadiusSq = np.sum(radiusSqList)
                    flowAmount = [flowAvailable * radiusSq / sumOfRadiusSq for radiusSq in radiusSqList]
                
                ## Manual manipulation !!! ###
                if currentNode == 0 and G.degree(currentNode) == 3:
                    edgeIndexToProcess = [0, 2, 1] # LICA/RICA/VA
                    inletFlow = nodeInfoDict[currentNode]['simulationData']['flow']
                    flowAmount = [inletFlow*0.4, inletFlow*0.4, inletFlow*0.2]
                # elif currentNode == 8:
                #     edgeIndexToProcess = [16, 17] # 
                #     inletFlow = nodeInfoDict[currentNode]['simulationData']['flow']
                #     flowAmount = [inletFlow*0.7, inletFlow*0.3]
                # elif currentNode == 9:
                #     edgeIndexToProcess = [18, 19] # 
                #     inletFlow = nodeInfoDict[currentNode]['simulationData']['flow']
                #     flowAmount = [inletFlow*0.7, inletFlow*0.3]

                for edgeIndex, flow in zip(edgeIndexToProcess, flowAmount):
                    edge = edgeList[edgeIndex]
                    radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
                    velocity = flow / (np.pi * radius**2) # m/s
                    edgeHead, edgeTail = edge[0], edge[1]
                    if nodeInfoDict[edgeHead]['depth'] > nodeInfoDict[edgeTail]['depth']:
                        edgeHead, edgeTail = edgeTail, edgeHead

                    pressureHead = nodeInfoDict[edgeHead]['simulationData']['pressure']
                    c, k = edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k']
                    length = edgeInfoDict[edgeIndex]['length'] * spacing # meter
                    deltaPressure = 10.67 * (velocity * np.pi * radius**2)**k * length / c**k / (2 * radius)**4.8704 # Pascal
                    if np.isnan(deltaPressure):
                        print('velocity={}, flow={}'.format(velocity, flow))
                    pressureTail = pressureHead - deltaPressure # Pascal

                    nodeInfoDict[edgeTail]['simulationData']['pressure'] = pressureTail
                    nodeInfoDict[edgeTail]['simulationData']['flow'] = flow
                    # print('Node {} (head={}, edgeIndex={}), flow={}'.format(edgeTail, edgeHead, edgeIndex, flow))
                    edgeInfoDict[edgeIndex]['simulationData']['velocity'] = velocity
                    edgeInfoDict[edgeIndex]['simulationData']['flow'] = flow
                    # print('Pressure at {} = {} mmHg, currentDepth={}'.format(edgeTail, pressureTail/13560/9.8*1000, currentDepth))
                    # if edgeIndex ==5 or edgeIndex == 6:
                    #     print('Node {}, edgeIndex={}, flow={} cm^3/s, deltaPressure={} mmHg'.format(currentNode, edgeIndex, flow*10**6, deltaPressure/13560/9.8*1000))
        
        velocityPressure = [0] * argsIndex
        for node in G.nodes():
            if 'argsIndex' in nodeInfoDict[node]:
                argsIndex = nodeInfoDict[node]['argsIndex']
                pressure = nodeInfoDict[node]['simulationData']['pressure']
                velocityPressure[argsIndex] = pressure
        
        for edgeIndex in edgeIndexList:
            if 'argsIndex' in edgeInfoDict[edgeIndex]:
                argsIndex = edgeInfoDict[edgeIndex]['argsIndex']
                velocity = edgeInfoDict[edgeIndex]['simulationData']['velocity']
                velocityPressure[argsIndex] = velocity
        
        # Save
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
        self.velocityPressure = velocityPressure # Ground truth solution
        self.velocityPressureGroundTruth = velocityPressure # Ground truth solution

        success = True
        return success
    
    def getVelocityPressure(self):
        """
        Extract velocity and pressure from edgeInfoDict and nodeInfoDict.

        Returns
        -------
        velocityPressure : list
            A list of velocities and pressures in the form of [v0, v1,..., vN, p0, p1,..., pN].
        """
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        velocityPressure = np.hstack((np.full((numOfEdges,), 0.0), np.full((numOfNodes,), 0.0))) # make sure dtype is float
        for node, info in nodeInfoDict.items():
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                pressure = info['simulationData']['pressure']
                velocityPressure[argsIndex] = pressure
        
        for edgeIndex, info in edgeInfoDict.items():
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                velocity = info['simulationData']['velocity']
                velocityPressure[argsIndex] = velocity
        
        return velocityPressure
    
    def getVolumePerPartition(self):
        """
        Calculate the total volume of each compartment.

        Returns
        volumePerPartition : dict
            A dictionary with compartments names as keys and volumes (with a unit of mm^3) as corresponding values.
        """
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}

        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing    
        volumePerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            totalVolume = 0
            for edgeIndex in visitedEdges:
                radius, length= itemgetter('meanRadius', 'length')(edgeInfoDict[edgeIndex])
                radius = radius * spacing * 1000 # mm
                length = length * spacing * 1000 # mm
                edgeVolume = np.pi * radius**2 * length # mm^3
                totalVolume += edgeVolume
            
            volumePerPartition[partitionName] = totalVolume
        
        return volumePerPartition

    def showTerminatingPressureAndPathLength(self):
        """
        Check terminating pressure vs path length relationship.
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        
        maxDepth = np.max([info['depth'] for node, info in nodeInfoDict.items()])
        terminatingNodes = [node for node in G.nodes() if nodeInfoDict[node]['depth'] == maxDepth]
        terminatingPressure = [nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000 for node in terminatingNodes] # mmHg
        termiantingPathLength = []
        for node in terminatingNodes:
            path = nx.shortest_path(G, self.rootNode, node)
            pathEdgeIndex = [G[path[ii]][path[ii+1]]['edgeIndex'] for ii in range(len(path) - 1)]
            pathLength = np.sum([edgeInfoDict[edgeIndex]['length'] * spacing for edgeIndex in pathEdgeIndex]) # meter
            termiantingPathLength.append(pathLength)
        
        fig = plt.figure(1, figsize=(15, 8))
        plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
        plt.plot(termiantingPathLength, terminatingPressure, 'bo')
        plt.xlabel('Path length (m)')
        plt.ylabel('Terminating pressure (mmHg)')
        plt.show()
    
    def setupFluidEquations(self, boundaryCondition=None):
        """
        Programmatically stores the info to generate the conservation equations used for fluid simulation (each dict represents an equation). 
        
        There are two kinds of equations: H-W equation for each edge and flow conservation equation for each node and optionally boundary 
        conditions. For the H-W equation, the 
        information is stored in a dictionay as:  
        {'type': 'pressure', 'radius': radius, 'length': length, 'velocityIndex': velocityIndex, 'c': c, 'k': k, 'edgeIndex': edgeIndex}  

        For the flow conservation equation, the information is stored as:  
        {'type': 'flow', 'velocityInIndexList': velocityInIndexList, 'radiusInList': radiusInList, 
         'velocityOutIndexList': velocityOutIndexList, 'radiusOutList': radiusOutList, 'coord': nodeInfoDict[node]['coord'], 
         'nodeIndex': nodeInfoDict[node]['nodeIndex'], 'neighborsInEdgeIndex': neighborsIndexIn, 'neighborsOutEdgeIndex': neighborsIndexOut}
        
        For the boundary conditions (inlet or outlet velocity), the information is stored as:  
        {'type': 'boundary', 'velocityIndex': velocityIndex, 'velocityIn': velocityIn}
        
        All of the units are SI units. The dictonaries that hold these equations are then stored in the `eqnInfoDictList`.
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing

        eqnInfoDictList = []
        numOfFlowEqns, numOfPressureEqns, numOfBoundaryConditionEqns = 0, 0, 0
        
        for node in G.nodes():
            if nodeInfoDict[node]['isBifurcatingNode']: 
                neighborsIndexIn = [G[node][neighborIn]['edgeIndex'] for neighborIn in G.neighbors(node) if 'depth' in G.node[neighborIn] and G.node[neighborIn]['depth'] < G.node[node]['depth']]
                neighborsIndexOut = [G[node][neighborOut]['edgeIndex'] for neighborOut in G.neighbors(node) if 'depth' in G.node[neighborOut] and G.node[neighborOut]['depth'] > G.node[node]['depth']]
    
                radiusInList = [edgeInfoDict[neighborIndexIn]['meanRadius'] * spacing for neighborIndexIn in neighborsIndexIn]
                radiusOutList = [edgeInfoDict[neighborIndexOut]['meanRadius'] * spacing for neighborIndexOut in neighborsIndexOut]
                velocityInIndexList = [edgeInfoDict[neighborIndexIn]['argsIndex'] for neighborIndexIn in neighborsIndexIn]
                velocityOutIndexList = [edgeInfoDict[neighborIndexOut]['argsIndex'] for neighborIndexOut in neighborsIndexOut]
                if len(radiusInList) != 0 and len(radiusOutList) != 0: # Exclude the nodes at inlet and outlet
                    eqnInfoDict = {'type': 'flow', 'velocityInIndexList': velocityInIndexList, 'radiusInList': radiusInList, 
                                   'velocityOutIndexList': velocityOutIndexList, 'radiusOutList': radiusOutList, 'coord': nodeInfoDict[node]['coord'], 
                                   'nodeIndex': nodeInfoDict[node]['nodeIndex'], 'neighborsInEdgeIndex': neighborsIndexIn, 'neighborsOutEdgeIndex': neighborsIndexOut}
                    eqnInfoDictList.append(eqnInfoDict)
                    numOfFlowEqns += 1
                else:
                    print('node={}, len(radiusInList)={}, len(radiusOutList)={}'.format(node, len(radiusInList), len(radiusOutList)))
       
        for edgeIndex in edgeIndexList:
            edge = edgeList[edgeIndex]
            radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing
            length = edgeInfoDict[edgeIndex]['length'] * spacing
            velocityIndex = edgeInfoDict[edgeIndex]['argsIndex']
            c, k = edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k']
            eqnInfoDict = {'type': 'pressure', 'radius': radius, 'length': length, 'velocityIndex': velocityIndex, 'c': c, 'k': k, 'edgeIndex': edgeIndex}
    
            if nodeInfoDict[edge[0]]['depth'] < nodeInfoDict[edge[-1]]['depth']:
                headNode, tailNode = edge[0], edge[-1]
            else:
                headNode, tailNode = edge[-1], edge[0]
            
            # head pressure
            if nodeInfoDict[headNode]['isEntryNode'] is True or G.degree(headNode) == 1:
                headPressure = nodeInfoDict[headNode]['simulationData']['pressure']
                eqnInfoDict['headPressureInfo'] = {'pressure': headPressure}
            else:
                headPressureIndex = nodeInfoDict[headNode]['argsIndex']
                headNodeIndex = nodeInfoDict[headNode]['nodeIndex']
                eqnInfoDict['headPressureInfo'] = {'pressureIndex': headPressureIndex, 'nodeIndex': headNodeIndex}
            
            # tail pressure
            if nodeInfoDict[tailNode]['isEntryNode'] is True or G.degree(tailNode) == 1:
                tailPressure = nodeInfoDict[tailNode]['simulationData']['pressure']
                eqnInfoDict['tailPressureInfo'] = {'pressure': tailPressure}
                # print('Tail node {} has pressure={} mmHg'.format(tailNode, tailPressure/13560/9.8*1000))
            else:
                tailPressureIndex = nodeInfoDict[tailNode]['argsIndex']
                tailNodeIndex = nodeInfoDict[tailNode]['nodeIndex']
                eqnInfoDict['tailPressureInfo'] = {'pressureIndex': tailPressureIndex, 'nodeIndex': tailNodeIndex}
            
            eqnInfoDictList.append(eqnInfoDict)
            numOfPressureEqns += 1
        
        if boundaryCondition is not None and len(boundaryCondition) != 0 and 'pressureIn' not in boundaryCondition:
            for boundaryNode, info in boundaryCondition.items():
                edgeIndex = info['edgeIndex']
                velocityIn = info['velocityIn']
                edge = edgeList[edgeIndex]
                velocityIndex = edgeInfoDict[edgeIndex]['argsIndex']
                eqnInfoDict = {'type': 'boundary', 'velocityIndex': velocityIndex, 'velocityIn': velocityIn}
                eqnInfoDictList.append(eqnInfoDict)
                numOfBoundaryConditionEqns += 1
        
        print('There are {} flow eqns, {} pressure eqns and {} boundary condition eqns'.format(numOfFlowEqns, numOfPressureEqns, numOfBoundaryConditionEqns))
    
        self.eqnInfoDictList = eqnInfoDictList
    
    def setupFluidEquationsMatLab(self, boundaryCondition=None):
        """
        Programmatically stores the info to generate the conservation equations used for fluid simulation (each dict represents an equation). 
        
        Note that the Python-MatLab bridge only accepts generic python types, and thus all numpy types need to be converted.
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing

        eqnInfoDictList = []
        numOfFlowEqns, numOfPressureEqns, numOfBoundaryConditionEqns = 0, 0, 0
        
        for node in G.nodes():
            if nodeInfoDict[node]['isBifurcatingNode']: 
                neighborsIndexIn = [G[node][neighborIn]['edgeIndex'] for neighborIn in G.neighbors(node) if 'depth' in G.node[neighborIn] and G.node[neighborIn]['depth'] < G.node[node]['depth']]
                neighborsIndexOut = [G[node][neighborOut]['edgeIndex'] for neighborOut in G.neighbors(node) if 'depth' in G.node[neighborOut] and G.node[neighborOut]['depth'] > G.node[node]['depth']]
    
                radiusInList = [float(edgeInfoDict[neighborIndexIn]['meanRadius'] * spacing) for neighborIndexIn in neighborsIndexIn]
                radiusOutList = [float(edgeInfoDict[neighborIndexOut]['meanRadius'] * spacing) for neighborIndexOut in neighborsIndexOut]
                velocityInIndexList = [int(edgeInfoDict[neighborIndexIn]['argsIndex']) for neighborIndexIn in neighborsIndexIn]
                velocityOutIndexList = [int(edgeInfoDict[neighborIndexOut]['argsIndex']) for neighborIndexOut in neighborsIndexOut]
                if len(radiusInList) != 0 and len(radiusOutList) != 0: # Exclude the nodes at inlet and outlet
                    eqnInfoDict = {'type': 'flow', 'velocityInIndexList': velocityInIndexList, 'radiusInList': radiusInList, 
                                   'velocityOutIndexList': velocityOutIndexList, 'radiusOutList': radiusOutList, 'coord': nodeInfoDict[node]['coord'], 
                                   'nodeIndex': int(nodeInfoDict[node]['nodeIndex']), 'neighborsInEdgeIndex': neighborsIndexIn, 'neighborsOutEdgeIndex': neighborsIndexOut}
                    eqnInfoDictList.append(eqnInfoDict)
                    numOfFlowEqns += 1
                else:
                    print('node={}, len(radiusInList)={}, len(radiusOutList)={}'.format(node, len(radiusInList), len(radiusOutList)))
       
        for edgeIndex in edgeIndexList:
            edge = edgeList[edgeIndex]
            radius = float(edgeInfoDict[edgeIndex]['meanRadius'] * spacing)
            length = float(edgeInfoDict[edgeIndex]['length'] * spacing)
            velocityIndex = int(edgeInfoDict[edgeIndex]['argsIndex'])
            c, k = float(edgeInfoDict[edgeIndex]['c']), float(edgeInfoDict[edgeIndex]['k'])
            eqnInfoDict = {'type': 'pressure', 'radius': radius, 'length': length, 'velocityIndex': velocityIndex, 'c': c, 'k': k, 'edgeIndex': int(edgeIndex)}
    
            if nodeInfoDict[edge[0]]['depth'] < nodeInfoDict[edge[-1]]['depth']:
                headNode, tailNode = edge[0], edge[-1]
            else:
                headNode, tailNode = edge[-1], edge[0]
            
            # head pressure
            if nodeInfoDict[headNode]['isEntryNode'] is True or G.degree(headNode) == 1:
                headPressure = float(nodeInfoDict[headNode]['simulationData']['pressure'])
                eqnInfoDict['headPressureInfo'] = {'pressure': headPressure}
            else:
                headPressureIndex = int(nodeInfoDict[headNode]['argsIndex'])
                headNodeIndex = int(nodeInfoDict[headNode]['nodeIndex'])
                eqnInfoDict['headPressureInfo'] = {'pressureIndex': headPressureIndex, 'nodeIndex': headNodeIndex}
            
            # tail pressure
            if nodeInfoDict[tailNode]['isEntryNode'] is True or G.degree(tailNode) == 1:
                tailPressure = float(nodeInfoDict[tailNode]['simulationData']['pressure'])
                eqnInfoDict['tailPressureInfo'] = {'pressure': tailPressure}
            else:
                tailPressureIndex = int(nodeInfoDict[tailNode]['argsIndex'])
                tailNodeIndex = int(nodeInfoDict[tailNode]['nodeIndex'])
                eqnInfoDict['tailPressureInfo'] = {'pressureIndex': tailPressureIndex, 'nodeIndex': tailNodeIndex}
            
            eqnInfoDictList.append(eqnInfoDict)
            numOfPressureEqns += 1
        
        if boundaryCondition is not None and len(boundaryCondition) != 0 and 'pressureIn' not in boundaryCondition:
            for boundaryNode, info in boundaryCondition.items():
                edgeIndex = int(info['edgeIndex'])
                velocityIn = float(info['velocityIn'])
                edge = edgeList[edgeIndex]
                velocityIndex = int(edgeInfoDict[edgeIndex]['argsIndex'])
                eqnInfoDict = {'type': 'boundary', 'velocityIndex': velocityIndex, 'velocityIn': velocityIn}
                eqnInfoDictList.append(eqnInfoDict)
                numOfBoundaryConditionEqns += 1
        
        print('There are {} flow eqns, {} pressure eqns and {} boundary condition eqns'.format(numOfFlowEqns, numOfPressureEqns, numOfBoundaryConditionEqns))
    
        self.eqnInfoDictList = eqnInfoDictList
    
    def setupEquationsForDistributeFlow(self):
        """
        Setup equations for distributeFlowTest(). This function is unfinished. TODO

        The resulting file is distributeFlowEqnDict and it contains three fields:
        -- 'connectInfoDictList' --
        It is a list of dicts and each dict represents an edge and it contains: 
            -- 'connection' -- In the form of [headNode, edgeIndex, tailNode]
            -- 'edgeInfo' -- Contains subfields 'c'/'k'/'radius'/'length'
        -- 'mergeInfoDict' -- 
        Each merging node is a key and the corresponding value is empty (for now)
        -- 'desiredTerminatingPressures' --
        Each terminating node is a key and the corresponding value is the desired terminating pressure for that node
        """
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        edgeList = self.edgeList
        spacing = self.spacing
        distributeFlowEqnDict = {'connectInfoDictList': [], 'mergeInfoDict': {}, 'desiredTerminatingPressures': {}}

        edgeDepthArray = np.array([edgeInfoDict[edgeIndex]['depth'] for edgeIndex in edgeIndexList])
        edgeIndexListSorted = np.array(edgeIndexList)[edgeDepthArray.argsort()].tolist()

        for edgeIndex in edgeIndexListSorted:
            edge = edgeList[edgeIndex]
            headNode, tailNode = edge
            if nodeInfoDict[headNode]['depth'] > nodeInfoDict[tailNode]['depth']:
                headNode, tailNode = tailNode, headNode
            
            radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing
            length = edgeInfoDict[edgeIndex]['length'] * spacing
            c, k = edgeInfoDict[edgeIndex]['c'], edgeInfoDict[edgeIndex]['k']

            distributeFlowEqnDict['connectInfoDictList'].append({'connection': [headNode, edgeIndex, tailNode], 'edgeInfo': {'radius': radius, 'length': length, 'c': c, 'k': k}})
        
        for currentNode in G.nodes():
            parentNodes = [node for node in G[currentNode].keys() if nodeInfoDict[node]['depth'] < nodeInfoDict[currentNode]['depth']]
            if len(parentNodes) > 1:
                distributeFlowEqnDict['mergeInfoDict'][currentNode] = {}
        
        for node in G.nodes():
            if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0:
                distributeFlowEqnDict['desiredTerminatingPressures'][node] = 13560*9.8*0.12 # Pascal
        
        print(edgeIndexListSorted)
        print(distributeFlowEqnDict['mergeInfoDict'])

        # Save #
        self.distributeFlowEqnDict = distributeFlowEqnDict

    def validateFluidEquations(self, velocityPressure=None, boundaryCondition=None):
        """
        Validate if all of the equations generated by `setupFluidEquations` are satisfied. This function will output errors for 
        each of the equations and corresponding details. Note that the error for each equations is amplified in the same way as 
        in the function `computeNetworkDetail`.

        Parameters
        ----------
        velocityPressure : list
            A list of velocities and pressures in the form of [v0, v1,..., vN, p0, p1,..., pN].
        """
        directory = self.directory
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        eqnInfoDictList = self.eqnInfoDictList
        if velocityPressure is None:
            velocityPressure = self.velocityPressure

        counter = 0
        pressureErrorList, flowErrorList = [], []
        pressureErrorTrueList, flowErrorTrueList = [], []
        for eqnInfoDict in eqnInfoDictList:
            eqnType = eqnInfoDict['type']
            if eqnType == 'pressure':
                radius, length, velocityIndex, edgeIndex = itemgetter('radius', 'length', 'velocityIndex', 'edgeIndex')(eqnInfoDict)
                velocity = np.abs(velocityPressure[velocityIndex])
                c, k = eqnInfoDict['c'], eqnInfoDict['k']
                if 'pressure' in eqnInfoDict['headPressureInfo']:
                    headPressure = eqnInfoDict['headPressureInfo']['pressure']
                elif 'pressureIndex' in eqnInfoDict['headPressureInfo']:
                    pressureIndex = eqnInfoDict['headPressureInfo']['pressureIndex']
                    headPressure = velocityPressure[pressureIndex]
                
                headPressureInmmHg = headPressure / 13560 / 9.8 * 1000
    
                if 'pressure' in eqnInfoDict['tailPressureInfo']:
                    tailPressure = eqnInfoDict['tailPressureInfo']['pressure']
                elif 'pressureIndex' in eqnInfoDict['tailPressureInfo']:
                    pressureIndex = eqnInfoDict['tailPressureInfo']['pressureIndex']
                    tailPressure = velocityPressure[pressureIndex]
                
                tailPressureInmmHg = tailPressure / 13560 / 9.8 * 1000
                
                deltaPressureByNode = np.abs(headPressure - tailPressure)
                deltaPressureByHW = 10.67 * (velocity * np.pi * radius**2)**k * length / c**k / (2 * radius)**4.8704
                error = np.abs(deltaPressureByNode - deltaPressureByHW)
                deltaPressureByHWInmmHg = deltaPressureByHW / 13560 / 9.8 * 1000
                errorInmmHg = error / 13560 / 9.8 * 1000
                pressureErrorList.append(errorInmmHg * 500)
                pressureErrorTrueList.append(errorInmmHg)
                print('error={:.4f} mmHg, headP={:.2f} mmHg, tailP={:.2f} mmHg, headP>tailP={}, deltaPByHW={:.2f} mmHg, velocity={:.3f} cm/s, radius={:.4f} cm, length={:.4f} cm, edgeIndex={}'.format(errorInmmHg, 
                    headPressureInmmHg, tailPressureInmmHg, headPressure>tailPressure, deltaPressureByHWInmmHg, velocity*100, radius*100, length*100, edgeIndex))
                if headPressure <= tailPressure:
                    counter += 1
                    
            elif eqnType == 'flow':
                velocityInIndexList, radiusInList = eqnInfoDict['velocityInIndexList'], eqnInfoDict['radiusInList']
                velocityOutIndexList, radiusOutList = eqnInfoDict['velocityOutIndexList'], eqnInfoDict['radiusOutList']
                neighborsInEdgeIndex, neighborsOutEdgeIndex = itemgetter('neighborsInEdgeIndex', 'neighborsOutEdgeIndex')(eqnInfoDict)
                velocityInList = [np.abs(velocityPressure[velocityIndex]) for velocityIndex in velocityInIndexList]
                velocityOutList = [np.abs(velocityPressure[velocityIndex]) for velocityIndex in velocityOutIndexList]
                flowIn = np.sum([velocity * np.pi * radius**2 for velocity, radius in zip(velocityInList, radiusInList)])
                flowOut = np.sum([velocity * np.pi * radius**2 for velocity, radius in zip(velocityOutList, radiusOutList)])
                error = np.abs(flowIn -flowOut)
                inVel = [np.round(100*vel, 4) for vel in velocityInList]
                inR = [np.round(100*r, 4) for r in radiusInList]
                inFlow = np.round(flowIn*10**6, 4)
                outVel = [np.round(100*vel, 4) for vel in velocityOutList]
                outR = [np.round(100*r, 4) for r in radiusOutList]
                outFlow = np.round(flowOut*10**6, 4)
                errorT = np.round(error*10**6, 4)
                coord = eqnInfoDict['coord']
                flowErrorList.append(error * 10**6 * 20000)
                flowErrorTrueList.append(error * 10**6)
                print('error={} cm^3/s, inVel={} cm/s, inR={} cm, inFlow={} cm^3/s, outVel={} cm/s, outR={} cm, outFlow={} cm^3/s, coord={}'.format(errorT, inVel, inR, inFlow, outVel, outR, outFlow, coord))
            
            elif eqnType == 'boundary':
                velocityIndex, velocityIn = eqnInfoDict['velocityIndex'], eqnInfoDict['velocityIn']
                velocityActual = np.abs(velocityPressure[velocityIndex])
                error = np.abs(velocityActual - velocityIn)
                print('error={}, desired inlet velocity={} cm/s, actual velocity={} cm/s'.format(error, velocityIn*100, velocityActual*100))
        
        totalErrorList = pressureErrorList + flowErrorList
        totalError = norm(totalErrorList)
        print('There are {} flow eqns where headPressure<=tailPressure'.format(counter))
        print('Pressure error: mean+-std={}+-{} mmHg, min={} mmHg, max={} mmHg'.format(np.mean(pressureErrorTrueList), np.std(pressureErrorTrueList), np.amin(pressureErrorTrueList), np.max(pressureErrorTrueList)))
        print('Flow error: mean+-std={}+-{} cm^3/s, min={} cm^3/s, max={} cm^3/s'.format(np.mean(flowErrorTrueList), np.std(flowErrorTrueList), np.amin(flowErrorTrueList), np.max(flowErrorTrueList)))
        print('Combined error (magnified): {}'.format(totalError))
    
    def BFS(self, startNodes, boundaryNodes):
        """
        Start from given node(s), visit other nodes at larger depth in a BFS fashion.

        Parameters
        ----------
        startNodes : list
            A list of nodes to start with.
        boundaryNodes : list
            A list of nodes used as the boundary.
        
        Returns
        -------
        resultDict : dict
            A dictionary containing the indices of visited edges and nodes.
        """
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        visitedNodes, visitedEdges = [], []
        for startNode in startNodes:
            nodesAtCurrentDepth = [startNode]
            while len(nodesAtCurrentDepth) != 0:
                nodesAtNextDepth = []
                for currentNode in nodesAtCurrentDepth:
                    visitedNodes.append(currentNode)
                    newNodes = [node for node in G[currentNode].keys() if nodeInfoDict[currentNode]['depth'] < nodeInfoDict[node]['depth'] and node not in boundaryNodes and node not in visitedNodes]
                    newEdges = [G[currentNode][newNode]['edgeIndex'] for newNode in newNodes]
                    nodesAtNextDepth += newNodes
                    visitedEdges += newEdges
                
                nodesAtCurrentDepth = nodesAtNextDepth
        
        resultDict = {'visitedNodes': visitedNodes, 'visitedEdges': visitedEdges}

        return resultDict

    def calculateVariableBounds(self):
        """
        Calculate the pressure bound for each node and velocity bound for each branch (because pressure at child nodes
        cannot be higher than that of the parent node).
        """
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict

        maxDepth = np.max([info['depth'] for node, info in nodeInfoDict.items()])
        for currentDepth in range(maxDepth-1, 0, -1):
            nodesAtCurrentDepth = [node for node in G.nodes() if nodeInfoDict[node]['depth'] == currentDepth and G.degree(node) != 1]
            for nodeAtCurrentDepth in nodesAtCurrentDepth:
                childNodes = [node for node in G[nodeAtCurrentDepth].keys() if nodeInfoDict[node]['depth'] > currentDepth]
                minPressureAtChildNodes = [nodeInfoDict[node]['simulationData']['minPressure'] if 'argsIndex' in nodeInfoDict[node] else nodeInfoDict[node]['simulationData']['pressure'] for node in childNodes]
                nodeInfoDict[nodeAtCurrentDepth]['simulationData']['minPressure'] = np.amax(minPressureAtChildNodes)
                # print('minPressure for node {} is set'.format(nodeAtCurrentDepth))
        
        # Save #
        self.nodeInfoDict = nodeInfoDict

    def perturbNetwork(self, option=1, extraInfo=None):
        """
        Perturb the network in various ways

        Option=1: randomly choose {numOfEdgesToPerturb} branches and decrease the radius by {reducePercentage}
        Option=2: use the radius from year={perturbedYear}
        Option=3: radius of the edges in {partitionToPerturb} are decreased by {reducePercentage}
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing

        if option == 1:
            if extraInfo is None:
                numOfEdgesToPerturb = 5
                reducePercentage = 30
            else:
                numOfEdgesToPerturb, reducePercentage = itemgetter('numOfEdgesToPerturb', 'reducePercentage')(extraInfo)
            
            edgeIndexToPerturb = np.random.choice(edgeIndexList, numOfEdgesToPerturb)
            for edgeIndex in edgeIndexToPerturb:
                edgeInfoDict[edgeIndex]['meanRadius'] *= (1 - reducePercentage / 100)
        
        elif option == 2:
            perturbedYear, excludedEdgeIndex = itemgetter('perturbedYear', 'excludedEdgeIndex')(extraInfo)
            self.loadNetwork(version=4, year=perturbedYear)
            resultDict = self.loadedNetwork
            GOld, segmentList, partitionInfo, chosenVoxels, segmentInfoDictOld, nodeInfoDictOld, resultADANDict = itemgetter('G', 'segmentList', 'partitionInfo', 'chosenVoxels', 'segmentInfoDict', 'nodeInfoDict', 'resultADANDict')(resultDict)

            for edgeIndex in edgeIndexList:
                if edgeIndex not in excludedEdgeIndex:
                    segmentIndex = edgeInfoDict[edgeIndex]['segmentIndex'] # segmentIndex is the index of the edges in the old files
                    perturbedRadius = segmentInfoDictOld[segmentIndex]['meanRadius']
                    edgeInfoDict[edgeIndex]['meanRadius'] = perturbedRadius
        
        elif option == 3:
            partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                             'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []},
                             'ACA': {'startNodes': [10], 'boundaryNodes': []}}
            
            partitionToPerturb, reducePercentage = itemgetter('partitionToPerturb', 'reducePercentage')(extraInfo)
            for partitionName, info in partitionInfo.items():
                if partitionName in partitionToPerturb:
                    startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
                    resultDict = self.BFS(startNodes, boundaryNodes)
                    visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
                    for edgeIndex in visitedEdges:
                        edgeInfoDict[edgeIndex]['meanRadius'] *= (1 - reducePercentage / 100)

        # Save
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
    
    def perturbTerminatingPressure(self, option=1, extraInfo=None):
        """
        Perturb the terminating pressure in various ways
        Option=1: pressureDecreasePerPartition = {'LMCA': 0.3, 'RMCA': -0.01, 'ACA': 0.05, 'LPCA': -0.02, 'RPCA': 0.02}
        Option=2: No change
        Option=3: All left compartments -30%, no change to all other compartments
        Option=4: pressureDropChangePerPartition = {'LMCA': 0.14, 'RMCA': -0.45, 'ACA': -0.26, 'LPCA': 0.095, 'RPCA': -0.44}
        Option=5: pressureDropChangePerPartition obtained from extraInfo
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]}, 'ACA': {'startNodes': [10], 'boundaryNodes': []},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}}
        if option == 1:
            pressureDecreasePerPartition = {'LMCA': 0.3, 'RMCA': -0.01, 'ACA': 0.05, 'LPCA': -0.02, 'RPCA': 0.02}
        elif option == 2:
            pressureDecreasePerPartition = {'LMCA': 0, 'RMCA': 0, 'ACA': 0, 'LPCA': 0, 'RPCA': 0}
        elif option == 3:
            pressureDecreasePerPartition = {'LMCA': -0.3, 'RMCA': 0, 'ACA': 0, 'LPCA': -0.3, 'RPCA': 0}
        elif option == 4:
            pressureDropChangePerPartition = {'LMCA': 0.14, 'RMCA': -0.45, 'ACA': -0.26, 'LPCA': 0.095, 'RPCA': 0.44}
        elif option == 5:
            pressureDropChangePerPartition = extraInfo['pressureDropChangePerPartition']

        rootPressure = 13560*9.8*0.12 # Pa
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            # terminatingPressuresInThisPartition = []
            for terminatingNode in terminatingNodesInThisPartition:
                if option in [1,2,3]:
                    decreaseAmount = pressureDecreasePerPartition[partitionName]
                    nodeInfoDict[terminatingNode]['simulationData']['pressure'] *= (1-decreaseAmount)
                elif option in [4, 5]:
                    changeAmount = pressureDropChangePerPartition[partitionName]
                    oldPressure = nodeInfoDict[terminatingNode]['simulationData']['pressure']
                    newPressure = rootPressure - (rootPressure - oldPressure) * (1+changeAmount)
                    nodeInfoDict[terminatingNode]['simulationData']['pressure'] = newPressure

            #     terminatingPressuresInThisPartition.append(np.round(nodeInfoDict[terminatingNode]['simulationData']['pressure']/13560/9.8*1000, 2)) # mmHg

            # terminatingPressuresInThisPartition = list(sorted(terminatingPressuresInThisPartition))
            # print('Terminating pressures in {} are {} mmHg'.format(partitionName, terminatingPressuresInThisPartition))
        
        self.nodeInfoDict = nodeInfoDict
    
    def printTerminatingPressurePerPartition(self, partitionInfo=None):
        """
        Print out terminating pressures in each compartment.
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        
        if partitionInfo is None:
            partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]}, 'ACA': {'startNodes': [10], 'boundaryNodes': []},
                             'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}}
        
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingPressuresInThisPartition = []
            for terminatingNode in terminatingNodesInThisPartition:
                terminatingPressuresInThisPartition.append(np.round(nodeInfoDict[terminatingNode]['simulationData']['pressure']/13560/9.8*1000, 2)) # mmHg

            terminatingPressuresInThisPartition = list(sorted(terminatingPressuresInThisPartition))
            print('Terminating pressures in {} are {} mmHg'.format(partitionName, terminatingPressuresInThisPartition))

    def setTerminatingPressure(self, option=1, extraInfo=None):
        """
        Set the terminating pressure based on the terminating pressure vs path length relationship found in ADAN.

        Note: make sure to use the right slope!!!
        Option=1: all partitions use the slope from ADAN dataset
        Option=2: use custom slope for each partition
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        directory = self.directory

        ADANFolder = os.path.abspath(join(directory, '../../../../'))
        with open(join(ADANFolder, 'ADAN-Web/resultADANDict.pkl'), 'rb') as f:
            resultADANDict = pickle.load(f)
            print('resultADANDict.pkl loaded from {}'.format(ADANFolder))

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10], 'pressureIn': 13560*9.8*0.115}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10], 'pressureIn': 13560*9.8*0.115},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': [], 'pressureIn': 13560*9.8*0.115}, 'RPCA': {'startNodes': [7], 'boundaryNodes': [], 'pressureIn': 13560*9.8*0.115},
                         'ACA': {'startNodes': [10], 'boundaryNodes': [], 'pressureIn': 13560*9.8*0.115}}
        # Use the slope and intercept from the ADAN dataset
        if option == 1:
            slopePressurePathLength, interceptPressurePathLength = itemgetter('slopePressurePathLength', 'interceptPressurePathLength')(resultADANDict)
            print('slope={}, intercept={}'.format(slopePressurePathLength, interceptPressurePathLength))
            fitResultPerPartition = {'LMCA': [slopePressurePathLength, interceptPressurePathLength], 'RMCA': [slopePressurePathLength, interceptPressurePathLength],
                                     'LPCA': [slopePressurePathLength, interceptPressurePathLength], 'RPCA': [slopePressurePathLength, interceptPressurePathLength],
                                     'ACA': [slopePressurePathLength, interceptPressurePathLength]}
        # Use the slope and intercept fitted from a ground truth solution
        elif option == 2:
            fitResultPerPartition = extraInfo['fitResultPerPartition']
        elif option == 3:
            pass
        
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes, pressureIn = itemgetter('startNodes', 'boundaryNodes', 'pressureIn')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingPressuresInThisPartition = []
            slopePressurePathLength, interceptPressurePathLength = fitResultPerPartition[partitionName]
            for terminatingNode in terminatingNodesInThisPartition:
                path = nx.shortest_path(G, startNodes[0], terminatingNode)
                pathEdgeIndexList = [G[path[ii]][path[ii + 1]]['edgeIndex'] for ii in range(len(path) - 1)]
                uniquePathEdgeIndexList = np.unique(pathEdgeIndexList)
                assert len(uniquePathEdgeIndexList) != 0
                pathLength = np.sum([edgeInfoDict[edgeIndex]['length'] * spacing for edgeIndex in uniquePathEdgeIndexList]) # meter
                pressure = pressureIn + pathLength * slopePressurePathLength * 0.8
                nodeInfoDict[terminatingNode]['simulationData']['pressure'] = pressure
                terminatingPressuresInThisPartition.append(np.round(pressure/13560/9.8*1000, 2)) # mmHg
            
            terminatingPressuresInThisPartition = list(sorted(terminatingPressuresInThisPartition))
            print('Terminating pressures in {} are {} mmHg'.format(partitionName, terminatingPressuresInThisPartition))
        
        self.nodeInfoDict = nodeInfoDict
    
    def fitTerminatingPressureToPathLength(self, showFittingResult=False, figIndex=1, isLastFigure=False):
        """
        Extract the terminating pressures from the existing fluid solution and fit them to path length per compartment.
        Check the manual correction for LMCA!
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        directory = self.directory

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10], 'color': 'r'}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10], 'color': 'g'},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': [], 'color': 'b'}, 'RPCA': {'startNodes': [7], 'boundaryNodes': [], 'color': 'y'},
                         'ACA': {'startNodes': [10], 'boundaryNodes': [], 'color': 'c'}}
        
        fitResultPerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        terminatingPressurePerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        pathLengthPerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingPressuresInThisPartition = [nodeInfoDict[node]['simulationData']['pressure'] for node in terminatingNodesInThisPartition] # Pascal
            pathLengthInThisPartition = []
            for terminatingNode in terminatingNodesInThisPartition:
                path = nx.shortest_path(G, startNodes[0], terminatingNode)
                pathEdgeIndexList = [G[path[ii]][path[ii + 1]]['edgeIndex'] for ii in range(len(path) - 1)]
                uniquePathEdgeIndexList = np.unique(pathEdgeIndexList)
                assert len(uniquePathEdgeIndexList) != 0
                pathLength = np.sum([edgeInfoDict[edgeIndex]['length'] * spacing for edgeIndex in uniquePathEdgeIndexList]) # meter
                pathLengthInThisPartition.append(pathLength)
            
            # Check this manual correction!
            # if partitionName == 'LMCA':
            #     terminatingPressuresInThisPartition = [val for val in terminatingPressuresInThisPartition if val <= 13560*9.8*0.1]
            #     pathLengthInThisPartition = [val1 for val1, val2 in zip(pathLengthInThisPartition, terminatingPressuresInThisPartition) if val2 <= 13560*9.8*0.1]

            terminatingPressurePerPartition[partitionName] = terminatingPressuresInThisPartition
            pathLengthPerPartition[partitionName] = pathLengthInThisPartition
            # slopeTerminatingPressureVSPathLength, interceptTerminatingPressureVSPathLength = np.polyfit(pathLengthInThisPartition, terminatingPressuresInThisPartition, 1)
            slopePressurePathLength, interceptPressurePathLength, rSqPressurePathLength, pPressurePathLength, stdErrorPressurePathLength = stats.linregress(pathLengthInThisPartition, terminatingPressuresInThisPartition)
            print('{}: slopePressurePathLength={} Pa/m, interceptPressurePathLength={} Pa, rSquared={}, pValue={}'.format(partitionName, slopePressurePathLength, interceptPressurePathLength, rSqPressurePathLength, pPressurePathLength))
            fitResultPerPartition[partitionName] = [slopePressurePathLength, interceptPressurePathLength]
        
        if showFittingResult:
            fig = plt.figure(figIndex, figsize=(15, 3))
            plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
            ax = fig.add_subplot(1,5,1)
            for partitionName, info in partitionInfo.items():
                terminatingPressuresInThisPartition = terminatingPressurePerPartition[partitionName]
                pathLengthInThisPartition = pathLengthPerPartition[partitionName]
                xValues = [val * 1000 for val in pathLengthInThisPartition] # mm
                yValues = [val / 13560 / 9.8 * 1000 for val in terminatingPressuresInThisPartition] # mmHg
                color = info['color']
                ax.scatter(xValues, yValues, c=color, label=partitionName)
            
            ax.set_xlabel('Path length (mm)')
            ax.set_ylabel('Terminating pressure (mmHg)')
            ax.legend(prop={'size': 6})

            if isLastFigure:
                plt.show()
        
        return fitResultPerPartition
    
    def updateNetworkWithSimulationResult(self, velocityPressure):
        """
        Update the flow rate and pressure in `edgeInfoDict` and `nodeInfoDict` with the given `velocityPressure`.
        """
        G = self.G
        edgeIndexList = self.edgeIndexList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing 

        for node in G.nodes():
            if 'argsIndex' in nodeInfoDict[node]:
                argsIndex = nodeInfoDict[node]['argsIndex']
                nodeInfoDict[node]['simulationData']['pressure'] = velocityPressure[argsIndex]
        
        for edgeIndex in edgeIndexList:
            if 'argsIndex' in edgeInfoDict[edgeIndex]:
                argsIndex = edgeInfoDict[edgeIndex]['argsIndex']
                radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
                velocity = velocityPressure[argsIndex] # m/s
                flow = velocity * np.pi * radius**2
                edgeInfoDict[edgeIndex]['simulationData']['velocity'] = velocity
                edgeInfoDict[edgeIndex]['simulationData']['flow'] = flow
        
        # Save
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict
    
    def loadFluidResult(self, loadFileName, return_ResultDict=False):
        """
        Load the saved fluid simulation result.
        For use with GBMTest()
        """
        directory = self.directory
        loadFolderPath = join(directory, 'fluidSimulationResult')
        # loadFileName = 'fluidSimulationResult(referenceYear={}, perturbedYear={}).pkl'.format(resultDict['referenceYear']['year'], resultDict['perturbedYear']['year'])
        with open(join(loadFolderPath, loadFileName), 'rb') as f:
            resultDict = pickle.load(f)
            print('{} loaded from {}'.format(loadFileName, loadFolderPath))
        
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        nodeInfoDictPerturbed, edgeInfoDictPerturbed = itemgetter('nodeInfoDict', 'edgeInfoDict')(resultDict['perturbedYear'])
        numOfNodes = len([node for node in nodeInfoDictPerturbed if 'argsIndex' in nodeInfoDictPerturbed[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDictPerturbed[edgeIndex]])
        velocityPressurePerturbed = [0] * (numOfNodes + numOfEdges)
        for node in G.nodes():
            info = nodeInfoDictPerturbed[node]
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                pressure = info['simulationData']['pressure']
                velocityPressurePerturbed[argsIndex] = pressure
        
        for edgeIndex in edgeIndexList:
            info = edgeInfoDictPerturbed[edgeIndex]
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                velocity = info['simulationData']['velocity']
                velocityPressurePerturbed[argsIndex] = velocity
        
        if return_ResultDict is False:
            return nodeInfoDictPerturbed, edgeInfoDictPerturbed, velocityPressurePerturbed
        else:
            return nodeInfoDictPerturbed, edgeInfoDictPerturbed, velocityPressurePerturbed, resultDict
    
    def loadFluidResult2(self, loadFileName):
        """
        Load the saved fluid simulation result.
        For use with computeNetworkTest()
        """
        directory = self.directory
        loadFolderPath = join(directory, 'fluidSimulationResultRandomNetwork')
        # loadFileName = 'fluidSimulationResult(referenceYear={}, perturbedYear={}).pkl'.format(resultDict['referenceYear']['year'], resultDict['perturbedYear']['year'])
        with open(join(loadFolderPath, loadFileName), 'rb') as f:
            resultDict = pickle.load(f)
            print('{} loaded from {}'.format(loadFileName, loadFolderPath))
        
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        nodeInfoDictPerturbed, edgeInfoDictPerturbed = itemgetter('nodeInfoDict', 'edgeInfoDict')(resultDict['perturbedYear'])
        numOfNodes = len([node for node in nodeInfoDictPerturbed if 'argsIndex' in nodeInfoDictPerturbed[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDictPerturbed[edgeIndex]])
        velocityPressurePerturbed = [0] * (numOfNodes + numOfEdges)
        for node in G.nodes():
            info = nodeInfoDictPerturbed[node]
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                pressure = info['simulationData']['pressure']
                velocityPressurePerturbed[argsIndex] = pressure
        
        for edgeIndex in edgeIndexList:
            info = edgeInfoDictPerturbed[edgeIndex]
            if 'argsIndex' in info:
                argsIndex = info['argsIndex']
                velocity = info['simulationData']['velocity']
                velocityPressurePerturbed[argsIndex] = velocity
        
        return nodeInfoDictPerturbed, edgeInfoDictPerturbed, velocityPressurePerturbed

    def GBMTest(self, saveResult=False):
        """
        Create a GBM network with radius following the BraVa distribution, generate a ground truth solution, then perturb the network 
        in a particular way while keeping the terminating pressures unchanged, then try to solve the network.
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,12]}
        # self.perturbNetwork(option=2, extraInfo=extraInfo)
        # self.setNetwork(option=2)

        success = self.createGroundTruth()
        self.showFlowInfo()
        if not success:
            return
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        
        edgeNameDict = {0: 'LICA', 3: 'LICA', 2: 'RICA', 7: 'RICA', 1: 'VA', 4: 'RPCA\nComm', 8: 'LMCA', 9: 'LM', 11: 'RM', 10: 'RMCA', 5: 'LPCA', 6: 'RPCA', 20: 'ACA'}
        # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
        # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeDepth
        nodeLabelDict = {} # None
        # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
        # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeDepth
        nodeValueList = [0 for node in G.nodes()] # None
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeDepth
        # edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['meanRadius']*spacing*1000, 2) for edge in G.edges()} # edge radius
        edgeLabelDict = {edge: edgeNameDict[G[edge[0]][edge[1]]['edgeIndex']] if G[edge[0]][edge[1]]['edgeIndex'] in edgeNameDict else '' for edge in G.edges()} # edge name
        # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
        # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeDepth
        # edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['meanRadius']*spacing*1000, 2) for edge in G.edges()] # edgeIndex
        edgeValueList = [0 for edge in G.edges()] # None
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': [],
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': [],
                    'figTitle': 'Major branch name'}
        self.plotNetwork(infoDict, figIndex=2, isLastFigure=True)
        return

        # print(G.edges(data=True))
        # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeLevel
        # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeLevel
        # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeLevel
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeLevel
        # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
        # edgeValueList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexList] # meanRadius
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
                    'figTitle': 'GBM Reference'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        resultDict['referenceYear'] = {'year': 'BraVa', 'nodeInfoDict': nodeInfoDict, 'edgeInfoDict': edgeInfoDict, 'G': G}

        ## Solve the system with perturbed network properties
        edgeIndexList = self.edgeIndexList
        # Manually perturb the network #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,12]}
        # self.perturbNetwork(option=2, extraInfo=extraInfo)
        # self.setNetwork(option=2)
        # self.showFlowInfo()
        # computeNetworkDetailExtraInfo = None

        # Load previous optimization result #
        loadFileName = 'fluidSimulationResult3(referenceYear=BraVa, perturbedYear=2013).pkl'
        nodeInfoDictPerturbed, edgeInfoDictPerturbed, velocityPressurePerturbed = self.loadFluidResult(loadFileName)
        velocityPressureInit = velocityPressurePerturbed
        self.nodeInfoDict = nodeInfoDictPerturbed
        self.edgeInfoDict = edgeInfoDictPerturbed
        computeNetworkDetailExtraInfo = {'excludedEdgeIndex': [0,1,2,3,4,5,6,7,10,11,12,13]}

        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
        velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
        
        velocityPressureInit = [float(p) for p in velocityPressureInit]
        # bounds in the form of ((min, min...), (max, max...)) #
        # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
        # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
        # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        # bounds in the form of ((min, max), (min, max)...) #
        boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
        # Improve the lower bound of pressures at each node
        self.calculateVariableBounds()
        for node in G.nodes():
            if 'argsIndex' in nodeInfoDict[node]:
                argsIndex = self.nodeInfoDict[node]['argsIndex']
                minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
                boundsVelocityPressure[argsIndex][0] = minPressure
        boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))

        fluidMethod = 'HW'
        # least square optimization #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # optResult = least_squares(computeNetworkDetail, velocityPressureInit, args=(eqnInfoDictList, fluidMethod), bounds=boundsVelocityPressure, ftol=1e-9, xtol=1e-9)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.cost
        # message = optResult.message

        # differential evolution, bounds in (min, max) pair form #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # errorNorm = 2
        # optResult = differential_evolution(computeNetworkDetail, args=(eqnInfoDictList, fluidMethod, errorNorm), bounds=boundsVelocityPressure, maxiter=2000, polish=True, disp=True)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.fun
        # message = optResult.message

        # basinhopping, bounds in (min, max) pair form #
        self.setupFluidEquations()
        eqnInfoDictList = self.eqnInfoDictList
        errorNorm = 2
        minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
        # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
        optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=100, stepsize=50, interval=5, niter_success=10, disp=True)
        velocityPressure = np.abs(optResult.x)
        cost = optResult.fun
        message = optResult.message

        print('cost={}, message={}'.format(cost, message))
        
        pressures = velocityPressure[numOfEdges:]
        print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
        velocities = velocityPressure[:numOfEdges]
        print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
        
        velocityPressureGroundTruth = self.velocityPressureGroundTruth
        self.velocityPressure = velocityPressure
        self.validateFluidEquations(velocityPressure=velocityPressure)
        print(list(zip(velocityPressureGroundTruth, velocityPressure)))
        
        self.updateNetworkWithSimulationResult(velocityPressure)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM {}'.format(extraInfo['perturbedYear'])}
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)

        resultDict['perturbedYear'] = {'year': 2013, 'nodeInfoDict': nodeInfoDict, 'edgeInfoDict': edgeInfoDict, 'G': G}
        if saveResult:
            directory = self.directory
            saveFolderPath = join(directory, 'fluidSimulationResult')
            saveFileName = 'fluidSimulationResult(referenceYear={}, perturbedYear={}).pkl'.format(resultDict['referenceYear']['year'], resultDict['perturbedYear']['year'])
            with open(join(saveFolderPath, saveFileName), 'wb') as f:
                pickle.dump(resultDict, f, 2)
                print('{} saved to {}'.format(saveFileName, saveFolderPath))
    
    def GBMTest2(self, perturbTerminatingPressureOption=1, saveResult=False):
        """
        Perturb the terminating pressure in a specific way and check if the new system could be solved.
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        success = self.createGroundTruth(option=2)
        self.printTerminatingPressurePerPartition()
        # self.showFlowInfo()
        if not success:
            return
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        
        # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeLevel
        # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeLevel
        # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeLevel
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeLevel
        # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
        # edgeValueList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexList] # meanRadius
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
                    'figTitle': 'GBM Reference'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        resultDict['referenceYear'] = {'year': 'BraVa', 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'G': copy.deepcopy(G)}

        ## Solve the system with perturbed network properties
        edgeIndexList = self.edgeIndexList
        # Manually perturb the network #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]} # perturbTerminatingPressureOption=2
        # perturbTerminatingPressureOption = 1
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        self.perturbTerminatingPressure(option=perturbTerminatingPressureOption)
        self.printTerminatingPressurePerPartition()
        # self.showFlowInfo()
        # computeNetworkDetailExtraInfo = None

        computeNetworkDetailExtraInfo = None

        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
        velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
        
        velocityPressureInit = [float(p) for p in velocityPressureInit]
        # bounds in the form of ((min, min...), (max, max...)) #
        # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
        # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
        # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        # bounds in the form of ((min, max), (min, max)...) #
        boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
        # Improve the lower bound of pressures at each node
        # self.calculateVariableBounds()
        # for node in G.nodes():
        #     if 'argsIndex' in nodeInfoDict[node]:
        #         argsIndex = self.nodeInfoDict[node]['argsIndex']
        #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
        #         boundsVelocityPressure[argsIndex][0] = minPressure
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))

        fluidMethod = 'HW'
        # basinhopping, bounds in (min, max) pair form #
        self.setupFluidEquations()
        eqnInfoDictList = self.eqnInfoDictList
        errorNorm = 2
        minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
        # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
        optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=1000, stepsize=500, interval=5, niter_success=15, disp=True)
        velocityPressure = np.abs(optResult.x)
        cost = optResult.fun
        message = optResult.message

        print('cost={}, message={}'.format(cost, message))
        
        pressures = velocityPressure[numOfEdges:]
        print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
        velocities = velocityPressure[:numOfEdges]
        print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
        
        velocityPressureGroundTruth = self.velocityPressureGroundTruth
        self.velocityPressure = velocityPressure
        self.validateFluidEquations(velocityPressure=velocityPressure)
        print(list(zip(velocityPressureGroundTruth, velocityPressure)))
        
        self.updateNetworkWithSimulationResult(velocityPressure)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
        
        # GBM_BraVa_Reference flow_perturbTerminatingPressureOption=4_GBMTest2
        # GBM_2013_Solved flow_perturbTerminatingPressureOption=4_GBMTest2
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM {}, TPOption={}'.format(extraInfo['perturbedYear'], perturbTerminatingPressureOption)} # TP->terminating pressure
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)

        resultDict['perturbedYear'] = {'year': 2013, 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'G': copy.deepcopy(G), 'velocityPressure': copy.deepcopy(velocityPressure)}
        if saveResult:
            directory = self.directory
            saveFolderPath = join(directory, 'fluidSimulationResult')
            saveFileName = 'fluidSimulationResultGBMTest2(referenceYear={}, perturbedYear={}, perturbTerminatingPressureOption={}).pkl'.format(resultDict['referenceYear']['year'], resultDict['perturbedYear']['year'], perturbTerminatingPressureOption)
            with open(join(saveFolderPath, saveFileName), 'wb') as f:
                pickle.dump(resultDict, f, 2)
                print('{} saved to {}'.format(saveFileName, saveFolderPath))
    
    def GBMTest3(self, perturbTerminatingPressureOption=1, saveResult=False):
        """
        Test the solver

        flowResult_referenceYear(BraVa)_groundTruthOption=1_GBMTest3
        flowResult_solvedYear(BraVa)_groundTruthOption=1_GBMTest3
        flowResult_referenceYear(BraVa)_groundTruthOption=2_GBMTest3
        flowResult_solvedYear(BraVa)_groundTruthOption=2_GBMTest3
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}, 'solvedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        success = self.createGroundTruth(option=2)
        # self.showFlowInfo()
        if not success:
            return
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        
        # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeLevel
        # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeLevel
        # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeLevel
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeLevel
        # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
        # edgeValueList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexList] # meanRadius
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
                    'figTitle': 'GBM Reference'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        resultDict['referenceYear'] = {'year': 'BraVa', 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'G': copy.deepcopy(G)}

        ## Solve the system with perturbed network properties
        edgeIndexList = self.edgeIndexList
        # Manually perturb the network #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        # perturbTerminatingPressureOption = 1
        # self.perturbNetwork(option=2, extraInfo=extraInfo)
        # self.setNetwork(option=2)
        # self.perturbTerminatingPressure(option=perturbTerminatingPressureOption)
        # self.showFlowInfo()
        # computeNetworkDetailExtraInfo = None

        # computeNetworkDetailExtraInfo = {'excludedEdgeIndex': [0,1,2,3,4,5,6,7,10,11,12,13]}
        computeNetworkDetailExtraInfo = None

        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
        velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
        # velocityPressureInit = self.getVelocityPressure() # Get velocityPressure from ground truth solution

        velocityPressureInit = [float(p) for p in velocityPressureInit]
        # bounds in the form of ((min, min...), (max, max...)) #
        # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
        # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
        # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        # bounds in the form of ((min, max), (min, max)...) #
        boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
        # Improve the lower bound of pressures at each node
        # self.calculateVariableBounds()
        # for node in G.nodes():
        #     if 'argsIndex' in nodeInfoDict[node]:
        #         argsIndex = self.nodeInfoDict[node]['argsIndex']
        #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
        #         boundsVelocityPressure[argsIndex][0] = minPressure
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))

        fluidMethod = 'HW'
        # basinhopping, bounds in (min, max) pair form #
        self.setupFluidEquations()
        eqnInfoDictList = self.eqnInfoDictList
        errorNorm = 2
        minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
        # computeNetworkDetail(velocityPressureInit, eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo)
        # self.validateFluidEquations(velocityPressure=velocityPressureInit)
        # print(list(zip(self.velocityPressureGroundTruth, velocityPressureInit)))
        # return
        # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
        optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=1000, stepsize=500, interval=5, niter_success=10, disp=True)
        velocityPressure = np.abs(optResult.x)
        cost = optResult.fun
        message = optResult.message

        print('cost={}, message={}'.format(cost, message))
        
        pressures = velocityPressure[numOfEdges:]
        print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
        velocities = velocityPressure[:numOfEdges]
        print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
        
        velocityPressureGroundTruth = self.velocityPressureGroundTruth
        self.velocityPressure = velocityPressure
        self.validateFluidEquations(velocityPressure=velocityPressure)
        print(list(zip(velocityPressureGroundTruth, velocityPressure)))
        
        self.updateNetworkWithSimulationResult(velocityPressure)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM Solved'} 
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)

        resultDict['solvedYear'] = {'year': 'BraVa', 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'G': copy.deepcopy(G), 'velocityPressure': copy.deepcopy(velocityPressure)}
        if saveResult:
            directory = self.directory
            saveFolderPath = join(directory, 'fluidSimulationResult')
            saveFileName = 'fluidSimulationResultGBMTest3(referenceYear={}, solvedYear={}, groundTruthOption=2).pkl'.format(resultDict['referenceYear']['year'], resultDict['solvedYear']['year'])
            with open(join(saveFolderPath, saveFileName), 'wb') as f:
                pickle.dump(resultDict, f, 2)
                print('{} saved to {}'.format(saveFileName, saveFolderPath))
    
    def GBMTest4(self, perturbNetworkOption=1, saveResult=False):
        """
        Perturb the radius in a specific way, set the TP using path length relationship and solve the network
        Option=1: all LMCA edge radius decrease by 10%
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        extraInfo = {'partitionToPerturb': ['LMCA'], 'reducePercentage': 10}
        self.perturbNetwork(option=perturbNetworkOption, extraInfo=extraInfo)
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        self.setTerminatingPressure(option=1, extraInfo=None)

        computeNetworkDetailExtraInfo = None
        
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
        velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
        
        velocityPressureInit = [float(p) for p in velocityPressureInit]
        # bounds in the form of ((min, min...), (max, max...)) #
        # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
        # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
        # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        # bounds in the form of ((min, max), (min, max)...) #
        boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
        # Improve the lower bound of pressures at each node
        # self.calculateVariableBounds()
        # for node in G.nodes():
        #     if 'argsIndex' in nodeInfoDict[node]:
        #         argsIndex = self.nodeInfoDict[node]['argsIndex']
        #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
        #         boundsVelocityPressure[argsIndex][0] = minPressure
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))

        fluidMethod = 'HW'
        # basinhopping, bounds in (min, max) pair form #
        self.setupFluidEquations()
        eqnInfoDictList = self.eqnInfoDictList
        errorNorm = 2
        minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
        # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
        optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=1000, stepsize=500, interval=5, niter_success=15, disp=True)
        velocityPressure = np.abs(optResult.x)
        cost = optResult.fun
        message = optResult.message

        print('cost={}, message={}'.format(cost, message))
        
        pressures = velocityPressure[numOfEdges:]
        print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
        velocities = velocityPressure[:numOfEdges]
        print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
        
        self.velocityPressure = velocityPressure
        self.validateFluidEquations(velocityPressure=velocityPressure)
        
        self.updateNetworkWithSimulationResult(velocityPressure)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM BraVa, perturbNetworkOption={}'.format(perturbNetworkOption)} 
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)

        resultDict['solvedYear'] = {'year': 'BraVa', 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'G': copy.deepcopy(G), 'velocityPressure': copy.deepcopy(velocityPressure)}
        if saveResult:
            directory = self.directory
            saveFolderPath = join(directory, 'fluidSimulationResult')
            saveFileName = 'fluidSimulationResultGBMTest4(solvedYear=BraVa, perturbNetworkOption={}).pkl'.format(perturbNetworkOption)
            with open(join(saveFolderPath, saveFileName), 'wb') as f:
                pickle.dump(resultDict, f, 2)
                print('{} saved to {}'.format(saveFileName, saveFolderPath))
    
    def GBMTest5(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, change the terminating pressure based on the volume change of the compartment.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)

        Saved Result:
        fluidSimulationResult_GBMTest5_Timestep={}_v1.pkl: everything normal
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2) # just to get nodeIndex and edgeIndex and isBifurcatingNode
        volumePerPartitionGroundTruth = self.getVolumePerPartition()
        print('Ground truth:')
        self.printTerminatingPressurePerPartition()

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        
        # Clear the simulation result #
        # for node in G.nodes():
        #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        # for edgeIndex in edgeIndexList:
        #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        maxTimeStep = numOfTimeSteps
        # fitResultPerPartition = self.fitTerminatingPressureToPathLength(showFittingResult=True, figIndex=2, isLastFigure=True)
        fitResultPerPartition = self.fitTerminatingPressureToPathLength()
        # Start from T1 because T0 is used as a reference case (but still solve T0 just to make a record)
        for currentTimeStep in range(4, 5):
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            volumePerPartition = self.getVolumePerPartition()
            pressureDropChangePerPartition = {}
            for partitionName, volume in volumePerPartition.items():
                volumeGroundTruth = volumePerPartitionGroundTruth[partitionName]
                volumeChange = (volume - volumeGroundTruth) / volumeGroundTruth
                pressureDropChangePerPartition[partitionName] = -volumeChange

            extraInfo = {'pressureDropChangePerPartition': pressureDropChangePerPartition}
            self.perturbTerminatingPressure(option=5, extraInfo=extraInfo)
            self.printTerminatingPressurePerPartition()
            
            computeNetworkDetailExtraInfo = None

            numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
            numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
            pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
            velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
            
            velocityPressureInit = [float(p) for p in velocityPressureInit]
            # bounds in the form of ((min, min...), (max, max...)) #
            # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
            # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
            # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
            # bounds in the form of ((min, max), (min, max)...) #
            boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
            # Improve the lower bound of pressures at each node
            # self.calculateVariableBounds()
            # for node in G.nodes():
            #     if 'argsIndex' in nodeInfoDict[node]:
            #         argsIndex = self.nodeInfoDict[node]['argsIndex']
            #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
            #         boundsVelocityPressure[argsIndex][0] = minPressure
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
    
            fluidMethod = 'HW'
            # basinhopping, bounds in (min, max) pair form #
            self.setupFluidEquations()
            eqnInfoDictList = self.eqnInfoDictList
            errorNorm = 2
            minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
            # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
            optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=1000, stepsize=500, interval=5, niter_success=15, disp=True)
            velocityPressure = np.abs(optResult.x)
            cost = optResult.fun
            message = optResult.message
    
            print('cost={}, message={}'.format(cost, message))
            
            pressures = velocityPressure[numOfEdges:]
            print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
            velocities = velocityPressure[:numOfEdges]
            print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
            
            self.velocityPressure = velocityPressure
            self.validateFluidEquations(velocityPressure=velocityPressure)
            
            if saveResult:
                directory = self.directory
                saveFolderPath = join(directory, 'fluidSimulationResult')
                saveFileName = 'fluidSimulationResult_GBMTest5_Timestep={}_v1.pkl'.format(currentTimeStep)
                resultDict = {'G': copy.deepcopy(self.G), 'nodeInfoDict': copy.deepcopy(self.nodeInfoDict), 'edgeInfoDict': copy.deepcopy(self.edgeInfoDict), 
                              'velocityPressure': copy.deepcopy(velocityPressure)}
                with open(join(saveFolderPath, saveFileName), 'wb') as f:
                    pickle.dump(resultDict, f, 2)
                    print('{} saved to {}'.format(saveFileName, saveFolderPath))
            
            elapsed = timeit.default_timer() - start_time
            print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

            # Clear the simulation result #
            # for node in G.nodes():
            #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            
            # for edgeIndex in edgeIndexList:
            #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
    
    def GBMTest5b(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, TODO !!!

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)

        Saved Result:
        fluidSimulationResultTest6_Timestep={}_v1.pkl: everything normal
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2) # just to get nodeIndex and edgeIndex and isBifurcatingNode
        volumePerPartitionGroundTruth = self.getVolumePerPartition()
        print('Ground truth:')
        self.printTerminatingPressurePerPartition()

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        
        # Clear the simulation result #
        # for node in G.nodes():
        #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        # for edgeIndex in edgeIndexList:
        #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        maxTimeStep = numOfTimeSteps
        # fitResultPerPartition = self.fitTerminatingPressureToPathLength(showFittingResult=True, figIndex=2, isLastFigure=True)
        fitResultPerPartition = self.fitTerminatingPressureToPathLength()
        # Start from T1 because T0 is used as a reference case (but still solve T0 just to make a record)
        for currentTimeStep in range(0, 5):
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            volumePerPartition = self.getVolumePerPartition()
            pressureDropChangePerPartition = {}
            for partitionName, volume in volumePerPartition.items():
                volumeGroundTruth = volumePerPartitionGroundTruth[partitionName]
                volumeChange = (volume - volumeGroundTruth) / volumeGroundTruth
                pressureDropChangePerPartition[partitionName] = -volumeChange

            print(pressureDropChangePerPartition)
    
    def GBMTest6(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Exactly the same as GBMTest5, tweaked the solver setting a little, trying to see if results can be improved.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)

        Saved Result:
        fluidSimulationResult_GBMTest6_Timestep={}_v1.pkl: everything normal
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2) # just to get nodeIndex and edgeIndex and isBifurcatingNode
        volumePerPartitionGroundTruth = self.getVolumePerPartition()
        print('Ground truth:')
        self.printTerminatingPressurePerPartition()

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        
        # Clear the simulation result #
        # for node in G.nodes():
        #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        # for edgeIndex in edgeIndexList:
        #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        maxTimeStep = numOfTimeSteps
        # fitResultPerPartition = self.fitTerminatingPressureToPathLength(showFittingResult=True, figIndex=2, isLastFigure=True)
        fitResultPerPartition = self.fitTerminatingPressureToPathLength()
        # Start from T1 because T0 is used as a reference case (but still solve T0 just to make a record)
        for currentTimeStep in range(0, 5):
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            volumePerPartition = self.getVolumePerPartition()
            pressureDropChangePerPartition = {}
            for partitionName, volume in volumePerPartition.items():
                volumeGroundTruth = volumePerPartitionGroundTruth[partitionName]
                volumeChange = (volume - volumeGroundTruth) / volumeGroundTruth
                pressureDropChangePerPartition[partitionName] = -volumeChange

            extraInfo = {'pressureDropChangePerPartition': pressureDropChangePerPartition}
            self.perturbTerminatingPressure(option=5, extraInfo=extraInfo)
            self.printTerminatingPressurePerPartition()
            
            computeNetworkDetailExtraInfo = None

            numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
            numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
            pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
            velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
            
            velocityPressureInit = [float(p) for p in velocityPressureInit]
            # bounds in the form of ((min, min...), (max, max...)) #
            # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
            # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
            # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
            # bounds in the form of ((min, max), (min, max)...) #
            boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
            # Improve the lower bound of pressures at each node
            # self.calculateVariableBounds()
            # for node in G.nodes():
            #     if 'argsIndex' in nodeInfoDict[node]:
            #         argsIndex = self.nodeInfoDict[node]['argsIndex']
            #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
            #         boundsVelocityPressure[argsIndex][0] = minPressure
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
    
            fluidMethod = 'HW'
            # basinhopping, bounds in (min, max) pair form #
            self.setupFluidEquations()
            eqnInfoDictList = self.eqnInfoDictList
            errorNorm = 2
            minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
            # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
            optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=2000, stepsize=1000, interval=5, niter_success=16, disp=True)
            velocityPressure = np.abs(optResult.x)
            cost = optResult.fun
            message = optResult.message
    
            print('cost={}, message={}'.format(cost, message))
            
            pressures = velocityPressure[numOfEdges:]
            print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
            velocities = velocityPressure[:numOfEdges]
            print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
            
            self.velocityPressure = velocityPressure
            self.validateFluidEquations(velocityPressure=velocityPressure)
            
            if saveResult:
                directory = self.directory
                saveFolderPath = join(directory, 'fluidSimulationResult')
                saveFileName = 'fluidSimulationResult_GBMTest6_Timestep={}_v1.pkl'.format(currentTimeStep)
                resultDict = {'G': copy.deepcopy(self.G), 'nodeInfoDict': copy.deepcopy(self.nodeInfoDict), 'edgeInfoDict': copy.deepcopy(self.edgeInfoDict), 
                              'velocityPressure': copy.deepcopy(velocityPressure)}
                with open(join(saveFolderPath, saveFileName), 'wb') as f:
                    pickle.dump(resultDict, f, 2)
                    print('{} saved to {}'.format(saveFileName, saveFolderPath))
            
            elapsed = timeit.default_timer() - start_time
            print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

    def computeNetworkTest(self, saveResult=False):
        """
        Check whether the solve can correctly solve a system by creating a ground truth model first and comparing the simulation result with it
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        directory = self.directory
        resultDict = {'reference': {}, 'perturbed': {}}
        
        self.generateNetwork(maxDepth=5, allowMerge=False)
        self.setNetwork(option=1)
        success = False
        self.createGroundTruth()
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'Ground truth'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        # self.showTerminatingPressureAndPathLength()
        resultDict['reference'] = {'G': G, 'nodeInfoDict': nodeInfoDict, 'edgeInfoDict': edgeInfoDict}

        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing

        numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
        numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
        pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that in generateNetwork()!
        velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
        # bounds in the form of ((min, min...), (max, max...)) #
        # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
        # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
        # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
        # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        # bounds in the form of ((min, max), (min, max)...) #
        boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
        # Improve the lower bound of pressures at each node
        self.calculateVariableBounds()
        for node in G.nodes():
            if 'argsIndex' in nodeInfoDict[node]:
                argsIndex = nodeInfoDict[node]['argsIndex']
                minPressure = nodeInfoDict[node]['simulationData']['minPressure']
                boundsVelocityPressure[argsIndex][0] = minPressure
        boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
        
        fluidMethod = 'HW'
        ## intensionally perturb the inlet/terminating pressure away from ground truth to see how solver reacts
        # self.nodeInfoDict[0]['simulationData']['pressure'] = 13560*9.8*0.12*(1-np.random.rand()*0.1) # perturb inlet pressure
        ## perturb terminating pressure
        perturbPercent = 0.1
        for node in G.nodes():
            if G.degree(node) == 1:
                self.nodeInfoDict[node]['simulationData']['pressure'] *= (np.random.rand() * perturbPercent + 1 - perturbPercent / 2)
        ## Perturb radius
        # extraInfo = {'numOfEdgesToPerturb': 10, 'reducePercentage': 30}
        # self.perturbNetwork(option=1, extraInfo=extraInfo)
        
        # least square optimization #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # optResult = least_squares(computeNetworkDetail, velocityPressureInit, args=(eqnInfoDictList, fluidMethod), bounds=boundsVelocityPressure, ftol=1e-9, xtol=1e-9)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.cost
        # message = optResult.message

        # minimize (L-BFGS-B), bounds in (min, max) pair form #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # errorNorm = 2
        # options = {'maxiter': 25000, 'maxfun': 25000}
        # optResult = minimize(computeNetworkDetail, velocityPressureInit, args=(eqnInfoDictList, fluidMethod, errorNorm), bounds=boundsVelocityPressure, method='L-BFGS-B', options=options)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.fun
        # message = optResult.message

        # minimize (BFGS), bounds in (min, max) pair form #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # errorNorm = 2
        # options = {'norm': 2, 'maxiter': 30000}
        # optResult = minimize(computeNetworkDetail, velocityPressureInit, args=(eqnInfoDictList, fluidMethod, errorNorm), method='BFGS', options=options)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.fun
        # message = optResult.message

        # basinhopping #
        self.setupFluidEquations()
        eqnInfoDictList = self.eqnInfoDictList
        errorNorm = 0
        minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'options': {'norm': np.inf, 'maxiter': 30000}}
        # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 30000, 'maxfun': 30000}}
        optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=300, T=5, stepsize=5, interval=5, niter_success=20, disp=True)
        velocityPressure = np.abs(optResult.x)
        cost = optResult.fun
        message = optResult.message

        # differential evolution, bounds in (min, max) pair form #
        # self.setupFluidEquations()
        # eqnInfoDictList = self.eqnInfoDictList
        # errorNorm = 2
        # optResult = differential_evolution(computeNetworkDetail, args=(eqnInfoDictList, fluidMethod, errorNorm), bounds=boundsVelocityPressure, maxiter=2000, polish=True, disp=True)
        # velocityPressure = np.abs(optResult.x)
        # cost = optResult.fun
        # message = optResult.message

        # Matlab fsolve #
        # self.setupFluidEquationsMatLab()
        # eqnInfoDictList = self.eqnInfoDictList
        # import matlab.engine, io
        # # eng = matlab.engine.start_matlab()
        # eng = matlab.engine.connect_matlab()
        # eng.addpath('/Users/zhuj10/Dropbox/NIH/Data/Ron Data/1358-Subject18016/fluidSimulationWithCoW')
        # print(matlab.engine.find_matlab())
        # out = io.StringIO()
        # err = io.StringIO()
        # solver = 'fsolve'
        # solver = 'lsqnonlin'
        # # solver = 'Validate'
        # # velocityPressureGroundTruth = self.velocityPressureGroundTruth
        # # velocityPressureInit = [float(p) for p in velocityPressureTrue]
        # velocityPressureInit = [float(p) for p in velocityPressureInit]
        # optResult = eng.performFluidSimulation4ForMatLab(eqnInfoDictList, solver, velocityPressureInit, stdout=out, stderr=err)
        # # optResult = eng.testMatLab1(eqnInfoDictList, solver, velocityPressureInit, stdout=out, stderr=err)
        # # print(optResult)
        # print(out.getvalue())
        # print(err.getvalue())
        # cost = optResult['error']
        # message = optResult['message']
        # velocityPressure = optResult['optParam'][0]
        ##

        print('cost={}, message={}'.format(cost, message))
        
        pressures = velocityPressure[numOfEdges:]
        print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
        velocities = velocityPressure[:numOfEdges]
        print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
        
        velocityPressureGroundTruth = self.velocityPressureGroundTruth
        self.velocityPressure = velocityPressure
        # self.validateFluidEquations(velocityPressure=velocityPressure)
        self.validateFluidEquations(velocityPressure=velocityPressure)
        print(list(zip(velocityPressureGroundTruth, velocityPressure)))

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        self.updateNetworkWithSimulationResult(velocityPressure)
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'Simulated'}
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)

        resultDict['perturbed'] = {'G': G, 'nodeInfoDict': nodeInfoDict, 'edgeInfoDict': edgeInfoDict}
        if saveResult:
            directory = self.directory
            saveFolderPath = join(directory, 'fluidSimulationResultRandomNetwork')
            saveFileName = 'fluidSimulationResult.pkl'
            with open(join(saveFolderPath, saveFileName), 'wb') as f:
                pickle.dump(resultDict, f, 2)
                print('{} saved to {}'.format(saveFileName, saveFolderPath))
    
    def argsBoundTest(self):
        """
        Test the function `calculateVariableBounds`
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        directory = self.directory
        
        # Artificial network
        # self.generateNetwork(maxDepth=5, allowMerge=False)
        # self.setNetwork(option=1)
        # GBM network
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)

        self.createGroundTruth()
        self.calculateVariableBounds()
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        for node in G.nodes():
            if 'argsIndex' not in nodeInfoDict[node]:
                pass
            else:
                if 'minPressure' not in nodeInfoDict[node]['simulationData']:
                    print('Node {} does not have minPressure'.format(node))
        
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'Ground truth'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)

        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) if 'argsIndex' not in nodeInfoDict[node] else np.round(nodeInfoDict[node]['simulationData']['minPressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) if 'argsIndex' not in nodeInfoDict[node] else np.round(nodeInfoDict[node]['simulationData']['minPressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'Ground truth'}
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)
    
    def distributeFlowTest(self):
        """
        Find a way (by optimization) to distribute the flow in the entire network such that the resulting terminating
        pressures match the desired values (does not need to be exactly the same but just to minimize the difference
        between them). Unfinished!
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.showFlowInfo()
        success = self.createGroundTruth()
        if not success:
            return
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict

        # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeLevel
        # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeLevel
        # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeLevel
        # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeLevel
        # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
        # edgeValueList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexList] # meanRadius
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
                    'figTitle': 'GBM Reference'}
        # self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        # resultDict['referenceYear'] = {'year': 'BraVa', 'nodeInfoDict': nodeInfoDict, 'edgeInfoDict': edgeInfoDict}

        ## 
        self.setupEquationsForDistributeFlow()

    def computeNetwork(self):
        pass
    
    def validateNetwork(self):
        pass
    
    def plotNetwork(self, infoDict: dict, figIndex: int=1, isLastFigure: bool=True, hideColorbar: bool=False):
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
        G = self.G
    
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
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=10, edge_color=edgeValueList, edge_cmap=plt.cm.jet, edge_vmin=edge_vmin, edge_vmax=edge_vmax, width=2)
        if len(nodeLabelDict) != 0:
            nx.draw_networkx_labels(G, pos, labels=nodeLabelDict, font_size=8)
        
        if len(edgeLabelDict) != 0:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabelDict, font_size=8)
        
        # node colorbar
        if len(nodeColorbarLabel) != 0 and not hideColorbar:
            # plt.colorbar(nodes, cmap=plt.cm.jet, label=nodeColorbarLabel) 
            ax1 = fig.add_axes([0.8, 0.05, 0.03, 0.9])
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=mpl.cm.jet, norm=norm, orientation='vertical')
            cb1.set_label(nodeColorbarLabel, size=10)
            cb1.ax.tick_params(labelsize=10)

        # edge colorbar
        if len(edgeColorbarLabel) != 0 and not hideColorbar:
            ax2 = fig.add_axes([0.9, 0.05, 0.03, 0.9])
            norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=mpl.cm.jet, norm=norm, orientation='vertical')
            cb2.set_label(edgeColorbarLabel, size=10)
            cb2.ax.tick_params(labelsize=10)
        
        if isLastFigure:
            plt.show()
    
    def getNetwork(self):
        return self.G
    
    def compareNetworkPropertyTest(self):
        """
        Compare the edge properties before and after perturbing the network.
        GBM_Radius ratio vs Graph level_Compartment(5)_Single row
        GBM_Radius ratio vs Graph level_Graph plot
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name

        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.nodeInfoDictBefore = copy.deepcopy(self.nodeInfoDict)
        self.edgeInfoDictBefore = copy.deepcopy(self.edgeInfoDict)

        ## Solve the system with perturbed network properties
        edgeIndexList = self.edgeIndexList
        # Manually perturb the network #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        self.nodeInfoDictAfter = copy.deepcopy(self.nodeInfoDict)
        self.edgeInfoDictAfter = copy.deepcopy(self.edgeInfoDict)

        edgeIndexList = sorted(list(self.edgeInfoDict.keys()))
        spacing = self.spacing
        print('Edge difference before and after:')
        for edgeIndex in edgeIndexList:
            radius, length, c, k = itemgetter('meanRadius', 'length', 'c', 'k')(self.edgeInfoDictBefore[edgeIndex])
            radiusBefore = np.round(radius * spacing * 1000, 3) # mm
            lengthBefore = np.round(length * spacing * 100, 3) # cm
            cBefore, kBefore = np.round(c, 3), np.round(k, 3)
            radius, length, c, k = itemgetter('meanRadius', 'length', 'c', 'k')(self.edgeInfoDictAfter[edgeIndex])
            radiusAfter = np.round(radius * spacing * 1000, 3) # mm
            lengthAfter = np.round(length * spacing * 100, 3) # cm
            cAfter, kAfter = np.round(c, 3), np.round(k, 3)
            print('edgeIndex={}, radius={}/{} mm, length={}/{} cm, c={}/{}, k={}/{}'.format(edgeIndex, radiusBefore, radiusAfter, lengthBefore, lengthAfter, cBefore, cAfter, kBefore, kAfter))
        
        G = self.G
        for edge in G.edges():
            edgeIndex = G[edge[0]][edge[1]]['edgeIndex']
            radiusRatio = np.round(self.edgeInfoDictAfter[edgeIndex]['meanRadius'] / self.edgeInfoDictBefore[edgeIndex]['meanRadius'], 2)
            self.edgeInfoDictAfter[edgeIndex]['radiusRatio'] = radiusRatio
            self.edgeInfoDictBefore[edgeIndex]['radiusRatio'] = radiusRatio
            self.edgeInfoDict[edgeIndex]['radiusRatio'] = radiusRatio
        
        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: nodeInfoDict[node]['nodeIndex'] for node in G.nodes()}
        nodeValueList = [nodeInfoDict[node]['nodeIndex'] for node in G.nodes()] 
        edgeLabelDict = {edge: self.edgeInfoDictAfter[G[edge[0]][edge[1]]['edgeIndex']]['radiusRatio'] for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [self.edgeInfoDictAfter[G[edge[0]][edge[1]]['edgeIndex']]['radiusRatio'] for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'Ground truth'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        
        partitionInfo = {'LMCA': {'startNodes': [5], 'boundaryNodes': [13]}, 'RMCA': {'startNodes': [6], 'boundaryNodes': [13]},
                         'LPCA': {'startNodes': [4], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [13], 'boundaryNodes': []}}
        
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}

        # fig = plt.figure(2, figsize=(15, 8))
        # plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
        fig = plt.figure(11, figsize=(15, 3))
        plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        
        subplotIndex = 1
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            print('{}:\nvisitedNodes={}\nvisitedEdges={}'.format(partitionName, visitedNodes, visitedEdges))

            ax = fig.add_subplot(1,5,subplotIndex)
            dictUsed = edgeInfoDict
            attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
            attribute1List = [edgeInfoDict[edgeIndex]['depth'] for edgeIndex in visitedEdges]
            # attribute2List = [edgeInfoDict[edgeIndex]['meanRadius']*spacing*1000 for edgeIndex in visitedEdges]
            attribute2List = [edgeInfoDict[edgeIndex]['radiusRatio'] for edgeIndex in visitedEdges]
            # attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] in partitionNames]
            # attribute2List = [info[attribute2]*spacing*1000 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] in partitionNames] # mm
            # ax.plot(attribute1List, attribute2List, 'bo')
            positions = np.sort(np.unique(attribute1List))
            values = []
            attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
            for segmentLevel in positions:
                locs = np.nonzero(attribute1Array == segmentLevel)[0]
                values.append((attribute2Array[locs]).tolist())
        
            mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Radius (mm)')
            ax.set_xlabel('Graph level')
            ax.set_ylabel('Radius ratio')
            ax.set_title(partitionName)

            subplotIndex += 1
        
        plt.show()
    
    def updateEdgeRadius(self, edgeRadiusList):
        """
        Update the edge radius with the supplied list.

        The i-th element in edgeRadiusList is the radius (in voxel) of the i-th edge.

        Parameters
        ----------
        edgeRadiusList : list
            A list of new edge radius.
        """
        edgeInfoDict = self.edgeInfoDict
        for edgeIndex, radius in enumerate(edgeRadiusList):
            edgeInfoDict[edgeIndex]['meanRadius'] = radius
        
        self.edgeInfoDict = edgeInfoDict
        self.setNetwork(option=2)
    
    def applyFlowToNetwork(self, edgeFlowList):
        """
        Apply the flow from edgeFlowList to the corresponding edges and recalculates all the pressures.

        The i-th element in edgeFlowList is the flow (in m^3/s) of the i-th edge.

        Parameters
        ----------
        edgeFlowList : list
            A list of flow rates to be applied to each edges.
        """
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        edgeList = self.edgeList
        spacing = self.spacing
        for edgeIndex, flow in enumerate(edgeFlowList):
            edgeInfoDict[edgeIndex]['simulationData']['flow'] = flow
            radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
            velocity = flow / (np.pi * radius**2) # m/s
            edgeInfoDict[edgeIndex]['simulationData']['velocity'] = velocity
        
        edgeDepthArray = np.array([edgeInfoDict[edgeIndex]['depth'] for edgeIndex in edgeIndexList])
        edgeIndexListSorted = np.array(edgeIndexList)[edgeDepthArray.argsort()].tolist()
        for edgeIndex in edgeIndexListSorted:
            edge = edgeList[edgeIndex]
            edgeHead, edgeTail = edge
            if nodeInfoDict[edgeHead]['depth'] > nodeInfoDict[edgeTail]['depth']:
                edgeHead, edgeTail = edgeTail, edgeHead
            
            pressureHead = nodeInfoDict[edgeHead]['simulationData']['pressure']
            radius = edgeInfoDict[edgeIndex]['meanRadius'] * spacing # meter
            length = edgeInfoDict[edgeIndex]['length'] * spacing # meter
            c, k = itemgetter('c', 'k')(edgeInfoDict[edgeIndex])
            flow = edgeFlowList[edgeIndex]
            deltaPressure = 10.67 * flow**k * length / c**k / (2*radius)**4.8704
            if pressureHead is None:
                print('Error! EdgeIndex={} has pressure = None'.format(edgeIndex))
            pressureTail = pressureHead - deltaPressure
            nodeInfoDict[edgeTail]['simulationData']['pressure'] = pressureTail
        
        self.nodeInfoDict = nodeInfoDict
        self.edgeInfoDict = edgeInfoDict

    def showVolumePerPartition(self, numOfTimeSteps=4, interpolationOption=1, figIndex=1, isLastFigure=True):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, and check how the volume of each partition changes among different time steps.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        edgeIndexList = self.edgeIndexList

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        volumeTimeStepListPerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': [], 'Left': [], 'Right': []}
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        for currentTimeStep in range(0, numOfTimeSteps):
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            volumePerPartition = self.getVolumePerPartition()
            for partitionName, volume in volumePerPartition.items():
                volumeTimeStepListPerPartition[partitionName].append(volume)
        
        volumeTimeStepListPerPartition['Left'] = (np.array(volumeTimeStepListPerPartition['LMCA']) + np.array(volumeTimeStepListPerPartition['LPCA'])).tolist()
        volumeTimeStepListPerPartition['Right'] = (np.array(volumeTimeStepListPerPartition['RMCA']) + np.array(volumeTimeStepListPerPartition['RPCA'])).tolist()
        print('volumeTimeStepListPerPartition={}'.format(volumeTimeStepListPerPartition))
        
        fig = plt.figure(figIndex, figsize=(7, 3))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        ax = fig.add_subplot(1,1,1)
        for partitionName, volumeList in volumeTimeStepListPerPartition.items():
            xValues = list(range(numOfTimeSteps))
            yValues = volumeList
            ax.plot(xValues, yValues, 'o-', label=partitionName)
            ax.set_xlabel('Time step')
            ax.set_xticks(xValues)
            ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
            ax.set_ylabel(r'Volume ($\mathrm{mm}^3/s$)')
        
        # ax.legend()
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=7, prop={'size': 8})
        if isLastFigure:
            plt.show()
    
    def test1(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, split the flow according to the cross-sectional area (option 2 in
        createGroundTruth()) and see how the terminating pressures change.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        aa = [self.edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in self.edgeIndexList]
        # print(aa)
        # success = self.createGroundTruth(option=2)

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        cTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        allNodes = list(range(np.max(list(self.nodeInfoDict.keys())) + 1))
        terminatingNodes = [node for node in G.nodes() if G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        nodePressuresTimeStepArray = np.zeros((len(allNodes), numOfTimeSteps))
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        for currentTimeStep in range(0, numOfTimeSteps):
            # print(currentTimeStep)
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            # print(radiusList)
            self.updateEdgeRadius(radiusList)
            success = self.createGroundTruth(option=2)
            if not success:
                print('Time step={} failed'.format(currentTimeStep))
            
            terminatingPressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in terminatingNodes]
            terminatingPressuresTimeStepArray[:, currentTimeStep] = terminatingPressures
            nodePressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in allNodes]
            nodePressuresTimeStepArray[:, currentTimeStep] = nodePressures
            cValues = [self.edgeInfoDict[edgeIndex]['c'] for edgeIndex in edgeIndexList]
            cTimeStepArray[edgeIndexList, currentTimeStep] = cValues

            # G = self.G
            # nodeInfoDict = self.nodeInfoDict
            # edgeInfoDict = self.edgeInfoDict
            # nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
            # nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
            # edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
            # edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
            # infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
            #             'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
            #             'figTitle': 'Ground truth'}
            # self.plotNetwork(infoDict, figIndex=1, isLastFigure=True)
            
            # Clear the simulation result #
            for node in G.nodes():
                self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            
            for edgeIndex in edgeIndexList:
                self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        print(terminatingPressuresTimeStepArray)
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]}, 'ACA': {'startNodes': [10], 'boundaryNodes': []},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}}
        
        fig = plt.figure(1, figsize=(15, 8))
        plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
        subplotIndex = 1
        colorList = ['r','g','b']

        # terminatingNodes = {'LMCA': [], 'RMCA': [], 'ACA': [], 'LPCA': [], 'RPCA': []}
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            # terminatingNodes[partitionName] = terminatingNodesInThisPartition
            ax = fig.add_subplot(2,4,subplotIndex, projection='3d')
            for ii, node in enumerate(terminatingNodesInThisPartition):
                rowNum = terminatingNodes.index(node)
                pressures = terminatingPressuresTimeStepArray[rowNum, :]
                xValues = [node] * numOfTimeSteps
                yValues = list(range(numOfTimeSteps))
                zValues = list(pressures)
                ax.plot(xValues, yValues, zValues, 'bo-')
                ax.set_xlabel('Node index')
                ax.set_ylabel('Time step')
                ax.set_zlabel('Terminating pressure (mmHg)')
                ax.set_title(partitionName)
            
            subplotIndex += 1
        
        edgeRadiusTimeStepArray = np.array(edgeRadiusTimeStepList)
        spacing = self.spacing
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            if partitionName != 'LPCA' and partitionName != 'LMCA' and partitionName != 'ACA':
                continue
            print('{}:'.format(partitionName))
            for terminatingNode in terminatingNodesInThisPartition:
                path = nx.shortest_path(G, startNodes[0], terminatingNode)
                edgeIndexAlongPath = [G[path[ii]][path[ii+1]]['edgeIndex'] for ii in range(len(path) - 1)]
                for currentTimeStep in range(numOfTimeSteps):
                    pressuresAlongPath = np.round(nodePressuresTimeStepArray[path, currentTimeStep], 2) # mmHg
                    edgeRadiusAlongPath = np.round(edgeRadiusTimeStepArray[edgeIndexAlongPath, currentTimeStep]*spacing*1000, 2) # mm
                    cAlongPath = np.round(cTimeStepArray[edgeIndexAlongPath, currentTimeStep], 3)
                    print('Terminating node {} (time step={}): pressures along path are {} mmHg, radius along path are {} mm, c={}'.format(terminatingNode, currentTimeStep, pressuresAlongPath, edgeRadiusAlongPath, cAlongPath))

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        plt.show()
    
    def test2(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, apply the same flow/different flow and check the differences in terminating pressures
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        edgeIndexList = self.edgeIndexList
        edgeFlowList = [0] * len(edgeIndexList)
        for edgeIndex in edgeIndexList:
            edgeFlowList[edgeIndex] = self.edgeInfoDict[edgeIndex]['simulationData']['flow']

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'BraVa'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        self.plotTerminatingPressures(figIndex=2, isLastFigure=False)

        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        self.applyFlowToNetwork(edgeFlowList)
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM 2013'}
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=False)
        self.plotTerminatingPressures(figIndex=4, isLastFigure=True)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
    
    def test3(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, use the same flow pattern as the BraVa for other time points and see how the
        terminating pressures change.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        edgeIndexList = self.edgeIndexList
        edgeFlowList = [0] * len(edgeIndexList)
        for edgeIndex in edgeIndexList:
            edgeFlowList[edgeIndex] = self.edgeInfoDict[edgeIndex]['simulationData']['flow']

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        cTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        flowTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        allNodes = list(range(np.max(list(self.nodeInfoDict.keys())) + 1))
        terminatingNodes = [node for node in G.nodes() if G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        nodePressuresTimeStepArray = np.zeros((len(allNodes), numOfTimeSteps))
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        for currentTimeStep in range(0, numOfTimeSteps):
            # print(currentTimeStep)
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            # print(radiusList)
            self.updateEdgeRadius(radiusList)
            self.applyFlowToNetwork(edgeFlowList)
            
            terminatingPressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in terminatingNodes]
            terminatingPressuresTimeStepArray[:, currentTimeStep] = terminatingPressures
            nodePressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in allNodes]
            nodePressuresTimeStepArray[:, currentTimeStep] = nodePressures
            cValues = [self.edgeInfoDict[edgeIndex]['c'] for edgeIndex in edgeIndexList]
            cTimeStepArray[edgeIndexList, currentTimeStep] = cValues
            flowValues = [self.edgeInfoDict[edgeIndex]['simulationData']['flow'] for edgeIndex in edgeIndexList] # m^3/s
            flowTimeStepArray[edgeIndexList, currentTimeStep] = flowValues

            # G = self.G
            # nodeInfoDict = self.nodeInfoDict
            # edgeInfoDict = self.edgeInfoDict
            # nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
            # nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
            # edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
            # edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
            # infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
            #             'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
            #             'figTitle': 'Ground truth'}
            # self.plotNetwork(infoDict, figIndex=1, isLastFigure=True)
            
            # Clear the simulation result #
            # for node in G.nodes():
            #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            
            # for edgeIndex in edgeIndexList:
            #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        # print(terminatingPressuresTimeStepArray)
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=11, isLastFigure=False)
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=2, figIndex=21, isLastFigure=False)
        # self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=3, figIndex=31, isLastFigure=False)
        self.plotTerminatingPressureVSPathLength(terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=31, isLastFigure=False)
        self.plotFlow(flowTimeStepArray, option=1, figIndex=41, isLastFigure=False)
        self.plotRootPressuresCompartment(nodePressuresTimeStepArray, option=1, figIndex=51, isLastFigure=False)
        self.plotFlowProportion(flowTimeStepArray, figIndex=61, isLastFigure=True) 
        # Flow proportion_Same flow_All CoW branches fixed_test3
        # Flow proportion_Same flow_LICA RICA VA fixed_test3

        # partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
        #                  'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        # G = self.G
        # nodeInfoDict = self.nodeInfoDict
        # edgeInfoDict = self.edgeInfoDict
        # edgeRadiusTimeStepArray = np.array(edgeRadiusTimeStepList)
        # spacing = self.spacing
        # for partitionName, info in partitionInfo.items():
        #     startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
        #     resultDict = self.BFS(startNodes, boundaryNodes)
        #     visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
        #     terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
        #     if partitionName != 'RMCA' and partitionName != 'LMCA' and partitionName != 'LPCA':
        #         continue
        #     print('{}:'.format(partitionName))
        #     for terminatingNode in terminatingNodesInThisPartition:
        #         path = nx.shortest_path(G, startNodes[0], terminatingNode)
        #         edgeIndexAlongPath = [G[path[ii]][path[ii+1]]['edgeIndex'] for ii in range(len(path) - 1)]
        #         for currentTimeStep in range(numOfTimeSteps):
        #             pressuresAlongPath = np.round(nodePressuresTimeStepArray[path, currentTimeStep], 2) # mmHg
        #             edgeRadiusAlongPath = np.round(edgeRadiusTimeStepArray[edgeIndexAlongPath, currentTimeStep]*spacing*1000, 2) # mm
        #             cAlongPath = np.round(cTimeStepArray[edgeIndexAlongPath, currentTimeStep], 3)
        #             print('Terminating node {} (time step={}): pressures along path are {} mmHg, radius along path are {} mm, c={}'.format(terminatingNode, currentTimeStep, pressuresAlongPath, edgeRadiusAlongPath, cAlongPath))

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        # plt.show()
    
    def test4(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, and check the differences in terminating pressures
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        edgeIndexList = self.edgeIndexList
        edgeFlowList = [0] * len(edgeIndexList)
        for edgeIndex in edgeIndexList:
            edgeFlowList[edgeIndex] = self.edgeInfoDict[edgeIndex]['simulationData']['flow']
        edgeRadiusTimeStepArray = np.zeros((len(edgeIndexList), 2))
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        for edgeIndex in edgeIndexList:
            edgeRadiusTimeStepArray[edgeIndex, 0] = self.edgeInfoDict[edgeIndex]['meanRadius'] * spacing * 1000

        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'BraVa'}
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        # self.plotTerminatingPressures(figIndex=2, isLastFigure=False)

        # Clear the simulation result #
        for node in G.nodes():
            self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        for edgeIndex in edgeIndexList:
            self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset

        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            edgeRadiusTimeStepArray[edgeIndex, 1] = self.edgeInfoDict[edgeIndex]['meanRadius'] * spacing * 1000
        for ii, row in enumerate(edgeRadiusTimeStepArray):
            radiusBefore, radiusAfter = np.round(row, 3).tolist()
            print('Edge {}: radius before/after = {}/{} mm'.format(ii, radiusBefore, radiusAfter))
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
                    'figTitle': 'GBM 2013'}
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=True)
        # self.plotTerminatingPressures(figIndex=4, isLastFigure=True)

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
    
    def test5(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, and see how the terminating pressures change.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)
        edgeIndexList = self.edgeIndexList
        edgeFlowList = [0] * len(edgeIndexList)
        for edgeIndex in edgeIndexList:
            edgeFlowList[edgeIndex] = self.edgeInfoDict[edgeIndex]['simulationData']['flow']

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        cTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        allNodes = list(range(np.max(list(self.nodeInfoDict.keys())) + 1))
        terminatingNodes = [node for node in G.nodes() if G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        nodePressuresTimeStepArray = np.zeros((len(allNodes), numOfTimeSteps))
        flowTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        
        # Clear the simulation result #
        for node in G.nodes():
            self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        for edgeIndex in edgeIndexList:
            self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        for currentTimeStep in range(0, numOfTimeSteps):
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            self.createGroundTruth(option=2)
            
            terminatingPressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in terminatingNodes]
            terminatingPressuresTimeStepArray[:, currentTimeStep] = terminatingPressures
            nodePressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in allNodes]
            nodePressuresTimeStepArray[:, currentTimeStep] = nodePressures
            cValues = [self.edgeInfoDict[edgeIndex]['c'] for edgeIndex in edgeIndexList]
            cTimeStepArray[edgeIndexList, currentTimeStep] = cValues
            flowValues = [self.edgeInfoDict[edgeIndex]['simulationData']['flow'] for edgeIndex in edgeIndexList] # m^3/s
            flowTimeStepArray[edgeIndexList, currentTimeStep] = flowValues

            # Clear the simulation result #
            for node in G.nodes():
                self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            
            for edgeIndex in edgeIndexList:
                self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset

        # print(terminatingPressuresTimeStepArray)
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=11, isLastFigure=False)
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=2, figIndex=21, isLastFigure=False)
        # self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=3, figIndex=31, isLastFigure=False)
        self.plotFlow(flowTimeStepArray, option=1, figIndex=41, isLastFigure=False)
        self.plotRootPressuresCompartment(nodePressuresTimeStepArray, option=1, figIndex=51, isLastFigure=False)
        self.plotFlowProportion(flowTimeStepArray, figIndex=61, isLastFigure=True) 
        # Flow proportion_Split flow with radius_All CoW branches fixed_test3
        # Flow proportion_Split flow with radius_LICA RICA VA fixed_test3
        
        print(edgeRadiusTimeStepList[8:12])
        print(flowTimeStepArray[[4,8,9,10,11],:])
        
        
        # edgeRadiusTimeStepArray = np.array(edgeRadiusTimeStepList)
        # spacing = self.spacing
        # for partitionName, info in partitionInfo.items():
        #     startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
        #     resultDict = self.BFS(startNodes, boundaryNodes)
        #     visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
        #     terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
        #     if partitionName != 'LPCA' and partitionName != 'LMCA' and partitionName != 'ACA':
        #         continue
        #     print('{}:'.format(partitionName))
        #     for terminatingNode in terminatingNodesInThisPartition:
        #         path = nx.shortest_path(G, startNodes[0], terminatingNode)
        #         edgeIndexAlongPath = [G[path[ii]][path[ii+1]]['edgeIndex'] for ii in range(len(path) - 1)]
        #         for currentTimeStep in range(numOfTimeSteps):
        #             pressuresAlongPath = np.round(nodePressuresTimeStepArray[path, currentTimeStep], 2) # mmHg
        #             edgeRadiusAlongPath = np.round(edgeRadiusTimeStepArray[edgeIndexAlongPath, currentTimeStep]*spacing*1000, 2) # mm
        #             cAlongPath = np.round(cTimeStepArray[edgeIndexAlongPath, currentTimeStep], 3)
        #             print('Terminating node {} (time step={}): pressures along path are {} mmHg, radius along path are {} mm, c={}'.format(terminatingNode, currentTimeStep, pressuresAlongPath, edgeRadiusAlongPath, cAlongPath))

        # elapsed = timeit.default_timer() - start_time
        # print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        plt.show()
    
    def test6(self, numOfTimeSteps=4, interpolationOption=1, saveResult=False):
        """
        Using the GBM network and the radius info from BraVa and 2013, interpolate the radius (in different ways) for
        the time point in between, set the terminating pressures using the terminating pressure vs path length
        relationship found with ADAN dataset, and solve the network. The purpose is to see when the network fails to
        produce a solution (i.e., optimization error becomes too large to be acceptable), and for those time steps that
        do not have a solution, perturb the terminating pressures with minimum effort (another optimization) so that
        there exists a solution.

        numOfTimeSteps has to be >= 2 (including the two end time steps)
        interpolationOption=1 interpolates the radius linearly, interpolationOption=2 uses a logistic curve (bent
        upwards), interpolationOption=3 uses a logistic curve (bent downwards)

        Note: check what slope is being used in setTerminatingPressures()!

        Saved Result:
        fluidSimulationResultTest6_Timestep={}_v1.pkl: everything normal
        fluidSimulationResultTest6_Timestep={}_v2.pkl: slope of terminating pressure vs path length reduced by 30%
        fluidSimulationResultTest6_Timestep={}_v3.pkl: slope of terminating pressure vs path length reduced by 40%
        fluidSimulationResultTest6_Timestep={}_v4.pkl: slope of terminating pressure vs path length reduced by 20%
        fluidSimulationResultTest6_Timestep={}_v5.pkl: slope of terminating pressure vs path length comes from fitting the ground truth solution (option=2)
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2) # just to get nodeIndex and edgeIndex and isBifurcatingNode
        # G = self.G
        # nodeInfoDict = self.nodeInfoDict
        # edgeInfoDict = self.edgeInfoDict
        # nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        # nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        # edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        # edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        # infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node',
        #             'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge',
        #             'figTitle': 'Ground truth'}
        # self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)

        edgeIndexList = self.edgeIndexList
        G = self.G
        edgeRadiusTimeStepList = np.zeros((len(edgeIndexList), numOfTimeSteps)).tolist()
        # cTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        # allNodes = list(range(np.max(list(self.nodeInfoDict.keys())) + 1))
        # terminatingNodes = [node for node in G.nodes() if G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        # terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        # nodePressuresTimeStepArray = np.zeros((len(allNodes), numOfTimeSteps))
        # flowTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][0] = radius

        # Change the radius #
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,9,11,5,6]}
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)
        # success = self.createGroundTruth(option=2)
        for edgeIndex in edgeIndexList:
            radius = self.edgeInfoDict[edgeIndex]['meanRadius']
            edgeRadiusTimeStepList[edgeIndex][-1] = radius
        
        # Interpolate the radius for other time steps #
        if interpolationOption == 1:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) / (numOfTimeSteps - 1) * ii + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        elif interpolationOption == 2:
            for edgeIndex in edgeIndexList:
                radiusHead, radiusTail = edgeRadiusTimeStepList[edgeIndex][0], edgeRadiusTimeStepList[edgeIndex][-1]
                for ii in range(1, numOfTimeSteps-1):
                    radius = (radiusTail - radiusHead) * np.tanh(ii / (numOfTimeSteps-1) * 2) + radiusHead
                    edgeRadiusTimeStepList[edgeIndex][ii] = radius
        
        # print(edgeRadiusTimeStepList)
        
        # Clear the simulation result #
        # for node in G.nodes():
        #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
        
        # for edgeIndex in edgeIndexList:
        #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset
        
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        maxTimeStep = 1 # numOfTimeSteps
        # fitResultPerPartition = self.fitTerminatingPressureToPathLength(showFittingResult=True, figIndex=2, isLastFigure=True)
        fitResultPerPartition = self.fitTerminatingPressureToPathLength()
        for currentTimeStep in range(0, maxTimeStep):
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            radiusList = [edgeRadiusTimeStepList[edgeIndex][currentTimeStep] for edgeIndex in edgeIndexList]
            self.updateEdgeRadius(radiusList)
            # self.createGroundTruth(option=2)
            extraInfo = None
            extraInfo = {'fitResultPerPartition': fitResultPerPartition}
            self.setTerminatingPressure(option=2, extraInfo=extraInfo)
            
            computeNetworkDetailExtraInfo = None

            numOfNodes = len([node for node in nodeInfoDict if 'argsIndex' in nodeInfoDict[node]])
            numOfEdges = len([edgeIndex for edgeIndex in edgeIndexList if 'argsIndex' in edgeInfoDict[edgeIndex]])
            pressureIn = 13560 * 9.8 * 0.12 # Pascal # check if this number is consistent with that was used in the reference case!
            velocityPressureInit = np.hstack((np.full((numOfEdges,), 0.4), np.linspace(pressureIn*0.8, pressureIn*0.5, numOfNodes)))
            
            velocityPressureInit = [float(p) for p in velocityPressureInit]
            # bounds in the form of ((min, min...), (max, max...)) #
            # boundsVelocityPressure = [[], []] # first sublist contains lower bound and the second sublist contains upper bound
            # boundsVelocityPressure[0] = [0] * numOfEdges + [13560*9.8*0.00] * numOfNodes # min velocity = 0 m/s, min pressure = 0 mmHg
            # boundsVelocityPressure[1] = [5] * numOfEdges + [13560*9.8*0.12] * numOfNodes # max velocity = 5 m/s, max pressure = 120 mmHg
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
            # bounds in the form of ((min, max), (min, max)...) #
            boundsVelocityPressure = [[0, 5]] * numOfEdges + [[13560*9.8*0.00, 13560*9.8*0.12]] * numOfNodes
            # Improve the lower bound of pressures at each node
            # self.calculateVariableBounds()
            # for node in G.nodes():
            #     if 'argsIndex' in nodeInfoDict[node]:
            #         argsIndex = self.nodeInfoDict[node]['argsIndex']
            #         minPressure = self.nodeInfoDict[node]['simulationData']['minPressure']
            #         boundsVelocityPressure[argsIndex][0] = minPressure
            # boundsVelocityPressure = tuple(map(tuple, boundsVelocityPressure))
    
            fluidMethod = 'HW'
            # basinhopping, bounds in (min, max) pair form #
            self.setupFluidEquations()
            eqnInfoDictList = self.eqnInfoDictList
            errorNorm = 2
            minimizer_kwargs = {'method': 'BFGS', 'args': (eqnInfoDictList, fluidMethod, errorNorm, computeNetworkDetailExtraInfo), 'options': {'norm': np.inf, 'maxiter': 40000}}
            # minimizer_kwargs = {'method': 'L-BFGS-B', 'args': (eqnInfoDictList, fluidMethod, errorNorm), 'bounds': boundsVelocityPressure, 'options': {'maxiter': 40000, 'maxfun': 40000}}
            optResult = basinhopping(computeNetworkDetail, velocityPressureInit, minimizer_kwargs=minimizer_kwargs, niter=100, T=1000, stepsize=500, interval=5, niter_success=15, disp=True)
            velocityPressure = np.abs(optResult.x)
            cost = optResult.fun
            message = optResult.message
    
            print('cost={}, message={}'.format(cost, message))
            
            pressures = velocityPressure[numOfEdges:]
            print('Minimum pressure is {} mmHg and maximum pressure is {} mmHg'.format((np.amin(pressures))/13560/9.8*1000, (np.amax(pressures))/13560/9.8*1000))
            velocities = velocityPressure[:numOfEdges]
            print('Minimum velocity is {} m/s and maximum velocity is {} m/s'.format(np.amin(velocities), np.amax(velocities)))
            
            self.velocityPressure = velocityPressure
            self.validateFluidEquations(velocityPressure=velocityPressure)
            
            if saveResult:
                directory = self.directory
                saveFolderPath = join(directory, 'fluidSimulationResult')
                saveFileName = 'fluidSimulationResultTest6_Timestep={}_v5.pkl'.format(currentTimeStep)
                resultDict = {'G': copy.deepcopy(self.G), 'nodeInfoDict': copy.deepcopy(self.nodeInfoDict), 'edgeInfoDict': copy.deepcopy(self.edgeInfoDict), 
                              'velocityPressure': copy.deepcopy(velocityPressure)}
                with open(join(saveFolderPath, saveFileName), 'wb') as f:
                    pickle.dump(resultDict, f, 2)
                    print('{} saved to {}'.format(saveFileName, saveFolderPath))

            # Clear the simulation result #
            # for node in G.nodes():
            #     self.nodeInfoDict[node]['simulationData'] = {'pressure': None, 'flow': None} # placeholders, None means unset
            
            # for edgeIndex in edgeIndexList:
            #     self.edgeInfoDict[edgeIndex]['simulationData'] = {'velocity': None, 'flow': None} # placeholders, None means unset

    def showResult_GBMTest5(self, numOfTimeSteps=5):
        """
        Plot the result obtained from `GBMTest5`.
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        terminatingNodes = [node for node in self.G.nodes() if self.G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))

        directory = self.directory
        edgeIndexList = self.edgeIndexList
        resultFolderPath = join(directory, 'fluidSimulationResult')
        numOfTimeSteps = 5
        incomingEdgesFlowTimeStepArray = np.zeros((3, numOfTimeSteps))
        flowTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        terminatingNodes = [node for node in self.G.nodes() if self.G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        for currentTimeStep in range(numOfTimeSteps):
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            resultFileName = 'fluidSimulationResult_GBMTest5_Timestep={}_v1.pkl'.format(currentTimeStep)
            with open(join(resultFolderPath, resultFileName), 'rb') as f:
                resultDict = pickle.load(f)
                print('{} loaded from {}.'.format(resultFileName, resultFolderPath))
            
            if 'velocityPressure' not in resultDict:
                if 'perturbedYear' in resultDict:
                    G, nodeInfoDict, edgeInfoDict = itemgetter('G', 'nodeInfoDict', 'edgeInfoDict')(resultDict['perturbedYear'])
                    velocityPressure = resultDict['perturbedYear']['velocityPressure'] #self.getVelocityPressure()

                    with open(join(resultFolderPath, resultFileName), 'wb') as f:
                        resultDictNew = {'G': copy.deepcopy(G), 'nodeInfoDict': copy.deepcopy(nodeInfoDict), 'edgeInfoDict': copy.deepcopy(edgeInfoDict), 'velocityPressure': copy.deepcopy(velocityPressure)}
                        pickle.dump(resultDictNew, f, 2)
                        print('{} saved to {}.'.format(resultFileName, resultFolderPath))
            else:
                G, nodeInfoDict, edgeInfoDict = itemgetter('G', 'nodeInfoDict', 'edgeInfoDict')(resultDict)
                velocityPressure = resultDict['velocityPressure']
            
            self.G = copy.deepcopy(G)
            self.nodeInfoDict = copy.deepcopy(nodeInfoDict)
            self.edgeInfoDict = copy.deepcopy(edgeInfoDict)

            self.setupFluidEquations()
            eqnInfoDictList = self.eqnInfoDictList
            self.validateFluidEquations(velocityPressure=velocityPressure)
            
            self.updateNetworkWithSimulationResult(velocityPressure=velocityPressure)
            nodeInfoDict = self.nodeInfoDict
            edgeInfoDict = self.edgeInfoDict
            nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
            nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
            edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
            edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
            infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Pressure (mmHg)',
                        'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': r'Flow ($\mathrm{cm}^3/s$)',
                        'figTitle': 'GBM_Time step={}'.format(currentTimeStep)}
            
            if currentTimeStep == numOfTimeSteps - 1:
                isLastFigure = True
            else:
                isLastFigure = False

            self.plotNetwork(infoDict, figIndex=currentTimeStep+1, isLastFigure=False)

            # Collect some results
            for edgeIndex in [0,1,2]:
                incomingEdgesFlowTimeStepArray[edgeIndex, currentTimeStep] = edgeInfoDict[edgeIndex]['simulationData']['flow']
            
            for edgeIndex in edgeIndexList:
                flowTimeStepArray[edgeIndex, currentTimeStep] = edgeInfoDict[edgeIndex]['simulationData']['flow']
            
            terminatingPressures = [self.nodeInfoDict[node]['simulationData']['pressure'] /13560/9.8*1000 for node in terminatingNodes]
            terminatingPressuresTimeStepArray[:, currentTimeStep] = terminatingPressures

        # Flow proportions_GBMTest5
        # self.plotFlowProportion(flowTimeStepArray, figIndex=21, isLastFigure=False)
        # Mean terminating pressure vs Time step_GBMTest5 and Terminating pressure vs Time step_Compartment_GBMTest5
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=31, isLastFigure=False)
        self.plotTerminatingPressures2(terminatingNodes, terminatingPressuresTimeStepArray, option=2, figIndex=41, isLastFigure=True)
        
        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
        # plt.show()
    
    def showResult2_GBMTest5(self, numOfTimeSteps=5):
        """
        Show graph plot of pressures and flows from `GBMTest5` between two time steps and share one legend.
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        terminatingNodes = [node for node in self.G.nodes() if self.G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))

        directory = self.directory
        edgeIndexList = self.edgeIndexList
        resultFolderPath = join(directory, 'fluidSimulationResult')
        numOfTimeSteps = 5
        incomingEdgesFlowTimeStepArray = np.zeros((3, numOfTimeSteps))
        flowTimeStepArray = np.zeros((len(edgeIndexList), numOfTimeSteps))
        terminatingNodes = [node for node in self.G.nodes() if self.G.degree(node) == 1 and self.nodeInfoDict[node]['depth'] != 0]
        terminatingPressuresTimeStepArray = np.zeros((len(terminatingNodes), numOfTimeSteps))
        infoDictList = []
        nodeValueListTotal, edgeValueListTotal = [], []
        timeStepsToUse = [0, 4]
        for currentTimeStep in timeStepsToUse:
            print('##### currentTimeStep={} #####'.format(currentTimeStep))
            resultFileName = 'fluidSimulationResult_GBMTest5_Timestep={}_v1.pkl'.format(currentTimeStep)
            with open(join(resultFolderPath, resultFileName), 'rb') as f:
                resultDict = pickle.load(f)
                print('{} loaded from {}.'.format(resultFileName, resultFolderPath))
            
            G, nodeInfoDict, edgeInfoDict = itemgetter('G', 'nodeInfoDict', 'edgeInfoDict')(resultDict)
            velocityPressure = resultDict['velocityPressure']
            
            self.G = copy.deepcopy(G)
            self.nodeInfoDict = copy.deepcopy(nodeInfoDict)
            self.edgeInfoDict = copy.deepcopy(edgeInfoDict)

            self.updateNetworkWithSimulationResult(velocityPressure=velocityPressure)
            nodeInfoDict = self.nodeInfoDict
            edgeInfoDict = self.edgeInfoDict
            nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
            nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
            edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
            edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
            infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Pressure (mmHg)',
                        'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': r'Flow ($\mathrm{cm}^3/s$)',
                        'figTitle': 'GBM_Time step={}'.format(currentTimeStep)}
            
            # self.plotNetwork(infoDict, figIndex=currentTimeStep+1, isLastFigure=False)
            infoDictList.append(infoDict)
            nodeValueListTotal += nodeValueList
            edgeValueListTotal += edgeValueList
        
        vmin, vmax = np.amin(nodeValueListTotal), np.amax(nodeValueListTotal)
        edge_vmin, edge_vmax = np.amin(edgeValueListTotal), np.amax(edgeValueListTotal)
        figIndex = 1
        # fluidSimulationResult_Time step=0_Compare between two time steps_Same Legend_GBMTest5
        # fluidSimulationResult_Time step=4_Compare between two time steps_Same Legend_GBMTest5
        # fluidSimulationResult_Legend_Time step=0,4_GBMTest5
        for infoDict in infoDictList:
            infoDict['vmin'] = vmin
            infoDict['vmax'] = vmax
            infoDict['edge_vmin'] = edge_vmin
            infoDict['edge_vmax'] = edge_vmax
            self.plotNetwork(infoDict, figIndex=figIndex, isLastFigure=False, hideColorbar=True)
            figIndex += 1
        
        extraInfo = {'nodeLabel': 'Pressure (mmHg)', 'nodeLabelSize': 18, 'nodeTickSize': 18,
                     'edgeLabel': r'Flow rate ($\mathrm{cm}^3/s$)', 'edgeLabelSize': 18, 'edgeTickSize': 18, 
                     'vmin': vmin, 'vmax': vmax, 'edge_vmin': edge_vmin, 'edge_vmax': edge_vmax}
        self.graphPlotStandaloneLegend(figIndex=10, isLastFigure=True, extraInfo=extraInfo)


        
        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))
    
    def graphPlotStandaloneLegend(self, figIndex=1, isLastFigure=True, orientation='horizontal', extraInfo=None):
        """
        Standalone legend for the graph plot.
        """
        fig = plt.figure(figIndex, figsize=(12, 8))
        plt.subplots_adjust(left=0.15, right=0.85, top=0.94, bottom=0.06, wspace=0.3, hspace=0.9)
        if orientation == 'horizontal':
            ax1 = fig.add_axes([0.15, 0.9, 0.7, 0.04])
            ax2 = fig.add_axes([0.15, 0.75, 0.7, 0.04])
        elif orientation == 'vertical':
            ax1 = fig.add_axes([0.05, 0.05, 0.04, 0.9])
            ax2 = fig.add_axes([0.2, 0.05, 0.04, 0.9])
        
        vmin, vmax, edge_vmin, edge_vmax = itemgetter('vmin', 'vmax', 'edge_vmin', 'edge_vmax')(extraInfo)
        nodeColorNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=mpl.cm.jet, norm=nodeColorNorm, orientation=orientation)
        nodeLabel, nodeLabelSize, nodeTickSize = itemgetter('nodeLabel', 'nodeLabelSize', 'nodeTickSize')(extraInfo)
        cb1.set_label(nodeLabel, size=nodeLabelSize)
        cb1.ax.tick_params(labelsize=nodeTickSize)

        edgeColorNorm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=mpl.cm.jet, norm=edgeColorNorm, orientation=orientation)
        edgeLabel, edgeLabelSize, edgeTickSize = itemgetter('edgeLabel', 'edgeLabelSize', 'edgeTickSize')(extraInfo)
        cb2.set_label(edgeLabel, size=edgeLabelSize)
        cb2.ax.tick_params(labelsize=edgeTickSize)
        
        if isLastFigure:
            plt.show()

    def plotTerminatingPressures(self, figIndex=1, isLastFigure=True):
        """
        Plot distribution of terminating pressures per compartment.
        """
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        
        terminatingNodes = {'LMCA': [], 'RMCA': [], 'ACA': [], 'LPCA': [], 'RPCA': []}
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingNodes[partitionName] = terminatingNodesInThisPartition
        
        # fig = plt.figure(figIndex, figsize=(15, 8))
        # plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
        fig = plt.figure(figIndex, figsize=(15, 3)) # 1*5 figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        subplotIndex = 1
        for partitionName, info in partitionInfo.items():
            terminatingPressuresInThisPartition = [nodeInfoDict[node]['simulationData']['pressure']/13560/9.8*1000 for node in terminatingNodes[partitionName]]
            ax = fig.add_subplot(1,5,subplotIndex)
            ax.hist(terminatingPressuresInThisPartition, bins=10)
            ax.set_xlabel('Terminating pressure (mmHg)')
            ax.set_ylabel('Count')
            ax.set_title(partitionName)
            subplotIndex += 1
        
        if isLastFigure:
            plt.show()
    
    def plotTerminatingPressures2(self, terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=1, isLastFigure=True):
        """
        """
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        numOfTimeSteps = terminatingPressuresTimeStepArray.shape[1]
        # Line plot of terminating pressures (one line=one terminating node) per compartment
        if option == 1:
            # fig = plt.figure(1, figsize=(15, 8))
            # plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
            fig = plt.figure(figIndex, figsize=(15, 3))
            plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
            fig2 = plt.figure(figIndex+1, figsize=(15, 3))
            plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
            subplotIndex = 1
    
            G = self.G
            nodeInfoDict = self.nodeInfoDict
            edgeInfoDict = self.edgeInfoDict
            
            meanTerminatingPressuresPerPartitionArray = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
            for partitionName, info in partitionInfo.items():
                startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
                resultDict = self.BFS(startNodes, boundaryNodes)
                visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
                terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
                meanTerminatingPressuresList = []

                ax = fig.add_subplot(1,5,subplotIndex)
                ax2 = fig2.add_subplot(1,5,1)
                # Terminating pressures vs Time steps(5)_Same flow_All CoW branches fixed_Compartment(5)_Single row
                # Terminating pressures vs Time steps(5)_Same flow_LICA RICA VA fixed_Compartment(5)_Single row
                # Terminating pressures vs Time steps(5)_Split flow with radius_All CoW branches fixed_Compartment(5)_Single row
                # Terminating pressures vs Time steps(5)_Split flow with radius_LICA RICA VA fixed_Compartment(5)_Single row
                for ii, node in enumerate(terminatingNodesInThisPartition):
                    rowNum = terminatingNodes.index(node)
                    pressures = terminatingPressuresTimeStepArray[rowNum, :]
                    xValues = list(range(numOfTimeSteps))
                    yValues = list(pressures)
                    ax.plot(xValues, yValues, 'o-')
                    ax.set_xlabel('Time step')
                    ax.set_xticks(xValues)
                    ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
                    if subplotIndex == 1:
                        ax.set_ylabel('Terminating pressure (mmHg)')
                    ax.set_title(partitionName)
                    meanTerminatingPressuresList.append(pressures)
                
                # Mean terminating pressures vs Time steps(5)_Same flow_All CoW branches fixed_Compartment(5)_Single row
                # Mean terminating pressures vs Time steps(5)_Same flow_LICA RICA VA fixed_Compartment(5)_Single row
                # Mean terminating pressures vs Time steps(5)_Split flow with radius_All CoW branches fixed_Compartment(5)_Single row
                # Mean terminating pressures vs Time steps(5)_Split flow with radius_LICA RICA VA fixed_Compartment(5)_Single row
                # ax2 = fig2.add_subplot(1,5,subplotIndex)
                meanTerminatingPressuresArray = np.array(meanTerminatingPressuresList)
                xValues = list(range(numOfTimeSteps))
                yValues = np.mean(meanTerminatingPressuresArray, axis=0)
                meanTerminatingPressuresPerPartitionArray[partitionName] = yValues
                ax2.plot(xValues, yValues, 'o-', label=partitionName)
                ax2.set_xlabel('Time step')
                ax2.set_xticks(xValues)
                ax2.set_xticklabels(['T{}'.format(ii) for ii in xValues])
                # if subplotIndex == 1:
                ax2.set_ylabel('Mean terimating pressure (mmHg)')
                # ax2.set_title(partitionName)
                
                subplotIndex += 1
            ax2.legend(prop={'size': 6})

            ax3 = fig2.add_subplot(1,5,2)
            xValues = list(range(numOfTimeSteps))
            yValuesLeft = (meanTerminatingPressuresPerPartitionArray['LMCA'] + meanTerminatingPressuresPerPartitionArray['LPCA']) / 2
            yValuesRight = (meanTerminatingPressuresPerPartitionArray['RMCA'] + meanTerminatingPressuresPerPartitionArray['RPCA']) / 2
            ax3.plot(xValues, yValuesLeft, 'o-', label='Left')
            ax3.plot(xValues, yValuesRight, 'o-', label='Right')
            ax3.set_xlabel('Time step')
            ax3.set_xticks(xValues)
            ax3.set_xticklabels(['T{}'.format(ii) for ii in xValues])
            ax3.legend()

        # Each plot represents a time step and shows TP distribution of different compartments (one color for each)
        # Terminating pressures distribution per time step_Same flow_All CoW branches fixed_Compartment(5)
        # Terminating pressures distribution per time step_Same flow_LICA RICA VA fixed_Compartment(5)
        # Terminating pressures distribution per time step_Split flow with radius_All CoW branches fixed_Compartment(5)
        # Terminating pressures distribution per time step_Split flow with radius_LICA RICA VA fixed_Compartment(5)
        elif option == 2:
            # Terminating pressure vs Time step vs Compartment(5)_3D Histogram_Same flow_All CoW branches fixed
            # Terminating pressure vs Time step vs Compartment(5)_3D Histogram_Same flow_LICA RICA VA fixed
            # Terminating pressure vs Time step vs Compartment(5)_3D Histogram_Split flow with radius_All CoW branches fixed
            # Terminating pressure vs Time step vs Compartment(5)_3D Histogram_Split flow with radius_LICA RICA VA fixed
            fig = plt.figure(figIndex, figsize=(8, 5))
            plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.5)
            
            subplotIndex = 1
            G = self.G
            nodeInfoDict = self.nodeInfoDict
            edgeInfoDict = self.edgeInfoDict
            nbins = 10
            colorList = ['r', 'g', 'b', 'y', 'c', 'm']
            colorDict = {'LMCA': 'r', 'RMCA': 'g', 'LPCA': 'b', 'RPCA': 'y', 'ACA': 'c'}
            ax = fig.add_subplot(1,1,subplotIndex, projection='3d')
            for currentTimeStep in range(numOfTimeSteps):
                data = []
                counter = 0
                for partitionName, info in partitionInfo.items():
                    startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
                    resultDict = self.BFS(startNodes, boundaryNodes)
                    visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
                    terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
                    terminatingPressuresInThisPartition = [terminatingPressuresTimeStepArray[terminatingNodes.index(node), currentTimeStep] for node in terminatingNodesInThisPartition]
                    data.append(terminatingPressuresInThisPartition)

                    # Bar plot (still is histogram)
                    hist, bins = np.histogram(terminatingPressuresInThisPartition, bins=nbins)
                    xs = (bins[:-1] + bins[1:])/2
                    color = colorDict[partitionName]
                    # if partitionName != 'LMCA':
                    #     continue
                    ax.bar(xs, hist, zs=currentTimeStep*10, zdir='y', color=color, ec=color, alpha=0.8)
                    # ax.bar3d(xs, counter*10, 0, 1, 0.1, hist, color=color, alpha=0.8)
                    ax.set_xlabel('Terminating pressure (mmHg)')
                    ax.set_ylabel('Time step')
                    ax.set_yticks([ii*10 for ii in range(numOfTimeSteps)])
                    ax.set_yticklabels(['T{}'.format(ii) for ii in range(numOfTimeSteps)])
                    ax.set_zlabel('Count')
                    
                    counter += 1
                
                subplotIndex += 1
            
            f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
            # ax.legend(list(partitionInfo.keys()), loc="upper right", bbox_to_anchor=f(30,45,10), bbox_transform=ax.transData) # for test3
            # ax.legend(list(partitionInfo.keys()), loc="upper right", bbox_to_anchor=f(50,45,6), bbox_transform=ax.transData) # for test5
            ax.legend(list(partitionInfo.keys()), loc="upper right", bbox_to_anchor=f(65,45,4), bbox_transform=ax.transData) # for showResult_GBMTest5
            # ax.legend(list(partitionInfo.keys()))

        # Each plot represents a compartment and shows distribution of different time steps (one color for each)
        # Terminating pressures distribution per compartment(5)_Same flow_All CoW branches fixed
        # Terminating pressures distribution per compartment(5)_Same flow_LICA RICA VA fixed
        # Terminating pressures distribution per compartment(5)_Split flow with radius_All CoW branches fixed
        # Terminating pressures distribution per compartment(5)_Split flow with radius_LICA RICA VA fixed
        elif option == 3:
            fig = plt.figure(figIndex, figsize=(9, 8))
            plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.5)
            
            subplotIndex = 1
            G = self.G
            nodeInfoDict = self.nodeInfoDict
            edgeInfoDict = self.edgeInfoDict
            nbins = 10
            colorList = ['r', 'g', 'b', 'y', 'c', 'm']
            colorDict = {'LMCA': 'r', 'RMCA': 'g', 'LPCA': 'b', 'RPCA': 'y', 'ACA': 'c'}
            ax = fig.add_subplot(1,1,subplotIndex, projection='3d')
            partitionCounter = 0
            for partitionName, info in partitionInfo.items():
                startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
                resultDict = self.BFS(startNodes, boundaryNodes)
                visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
                terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]

                data = []
                for currentTimeStep in range(numOfTimeSteps):
                    terminatingPressuresAtThisTimeStep = [terminatingPressuresTimeStepArray[terminatingNodes.index(node), currentTimeStep] for node in terminatingNodesInThisPartition]
                    data.append(terminatingPressuresAtThisTimeStep)
                
                # Bar plot (still is histogram)
                    hist, bins = np.histogram(terminatingPressuresAtThisTimeStep, bins=nbins)
                    xs = (bins[:-1] + bins[1:])/2
                    color = colorList[currentTimeStep]
                    # if partitionName != 'LMCA':
                    #     continue
                    ax.bar(xs, hist, zs=partitionCounter*10, zdir='y', color=color, ec=color, alpha=0.8)
                    ax.set_xlabel('Terminating Pressure (mmHg)')
                    ax.set_ylabel('Compartment')
                    ax.set_zlabel('Count')
                    
                partitionCounter += 1

                subplotIndex += 1
        
        if isLastFigure:
            plt.show()
    
    def plotFlow(self, flowTimeStepArray, option=1, figIndex=1, isLastFigure=True):
        """
        Plot the flow to each of the compartments.
        """
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]}, 'ACA': {'startNodes': [10], 'boundaryNodes': []},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}}
        timestepList = []
        numOfTimeSteps = flowTimeStepArray.shape[1]
        fig = plt.figure(figIndex, figsize=(8, 3))
        plt.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        # Total flow vs Time steps(5)_Split flow with radius_All CoW branches fixed_Compartment(5)_Single row
        # Total flow vs Time steps(5)_Split flow with radius_LICA RICA VA fixed_Compartment(5)_Single row
        subplotIndex = 1
        ax = fig.add_subplot(1,1,subplotIndex)
        ax.set_xlabel('Time step')
        # ax.set_ylabel(r'Flow ($\mathrm{cm}^3 /s$)')
        ax.set_ylabel('Percentage of flow change (%)')
        for partitionName in ['LMCA', 'RMCA', 'LPCA', 'RPCA', 'ACA', 'Left', 'Right']:
            if partitionName == 'LMCA':
                flowValues = flowTimeStepArray[8, :] / flowTimeStepArray[8, 0] * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
                print('LMCA flow: {}'.format(flowValues))
            elif partitionName == 'RMCA':
                flowValues = flowTimeStepArray[10, :] / flowTimeStepArray[10, 0] * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
            elif partitionName == 'LPCA':
                flowValues = flowTimeStepArray[5, :] / flowTimeStepArray[5, 0] * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
                print('LPCA flow: {}'.format(flowValues))
            elif partitionName == 'RPCA':
                flowValues = flowTimeStepArray[6, :] / flowTimeStepArray[6, 0] * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
                print('RPCA flow: {}'.format(flowValues))
            elif partitionName == 'ACA':
                flowValues = flowTimeStepArray[20, :] / flowTimeStepArray[20, 0] * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
            elif partitionName == 'Left':
                flowValues = (flowTimeStepArray[8, :] + flowTimeStepArray[5, :]) / (flowTimeStepArray[8, 0] + flowTimeStepArray[5, 0]) * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
            elif partitionName == 'Right':
                flowValues = (flowTimeStepArray[10, :] + flowTimeStepArray[6, :]) / (flowTimeStepArray[10, 0] + flowTimeStepArray[6, 0]) * 100 - 100
                timeStepValues = list(range(numOfTimeSteps))
            
            # ax = fig.add_subplot(1,1,subplotIndex)
            ax.plot(timeStepValues, flowValues, 'o-', label=partitionName)
            print('{}: {}% change in flow at 2013'.format(partitionName, np.round(flowValues[-1], 2)))
        
        ax.set_xticks(list(range(numOfTimeSteps)))
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=7)
        
        if isLastFigure:
            plt.show()
    
    def plotRootPressuresCompartment(self, nodePressuresTimeStepArray, option=1, figIndex=1, isLastFigure=True):
        """
        Plot the root pressures of each compartment over time.

        Root pressure per compartment_Split flow with radius_All CoW branches fixed
        Root pressure per compartment_Split flow with radius_LICA RICA VA fixed
        Root pressure per compartment_Same flow_LICA RICA VA fixed
        """
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        numOfTimeSteps = nodePressuresTimeStepArray.shape[1]

        fig = plt.figure(figIndex, figsize=(6, 3))
        plt.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        subplotIndex = 1
        ax = fig.add_subplot(1,1,subplotIndex)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Root pressure (mmHg)')
        for partitionName in ['LMCA', 'RMCA', 'LPCA', 'RPCA', 'ACA']:
            rootNode = partitionInfo[partitionName]['startNodes'][0]
            rootPressures = nodePressuresTimeStepArray[rootNode, :]
            timeStepValues = list(range(numOfTimeSteps))
            print(rootPressures)

            ax.plot(timeStepValues, rootPressures, 'o-', label=partitionName)
        
        ax.set_xticks(list(range(numOfTimeSteps)))
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        
        
        if isLastFigure:
            plt.show()
    
    def plotTerminatingPressureVSPathLength(self, terminatingNodes, terminatingPressuresTimeStepArray, option=1, figIndex=1, isLastFigure=True):
        """
        Scatter plot of terminating pressure vs path length.
        """
        G = self.G
        edgeList = self.edgeList
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        edgeIndexList = self.edgeIndexList
        spacing = self.spacing
        directory = self.directory

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10], 'color': 'r'}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10], 'color': 'g'},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': [], 'color': 'b'}, 'RPCA': {'startNodes': [7], 'boundaryNodes': [], 'color': 'y'},
                         'ACA': {'startNodes': [10], 'boundaryNodes': [], 'color': 'c'}}
        if option == 1:
            pass
        elif option == 2:
            pass
        elif option == 3:
            pass
        
        terminatingPressureVSPathLength = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        terminatingNodesPerPartition = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingNodesPerPartition[partitionName] = terminatingNodesInThisPartition
            terminatingPressuresInThisPartition = []
            for terminatingNode in terminatingNodesInThisPartition:
                path = nx.shortest_path(G, startNodes[0], terminatingNode)
                pathEdgeIndexList = [G[path[ii]][path[ii + 1]]['edgeIndex'] for ii in range(len(path) - 1)]
                uniquePathEdgeIndexList = np.unique(pathEdgeIndexList)
                assert len(uniquePathEdgeIndexList) != 0
                pathLength = np.sum([edgeInfoDict[edgeIndex]['length'] * spacing * 1000 for edgeIndex in uniquePathEdgeIndexList]) # millimeter
                pressure = nodeInfoDict[terminatingNode]['simulationData']['pressure'] / 13560 / 9.8 * 1000 # mmHg
                terminatingPressureVSPathLength[partitionName].append([pathLength, pressure])
                nodeInfoDict[terminatingNode]['pathLength'] = pathLength
                nodeInfoDict[terminatingNode]['partitionName'] = partitionName
        
        terminatingNodesPathLengthList = [nodeInfoDict[node]['pathLength'] for node in terminatingNodes]
        fig = plt.figure(figIndex, figsize=(15, 3))
        plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        numOfTimeSteps = np.shape(terminatingPressuresTimeStepArray)[1]
        for currentTimeStep in range(0, numOfTimeSteps):
            ax = fig.add_subplot(1,5,currentTimeStep+1)
            for partitionName, info in partitionInfo.items():
                terminatingNodesInThisPartition = terminatingNodesPerPartition[partitionName]
                pathLengthList = [nodeInfoDict[node]['pathLength'] for node in terminatingNodesInThisPartition]
                pressureList = [terminatingPressuresTimeStepArray[terminatingNodes.index(node), currentTimeStep] for node in terminatingNodesInThisPartition]
                color = info['color']
                ax.scatter(pathLengthList, pressureList, c=color, label=partitionName)
            
            ax.legend(prop={'size': 6})
            ax.set_xlabel('Path length (mm)')
            ax.set_ylabel('Terminating pressure (mmHg)')
            ax.set_title('Timestep={}'.format(currentTimeStep))
        
        if isLastFigure:
            plt.show()
    
    def plotFlowProportion(self, flowTimeStepArray, figIndex=1, isLastFigure=True):
        """
        """
        numOfTimeSteps = flowTimeStepArray.shape[1]
        # Flow proportions_GBMTest5    
        fig = plt.figure(figIndex, figsize=(15, 3))
        plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        # incomingFlowInfo_GBMTest5
        ax = fig.add_subplot(1,5,1)
        edgeNameList = ['LICA', 'VA', 'RICA'] # corresponds to edgeIndex=0/1/2
        for edgeIndex in [0,1,2]:
            xValues = list(range(numOfTimeSteps))
            yValues = flowTimeStepArray[edgeIndex, :] / np.sum(flowTimeStepArray[:3, :], axis=0)
            ax.plot(xValues, yValues, 'o-', label=edgeNameList[edgeIndex])
            ax.set_xlabel('Time step')
            ax.set_xticks(xValues)
            ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
            ax.set_ylabel('Flow proportion')
        
        ax.legend()
        
        ax = fig.add_subplot(1,5,2)
        partitionProportionTimeStepDict = {'LMCA': [], 'RMCA': [], 'LPCA': [], 'RPCA': [], 'ACA': []}
        partitionProportionTimeStepDict['LMCA'] = flowTimeStepArray[8, :] / flowTimeStepArray[3, :]
        partitionProportionTimeStepDict['RMCA'] = flowTimeStepArray[10, :] / (flowTimeStepArray[4, :] + flowTimeStepArray[7, :])
        partitionProportionTimeStepDict['LPCA'] = flowTimeStepArray[5, :] / flowTimeStepArray[1, :]
        partitionProportionTimeStepDict['RPCA'] = flowTimeStepArray[6, :] / flowTimeStepArray[1, :]
        partitionProportionTimeStepDict['ACA'] = flowTimeStepArray[20, :] / (flowTimeStepArray[9, :] + flowTimeStepArray[11, :])
        for partitionName, proportionList in partitionProportionTimeStepDict.items():
            xValues = list(range(numOfTimeSteps))
            yValues = proportionList
            ax.plot(xValues, yValues, 'o-', label=partitionName)

        ax.set_xlabel('Time step')
        ax.set_xticks(xValues)
        ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
        ax.set_ylabel(r'Compartment flow proportion')
        ax.legend(prop={'size': 6})
        # ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5, prop={'size': 6})

        ax = fig.add_subplot(1,5,3)
        xValues = list(range(numOfTimeSteps))
        yValues = np.sum(flowTimeStepArray[:3, :], axis=0) * 10**6
        ax.plot(xValues, yValues, 'o-')
        ax.set_xlabel('Time step')
        ax.set_xticks(xValues)
        ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
        ax.set_ylabel(r'Total flow rate ($\mathrm{cm}^3/s$)')

        ax = fig.add_subplot(1,5,4)
        xValues = list(range(numOfTimeSteps))
        edgeNameDict = {0: 'LICA', 1: 'VA', 2: 'RICA', 4: 'RPCA Comm', 9: 'LM', 11: 'RM', 8: 'LMCA', 20: 'ACA', 10: 'RMCA', 5: 'LPCA', 6: 'RPCA'}

        from matplotlib.cm import get_cmap
        name = "tab20"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list
        ax.set_prop_cycle(color=colors)

        for edgeIndex, edgeName in edgeNameDict.items():
            yValues = flowTimeStepArray[edgeIndex, :] * 10**6
            ax.plot(xValues, yValues, 'o-', label=edgeName)
        ax.set_xlabel('Time step')
        ax.set_xticks(xValues)
        ax.set_xticklabels(['T{}'.format(ii) for ii in xValues])
        ax.set_ylabel(r'Flow rate ($\mathrm{cm}^3/s$)')
        # ax.legend(prop={'size': 6})
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        if isLastFigure:
            plt.show()

    def BFSTest(self):
        """
        Test BFS function, and plot radius vs graph level for each compartment
        GBM_BraVa distribution_Radius vs Graph level_Compartment(5)_Full range_Single row
        """
        partitionInfo = {'LMCA': {'startNodes': [5], 'boundaryNodes': [13]}, 'RMCA': {'startNodes': [6], 'boundaryNodes': [13]},
                         'LPCA': {'startNodes': [4], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [13], 'boundaryNodes': []}}
        
        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)

        # edgeIndexList = self.edgeIndexList
        # extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,4,7,12]}
        # self.perturbNetwork(option=2, extraInfo=extraInfo)
        # self.setNetwork(option=2)

        # fig = plt.figure(1, figsize=(15, 8))
        # plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3, hspace=0.3)
        fig = plt.figure(11, figsize=(15, 3))
        plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        
        subplotIndex = 1
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        spacing = self.spacing
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            print('{}:\nvisitedNodes={}\nvisitedEdges={}'.format(partitionName, visitedNodes, visitedEdges))

            ax = fig.add_subplot(1,5,subplotIndex)
            dictUsed = edgeInfoDict
            attribute1, attribute2, attribute3 = 'segmentLevel', 'meanRadius', 'partitionName'
            attribute1List = [edgeInfoDict[edgeIndex]['depth'] for edgeIndex in visitedEdges]
            attribute2List = [edgeInfoDict[edgeIndex]['meanRadius']*spacing*1000 for edgeIndex in visitedEdges]
            # attribute1List = [info[attribute1] for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] in partitionNames]
            # attribute2List = [info[attribute2]*spacing*1000 for _, info in dictUsed.items() if attribute1 in info and attribute2 in info and attribute3 in info and info[attribute3] in partitionNames] # mm
            # ax.plot(attribute1List, attribute2List, 'bo')
            positions = np.sort(np.unique(attribute1List))
            values = []
            attribute1Array, attribute2Array = np.array(attribute1List), np.array(attribute2List)
            for segmentLevel in positions:
                locs = np.nonzero(attribute1Array == segmentLevel)[0]
                values.append((attribute2Array[locs]).tolist())
        
            mf.boxPlotWithWhiskers(values, ax, positions=positions, whis='range', xlabel='Graph level', ylabel='Radius (mm)')
            ax.set_xlabel('Graph level')
            ax.set_ylabel('Radius (mm)')
            ax.set_title(partitionName)

            subplotIndex += 1
            # if partitionName == 'LPCA':
            #     print(sorted(attribute2List))
        
        plt.show()
    
    def examineFluidResult(self):
        """
        Examine the result obtained by solving the network.
        """
        start_time = timeit.default_timer()
        functionName = inspect.currentframe().f_code.co_name
        resultDict = {'referenceYear': {}, 'perturbedYear': {}}
        self.loadNetwork(version=4, year='BraVa')
        self.convertNetowrk()
        self.adjustNetwork()
        self.setNetwork(option=2)
        self.createGroundTruth(option=2)

        loadFileName = 'fluidSimulationResultGBMTest2(referenceYear=BraVa, perturbedYear=2013, perturbTerminatingPressureOption=1).pkl'
        _, _, _, resultDict = self.loadFluidResult(loadFileName, return_ResultDict=True)
        self.nodeInfoDict = resultDict['referenceYear']['nodeInfoDict']
        self.edgeInfoDict = resultDict['referenceYear']['edgeInfoDict']

        partitionInfo = {'LMCA': {'startNodes': [4], 'boundaryNodes': [10]}, 'RMCA': {'startNodes': [5], 'boundaryNodes': [10]},
                         'LPCA': {'startNodes': [6], 'boundaryNodes': []}, 'RPCA': {'startNodes': [7], 'boundaryNodes': []}, 'ACA': {'startNodes': [10], 'boundaryNodes': []}}
        
        terminatingNodes = {'LMCA': [], 'RMCA': [], 'ACA': [], 'LPCA': [], 'RPCA': []}
        terminatingPressures = {'LMCA': [], 'RMCA': [], 'ACA': [], 'LPCA': [], 'RPCA': []}
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        for partitionName, info in partitionInfo.items():
            startNodes, boundaryNodes = itemgetter('startNodes', 'boundaryNodes')(info)
            resultDict = self.BFS(startNodes, boundaryNodes)
            visitedNodes, visitedEdges = itemgetter('visitedNodes', 'visitedEdges')(resultDict)
            terminatingNodesInThisPartition = [node for node in visitedNodes if G.degree(node) == 1 and nodeInfoDict[node]['depth'] != 0]
            terminatingNodes[partitionName] = terminatingNodesInThisPartition
            terminatingPressuresInThisPartition = [nodeInfoDict[node]['simulationData']['pressure']/13560/9.8*1000 for node in terminatingNodesInThisPartition]
            terminatingPressures[partitionName].append(terminatingPressuresInThisPartition)
        
        # GBM reference flow_BraVa time step_Ground truth option=2
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Pressure (mmHg)',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': r'Flow rate ($\mathrm{cm}^3/s$)',
                    'figTitle': 'GBM Reference (BraVa)'} # TP->terminating pressure
        self.plotNetwork(infoDict, figIndex=1, isLastFigure=False)
        
        # Manually perturb the network #
        extraInfo = {'perturbedYear': 2013, 'excludedEdgeIndex': [0,1,2,3,7]}
        self.perturbNetwork(option=2, extraInfo=extraInfo)
        self.setNetwork(option=2)

        # Load result
        loadFileName = 'fluidSimulationResultGBMTest2(referenceYear=BraVa, perturbedYear=2013, perturbTerminatingPressureOption=1).pkl'
        nodeInfoDictPerturbed, edgeInfoDictPerturbed, velocityPressurePerturbed = self.loadFluidResult(loadFileName)
        self.nodeInfoDict = nodeInfoDictPerturbed
        self.edgeInfoDict = edgeInfoDictPerturbed
        self.setupFluidEquations()
        self.validateFluidEquations(velocityPressure=velocityPressurePerturbed)

        for partitionName, info in partitionInfo.items():
            terminatingNodesInThisPartition = terminatingNodes[partitionName]
            terminatingPressuresInThisPartition = [self.nodeInfoDict[node]['simulationData']['pressure']/13560/9.8*1000 for node in terminatingNodesInThisPartition]
            terminatingPressures[partitionName].append(terminatingPressuresInThisPartition)
        
        # GBM fluid solution_GBMTest2(referenceYear=BraVa, perturbedYear=2013, perturbTerminatingPressureOption=1)
        G = self.G
        nodeInfoDict = self.nodeInfoDict
        edgeInfoDict = self.edgeInfoDict
        nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
        nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
        edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
        edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
        infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Pressure (mmHg)',
                    'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': r'Flow rate ($\mathrm{cm}^3/s$)',
                    'figTitle': 'GBM {}'.format(extraInfo['perturbedYear'])} # TP->terminating pressure
        self.plotNetwork(infoDict, figIndex=3, isLastFigure=False)
        
        # GBM terminating pressure distribution per compartment(referenceYear=BraVa, perturbedYear=2013, perturbTerminatingPressureOption=1)
        self.plotTerminatingPressures(figIndex=11, isLastFigure=True)

        # fig = plt.figure(20, figsize=(15, 3)) # 1*5 figure
        # plt.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)
        # subplotIndex = 1
        # for partitionName, info in partitionInfo.items():
        #     pressures = terminatingPressures[partitionName]
        #     ax = fig.add_subplot(1,5,subplotIndex)
        #     ax.hist(pressures, bins=10)
        #     ax.set_xlabel('Terminating pressure (mmHg)')
        #     ax.set_ylabel('Count')
        #     ax.set_title(partitionName)
        #     ax.legend(['BraVa', '2013'])
        #     subplotIndex += 1

        elapsed = timeit.default_timer() - start_time
        print('Elapsed time for function {}: {} sec'.format(functionName, elapsed))

        # plt.show()

def computeNetworkDetail(args, eqnInfoDictList, method='HW', errorNorm=0, extraInfo=None):
    """
    Given a network, the inlet pressure and all the terminating pressure, find the velocity/pressure of the remaining branches/nodes.
    """
    rouBlood = 1050 # kg/m^3
    rouMercury = 13560 # kg/m^3
    g = 9.8 # m/s^2
    rougBlood = rouBlood * g
    # f in D-W equation = 64/Re = 64/(vD/nu) = 64*nu/(v*D) where nu is kinematic viscosity, D is diameter
    # nu for blood is 2.6e-6 m^2/s
    eqnList, eqnFlowList, eqnPressureList, eqnBoundaryList = [], [], [], []

    for eqnInfoDict in eqnInfoDictList:
        eqnType = eqnInfoDict['type']
        if eqnType == 'flow':
            velocityInIndexList, radiusInList = eqnInfoDict['velocityInIndexList'], eqnInfoDict['radiusInList']
            velocityOutIndexList, radiusOutList = eqnInfoDict['velocityOutIndexList'], eqnInfoDict['radiusOutList']
            velocityInList = [args[velocityIndex] for velocityIndex in velocityInIndexList]
            velocityOutList = [args[velocityIndex] for velocityIndex in velocityOutIndexList]
            QIn = np.sum([np.abs(velocity) * np.pi * radius**2 for velocity, radius in zip(velocityInList, radiusInList)])
            QOut = np.sum([np.abs(velocity) * np.pi * radius**2 for velocity, radius in zip(velocityOutList, radiusOutList)])
            eqn = np.abs(QIn - QOut)
            eqnFlowList.append(eqn)
        elif eqnType == 'pressure':
            radius, length, velocityIndex, c, k, edgeIndex = itemgetter('radius', 'length', 'velocityIndex', 'c', 'k', 'edgeIndex')(eqnInfoDict)
            velocity = np.abs(args[velocityIndex])
            if 'pressure' in eqnInfoDict['headPressureInfo']:
                headPressure = eqnInfoDict['headPressureInfo']['pressure']
            elif 'pressureIndex' in eqnInfoDict['headPressureInfo']:
                pressureIndex = eqnInfoDict['headPressureInfo']['pressureIndex']
                headPressure = args[pressureIndex]
            
            if 'pressure' in eqnInfoDict['tailPressureInfo']:
                tailPressure = eqnInfoDict['tailPressureInfo']['pressure']
            elif 'pressureIndex' in eqnInfoDict['tailPressureInfo']:
                pressureIndex = eqnInfoDict['tailPressureInfo']['pressureIndex']
                tailPressure = args[pressureIndex]
            
            if method == 'HW':
                
                deltaPressureByNode = headPressure - tailPressure
                deltaPressureByHW = 10.67 * (velocity * np.pi * radius**2)**k * length / c**k / (2 * radius)**4.8704
                if np.isnan(deltaPressureByHW):
                    print('velocity={}, radius={}, length={}, c={}, k={}'.format(velocity, radius, length, c, k))
                
                if headPressure > tailPressure:
                    eqn = np.abs(deltaPressureByNode - deltaPressureByHW)*2
                else:
                    eqn = 10 * np.abs(tailPressure + deltaPressureByHW - headPressure)
                
                if extraInfo is not None and 'excludedEdgeIndex' in extraInfo:
                    excludedEdgeIndex = extraInfo['excludedEdgeIndex']
                    if edgeIndex in excludedEdgeIndex:
                        eqn /= 100 # put less weight on the errors from the specified edges

                eqnPressureList.append(eqn)
            elif method == 'DW':
                pass
        elif eqnType == 'boundary':
            velocityIndex, velocityIn = eqnInfoDict['velocityIndex'], eqnInfoDict['velocityIn']
            eqn = args[velocityIndex] - velocityIn
            eqnBoundaryList.append(eqn*10)
    
    if np.any(np.isnan(eqnFlowList)):
        print('eqnFlowList has NaNs')
    elif np.any(np.isinf(eqnFlowList)):
        print('eqnFlowList has infs')
    
    if np.any(np.isnan(eqnPressureList)):
        print('eqnPressureList has NaNs')
    elif np.any(np.isinf(eqnPressureList)):
        print('eqnPressureList has infs')
    
    if len(eqnFlowList) == 0:
        print('eqnFlowList is empty')
    elif len(eqnPressureList) == 0:
        print('eqnPressureList is empty')
    
    flowErrorFactor = 10**6 * 20000 # 10**6 convert to cm^3/s, another 20000 to focus more on flow error
    pressureErrorFactor = 1000/13560/9.8 * 500 # convert to mmHg, another 50 to create larger motivation for optimization
    eqnFlowList = [eqn * flowErrorFactor for eqn in eqnFlowList] # ensure that magnitude of flow error is roughly the same as that of pressure error
    eqnPressureList = [eqn * pressureErrorFactor for eqn in eqnPressureList] # ensure that magnitude of flow error is roughly the same as that of pressure error
    # eqnList = eqnFlowList + eqnPressureList
    eqnList = eqnFlowList + eqnPressureList + eqnBoundaryList

    if errorNorm != 0:
        error = norm(eqnList, ord=errorNorm)
    else:
        error = eqnList

    # print('Error={}'.format(error))
    
    return error

def distributeFlowDetail(args, distributeFlowEqnDict, nodeInfoDict):
    """
    Given a network and the inlet pressure/flow, find the distribution of flow in each branches such that the terminating pressures match the desired values.
    """
    connectInfoDictList, mergeInfoDict, desiredTerminatingPressures = itemgetter('connectInfoDictList', 'mergeInfoDict', 'desiredTerminatingPressures')(distributeFlowEqnDict)
    for connectInfoDict in connectInfoDictList:
        # Unpack # 
        connection, edgeInfo = itemgetter('connection', 'edgeInfo')(connectInfoDict)
        headNode, edgeIndex, tailNode = connection
        radius, length, c, k = itemgetter('radius', 'length', 'c', 'k')(edgeInfo)
        flowIn = nodeInfoDict[headNode]['simulationData']['flow'] # m^3/s
        headPressure = np.mean(nodeInfoDict[headNode]['simulationData']['pressure']) # Pascal, use average value in case of a merging node
        splitPercentage = args[edgeIndex] # 0-1
        # Calculate #
        flowUsed = flowIn * splitPercentage
        deltaPressure = 10.67 * flowUsed**k * length / c**k / (2*radius)**4.8704 # Pascal
        tailPressure = headPressure - deltaPressure
        # Save #
        if nodeInfoDict[tailNode]['simulationData']['pressure'] is None:
            nodeInfoDict[tailNode]['simulationData']['pressure'] = tailPressure
        else:
            # Use optimization to minimize the difference between these two pressures
            nodeInfoDict[tailNode]['simulationData']['pressure'] = [nodeInfoDict[tailNode]['simulationData']['pressure'], tailPressure]

def main():
    fn = FluidNetwork()
    # fn.generateNetwork(maxDepth=5, allowMerge=True)
    # edgeList = fn.edgeList
    # temp = [[ii, edge] for ii, edge in enumerate(edgeList)]
    # print(temp)
    # fn.setNetwork(option=1)
    # ##
    # G = fn.getNetwork()
    # nodeInfoDict = fn.nodeInfoDict
    # edgeInfoDict = fn.edgeInfoDict
    # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
    # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
    # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
    # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
    # infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
    #             'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
    #             'figTitle': 'Test'}
    # fn.plotNetwork(infoDict)
    # ##
    # fn.createGroundTruth()
    # fn.setupFluidEquations()
    # fn.validateFluidEquations()
    # fn.computeNetworkTest(saveResult=True)
    # fn.GBMTest(saveResult=True)
    # fn.GBMTest2(perturbTerminatingPressureOption=1, saveResult=True)
    # fn.GBMTest2(perturbTerminatingPressureOption=2, saveResult=True)
    # fn.GBMTest2(perturbTerminatingPressureOption=3, saveResult=True)
    # fn.GBMTest2(perturbTerminatingPressureOption=4, saveResult=True)
    # fn.argsBoundTest()
    # fn.distributeFlowTest()
    # fn.compareNetworkPropertyTest()
    # fn.test1(numOfTimeSteps=5, interpolationOption=1, saveResult=False)
    # fn.test2()
    # fn.test3(numOfTimeSteps=5, interpolationOption=1, saveResult=False)
    # fn.test4()
    # fn.test5(numOfTimeSteps=5, interpolationOption=1, saveResult=False)
    # fn.BFSTest()
    # fn.examineFluidResult()
    # fn.test6(numOfTimeSteps=5, interpolationOption=1, saveResult=True)
    # fn.GBMTest3(saveResult=True)
    # fn.GBMTest4(perturbNetworkOption=3, saveResult=True)
    # fn.showVolumePerPartition(numOfTimeSteps=5, interpolationOption=1, figIndex=1, isLastFigure=True)
    # fn.GBMTest5(numOfTimeSteps=5, interpolationOption=1, saveResult=True)
    # fn.GBMTest5b(numOfTimeSteps=5, interpolationOption=1, saveResult=True)
    fn.showResult_GBMTest5(numOfTimeSteps=5)
    # fn.showResult2_GBMTest5(numOfTimeSteps=5)
    # fn.GBMTest6(numOfTimeSteps=5, interpolationOption=1, saveResult=True)

    G = fn.getNetwork()
    nodeInfoDict = fn.nodeInfoDict
    edgeInfoDict = fn.edgeInfoDict
    edgeIndexList = fn.edgeIndexList
    # print(G.edges(data=True))
    # nodeLabelDict = {node: G.node[node]['depth'] for node in G.nodes()} # nodeLevel
    # nodeLabelDict = {node: G.node[node]['nodeIndex'] for node in G.nodes()} # nodeIndex
    # nodeLabelDict = {node: np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()} # ground truth pressure in mmHg
    # nodeValueList = [G.node[node]['depth'] for node in G.nodes()] # nodeLevel
    # nodeValueList = [G.node[node]['nodeIndex'] for node in G.nodes()] # nodeIndex
    # nodeValueList = [np.round(nodeInfoDict[node]['simulationData']['pressure'] / 13560 / 9.8 * 1000, 1) for node in G.nodes()] # ground truth pressure in mmHg
    # edgeLabelDict = {edge: G[edge[0]][edge[1]]['depth'] for edge in G.edges()} # edgeLevel
    # edgeLabelDict = {edge: G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()} # edgeIndex
    # edgeLabelDict = {edge: np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()} # ground truth flow in cm^3/s
    # edgeValueList = [G[edge[0]][edge[1]]['depth'] for edge in G.edges()] # edgeLevel
    # edgeValueList = [G[edge[0]][edge[1]]['edgeIndex'] for edge in G.edges()] # edgeIndex
    # edgeValueList = [edgeInfoDict[edgeIndex]['meanRadius'] for edgeIndex in edgeIndexList] # meanRadius
    # edgeValueList = [np.round(edgeInfoDict[G[edge[0]][edge[1]]['edgeIndex']]['simulationData']['flow']*10**6, 2) for edge in G.edges()] # ground truth flow in cm^3/s
    # infoDict = {'nodeLabelDict': nodeLabelDict, 'nodeValueList': nodeValueList, 'nodeColorbarLabel': 'Node depth',
    #             'edgeLabelDict': edgeLabelDict, 'edgeValueList': edgeValueList, 'edgeColorbarLabel': 'Edge depth',
    #             'figTitle': 'Test'}
    # fn.plotNetwork(infoDict)

if __name__ == "__main__":
    seed = None
    # seed = 200 # generate the ground truth model using the artificial network
    np.random.seed(seed=seed)
    main()
