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

class Vessel(QtGui.QWidget):
    """
    A custom GUI framework based on QWidget
    """
    def __init__(self, app=None):
        super(Vessel, self).__init__()
        self.app = app
        self.init_ui()
        self.init_variable()
        self.qt_connections()

    def init_ui(self):
        pass
    
    def init_variable(self):
        pass

    def qt_connections(self):
        pass

class PlotObject(gl.GLViewWidget):
    """ 
    Override GLViewWidget with enhanced behavior
    This widget is based on the framework I found [here](https://groups.google.com/d/msg/pyqtgraph/mZiiLO8hS70/740KYx-vAAAJ), 
    which enables a user to select a point in 3D using pyqtgraph.
    """
    App = None

    def __init__(self, app=None):

        if self.App is None:
            if app is not None:
                self.App = app
            else:
                self.App = QtGui.QApplication([])
        super(PlotObject,self).__init__()
        self.skeletonNodesStartIndex = 0
        self.segmentStartIndex = 1
        self.offset = np.array([0, 0, 0])
        self.customInit()
    
    def customInit(self):
        pass

    def mousePressEvent(self, ev):
        """ 
        Store the position of the mouse press for later use.
        """
        super(PlotObject, self).mousePressEvent(ev)
        self._downpos = self.mousePos

    def mouseReleaseEvent(self, ev):
        """ 
        Allow for single click to move and right click for context menu.
        Also emits a sigUpdate to refresh listeners.
        """
        super(PlotObject, self).mouseReleaseEvent(ev)
        if self._downpos == ev.pos():
            x = ev.pos().x()
            y = ev.pos().y()
            if ev.button() == 2 :
                try:
                    self.mPosition()
                except Exception:
                    print(traceback.format_exc())
            elif ev.button() == 1:
                x = x - self.width() / 2
                y = y - self.height() / 2
                #self.pan(-x, -y, 0, relative=True)
        self._prev_zoom_pos = None
        self._prev_pan_pos = None

    def mPosition(self):
        #This function is called by a mouse event
        ## Get mouse coordinates saved when the mouse is clicked( incase dragging)
        mx = self._downpos.x()
        my = self._downpos.y()
        self.Candidates = [] #Initiate a list for storing indices of picked points
        #Get height and width of 2D Viewport space
        view_w = self.width()
        view_h = self.height()
        #Convert pixel values to normalized coordinates
        x = 2.0 * mx / view_w - 1.0
        y = 1.0 - (2.0 * my / view_h)
        # Convert projection and view matrix to np types and inverse them
        PMi = self.projectionMatrix().inverted()[0]
        VMi = self.viewMatrix().inverted()[0]
        ray_clip = QtGui.QVector4D(x, y, -1.0, 1.0) # get transpose for matrix multiplication
        ray_eye = PMi * ray_clip
        ray_eye.setZ(-1)
        ray_eye.setW(0)
        #Convert to world coordinates
        ray_world = VMi * ray_eye
        ray_world = QtGui.QVector3D(ray_world.x(), ray_world.y(), ray_world.z()) # get transpose for matrix multiplication
        ray_world.normalize()
        O = np.matrix(self.cameraPosition())  # camera position should be starting point of the ray
        ray_world = np.matrix([ray_world.x(), ray_world.y(), ray_world.z()])
        # print('O={}, ray_world={}'.format(O, ray_world))
        skeletonNodesStartIndex = self.skeletonNodesStartIndex
        skeletonNodesPlotItem = self.items[skeletonNodesStartIndex]
        skeletonColor = skeletonNodesPlotItem.color
        for i, C in enumerate(skeletonNodesPlotItem.pos): # Iterate over all points
            offset = self.offset
            CView = C + offset
            OC = O - CView
            b = np.inner(ray_world, OC)
            b = b.item(0)
            c = np.inner(OC, OC)
            c = c.item(0) - (0.4)**2   #np.square((self.Sizes[i]))
            bsqr = np.square(b)
            if (bsqr - c) >= 0: # means intersection
                self.currentVoxelIndex = i
                self.currentVoxel = tuple(C)
                stop = self.pointSelectionLogic()
                if stop:
                    break
    
    def pointSelectionLogic(self):
        pass
    
    def addExtraInfo(self, **kwds):
        for arg in kwds.keys():
            setattr(self, arg, kwds[arg])

class myVessel(Vessel):
    def init_ui(self):
        self.setWindowTitle('Vessel')
        hbox = QtGui.QHBoxLayout()
        self.setLayout(hbox)
        
        app = self.app
        self.plotwidget = myPlotObject(app=app)
        
        hbox.addWidget(self.plotwidget, 4)
        
        vbox = QtGui.QVBoxLayout()
        self.chosenVoxelsButtonGroup = QtGui.QButtonGroup()
        self.labelInitialVoxelsButton = QtGui.QPushButton("Label Initial Voxels")
        self.labelInitialVoxelsButton.setCheckable(True)
        self.labelBoundaryVoxelsButton = QtGui.QPushButton("Label Boundary Voxels")
        self.labelBoundaryVoxelsButton.setCheckable(True)
        self.chosenVoxelsButtonGroup.addButton(self.labelInitialVoxelsButton, 1)
        self.chosenVoxelsButtonGroup.addButton(self.labelBoundaryVoxelsButton, 2)
        self.partitionNamesButtonGroup = QtGui.QButtonGroup()
        self.LCAButton = QtGui.QPushButton("LCA")
        self.LCAButton.setCheckable(True)
        self.RCAButton = QtGui.QPushButton("RCA")
        self.RCAButton.setCheckable(True)
        self.PCAButton = QtGui.QPushButton("PCA")
        self.PCAButton.setCheckable(True)
        self.LACAButton = QtGui.QPushButton("LACA")
        self.LACAButton.setCheckable(True)
        self.RACAButton = QtGui.QPushButton("RACA")
        self.RACAButton.setCheckable(True)
        self.partitionNamesButtonGroup.addButton(self.LCAButton, 11)
        self.partitionNamesButtonGroup.addButton(self.RCAButton, 12)
        self.partitionNamesButtonGroup.addButton(self.PCAButton, 13)
        self.partitionNamesButtonGroup.addButton(self.LACAButton, 14)
        self.partitionNamesButtonGroup.addButton(self.RACAButton, 15)
        self.loadChosenVoselsButton = QtGui.QPushButton("Load Chosen Voxels")
        self.saveChosenVoselsButton = QtGui.QPushButton("Save Chosen Voxels")
        self.clearChosenVoselsButton = QtGui.QPushButton("Clear Chosen Voxels")
        self.showPartitionsButton = QtGui.QPushButton("Show Partitions")
        self.randomWalkBFSButton = QtGui.QPushButton("Random Walk BFS")
        self.loadSegmentNodeInfoDictButton = QtGui.QPushButton("Load segment/node InfoDict")
        self.showNodeButton = QtGui.QPushButton("Show Node")
        self.performFluidSimulationButton = QtGui.QPushButton("Fluid Simulation")
        self.loadFluidResultButton = QtGui.QPushButton("Load Fluid Result")
        self.showPressureResultButton = QtGui.QPushButton("Show Pressure Result")
        self.showVelocityResultButton = QtGui.QPushButton("Show Velocity Result")
        self.showQuantityButton = QtGui.QPushButton("Show Quantity")
        self.applyPressureVelocityDistributionButton = QtGui.QPushButton("Apply Pressure/Velocity Distribution")
        self.showSegmentButton = QtGui.QPushButton("Show Segment")
        self.segmentIndexBox = QtGui.QLineEdit()

        vbox.addWidget(self.labelInitialVoxelsButton, 1)
        vbox.addWidget(self.labelBoundaryVoxelsButton, 1)
        vbox.addWidget(self.LCAButton, 1)
        vbox.addWidget(self.RCAButton, 1)
        vbox.addWidget(self.PCAButton, 1)
        vbox.addWidget(self.LACAButton, 1)
        vbox.addWidget(self.RACAButton, 1)
        vbox.addWidget(self.loadChosenVoselsButton, 1)
        vbox.addWidget(self.saveChosenVoselsButton, 1)
        vbox.addWidget(self.clearChosenVoselsButton, 1)
        vbox.addWidget(self.randomWalkBFSButton, 1)
        vbox.addWidget(self.showPartitionsButton, 1)
        vbox.addWidget(self.loadSegmentNodeInfoDictButton, 1)
        vbox.addWidget(self.showNodeButton, 1)
        vbox.addWidget(self.performFluidSimulationButton, 1)
        vbox.addWidget(self.loadFluidResultButton, 1)
        vbox.addWidget(self.showPressureResultButton, 1)
        vbox.addWidget(self.showVelocityResultButton, 1)
        vbox.addWidget(self.showQuantityButton, 1)
        vbox.addWidget(self.applyPressureVelocityDistributionButton, 1)
        vbox.addWidget(self.showSegmentButton, 1)
        vbox.addWidget(self.segmentIndexBox, 1)
        vbox.addStretch(1)
        hbox.addLayout(vbox, 1)

        self.setGeometry(30, 30, 1500, 900)
        self.show()
    
    def init_variable(self):
        self.labelInitialVoxelsButtonClicked = False
        self.labelBoundaryVoxelsButtonClicked = False
        self.LCAButtonClicked = False
        self.RCAButtonClicked = False
        self.PCAButtonClicked = False
        self.LACAButtonClicked = False
        self.RACAButtonClicked = False
        self.chosenVoxels = {}
        self.directory = ''
        self.buttonIDMap = {-1: 'unused', 1: 'initialVoxels', 2: 'boundaryVoxels', 11: 'LCA', 12: 'RCA', 13: 'PCA', 14: 'LACA', 15: 'RACA'}
        
    
    def qt_connections(self):
        self.loadChosenVoselsButton.clicked.connect(self.onLoadChosenVoxelsButtonClicked)
        self.saveChosenVoselsButton.clicked.connect(self.onSaveChosenVoxelsButtonClicked)
        self.clearChosenVoselsButton.clicked.connect(self.onClearChosenVoxelsButtonClicked)
        self.showPartitionsButton.clicked.connect(self.onShowPartitionsButtonClicked)
        self.randomWalkBFSButton.clicked.connect(self.onRandomWalkBFSButtonClicked)
        self.loadSegmentNodeInfoDictButton.clicked.connect(self.onLoadSegmentNodeInfoDictButtonClicked)
        self.showNodeButton.clicked.connect(self.onShowNodeButtonClicked)
        self.performFluidSimulationButton.clicked.connect(self.onPerformFluidSimulationButtonClicked)
        self.loadFluidResultButton.clicked.connect(self.onLoadFluidResultButtonClicked)
        self.showPressureResultButton.clicked.connect(self.onShowPressureResultButtonClicked)
        self.showVelocityResultButton.clicked.connect(self.onShowVelocityResultButtonClicked)
        self.showQuantityButton.clicked.connect(self.onShowQuantityButtonClicked)
        self.applyPressureVelocityDistributionButton.clicked.connect(self.onApplyPressureVelocityDistributionButtonClicked)
        self.showSegmentButton.clicked.connect(self.onShowSegmentButtonClicked)
    
    def addExtraInfo(self, **kwds):
        for arg in kwds.keys():
            setattr(self, arg, kwds[arg])

    def onLoadChosenVoxelsButtonClicked(self):
        directory = self.directory
        chosenVoxelsForPartitionFileName = 'chosenVoxelsForPartition.pkl'
        chosenVoxelsForPartitionFilePath = os.path.join(directory, chosenVoxelsForPartitionFileName)
        if os.path.exists(chosenVoxelsForPartitionFilePath):
            with open(chosenVoxelsForPartitionFilePath, 'rb') as f:
                self.plotwidget.chosenVoxels = pickle.load(f)
                print('{} loaded from {}.'.format(chosenVoxelsForPartitionFileName, chosenVoxelsForPartitionFilePath))
        else:
            print('{} does not exist at {}.'.format(chosenVoxelsForPartitionFileName, chosenVoxelsForPartitionFilePath))

    def onSaveChosenVoxelsButtonClicked(self):
        directory = self.directory
        chosenVoxels = self.plotwidget.chosenVoxels
        partitionInfo = self.plotwidget.partitionInfo
        G = self.plotwidget.G

        chosenVoxelsForPartitionFileName = 'chosenVoxelsForPartition.pkl'
        chosenVoxelsForPartitionFilePath = os.path.join(directory, chosenVoxelsForPartitionFileName)
        with open(chosenVoxelsForPartitionFilePath, 'wb') as f:
            pickle.dump(chosenVoxels, f, 2)  
            print('{} saved to {}.'.format(chosenVoxelsForPartitionFileName, chosenVoxelsForPartitionFilePath)) 
        
        partitionInfoFileName = 'partitionInfo.pkl'
        partitionInfoFilePath = os.path.join(directory, partitionInfoFileName)
        with open(partitionInfoFilePath, 'wb') as f:
            pickle.dump(partitionInfo, f, 2)  
            print('{} saved to {}.'.format(partitionInfoFileName, partitionInfoFilePath)) 
        
        graphCleanedWithAdvancedInfoFileName = 'graphRepresentationCleanedWithAdvancedInfo.graphml'
        graphCleanedWithAdvancedInfoFilePath = os.path.join(directory, graphCleanedWithAdvancedInfoFileName)
        nx.write_graphml(G, graphCleanedWithAdvancedInfoFilePath)
        print('{} saved to {}.'.format(graphCleanedWithAdvancedInfoFileName, graphCleanedWithAdvancedInfoFilePath))

    
    def onClearChosenVoxelsButtonClicked(self):
        self.plotwidget.clearChosenList()
    
    def onRandomWalkBFSButtonClicked(self, chosenPartitionName=None):
        # print(chosenPartitionName)
        if chosenPartitionName is None or chosenPartitionName not in self.plotwidget.partitionNames:
            chosenPartitionName = self.buttonIDMap[self.partitionNamesButtonGroup.checkedId()]
            # print(self.partitionNamesButtonGroup.checkedId(), chosenPartitionName)

        initialVoxels = self.plotwidget.chosenVoxels[chosenPartitionName]['initialVoxels']
        boundaryVoxels = self.plotwidget.chosenVoxels[chosenPartitionName]['boundaryVoxels']
        # voxelSegmentIndexArray = self.plotwidget.voxelSegmentIndexArray
        
        G = self.plotwidget.G
        G, visitedVoxels, segmentIndexList = mf.randomWalkBFS(G, initialVoxels, boundaryVoxels)
        tempDict = {voxel: chosenPartitionName for voxel in visitedVoxels}
        nx.set_node_attributes(G, tempDict, 'partitionName')
        self.plotwidget.G = G
        # depthLevelList = [self.plotwidget.G.node[node]['depthLevel'] for node in visitedVoxels if 'depthLevel' in self.plotwidget.G.node[node]]
        # print('Max depthLevel is {}'.format(np.max(depthLevelList)))
        # pathDistanceList = [self.plotwidget.G.node[node]['pathDistance'] for node in visitedVoxels if 'pathDistance' in self.plotwidget.G.node[node]]
        # print('Max pathDistance is {}'.format(np.max(pathDistanceList)))
        # print(self.plotwidget.chosenVoxels)
        if len(visitedVoxels) != 0:
            self.plotwidget.partitionInfo[chosenPartitionName] = {}
            self.plotwidget.partitionInfo[chosenPartitionName]['visitedVoxels'] = visitedVoxels
            # segmentIndexList = self.plotwidget.getPartitionSegments()
            segmentIndexList = list(np.unique(segmentIndexList)) # in case there is duplicate...
            self.plotwidget.partitionInfo[chosenPartitionName]['segmentIndexList'] = segmentIndexList
            # add segment level (based on depthLevel of voxels) to each segment
            for segmentIndex in segmentIndexList:
                segment = self.plotwidget.segmentList[segmentIndex]
                depthLevelList = [self.plotwidget.G.node[segment[ii]]['depthLevel'] for ii in range(len(segment)) if 'depthLevel' in self.plotwidget.G.node[segment[ii]]]
                self.plotwidget.G.add_path(segment, segmentLevel=int(np.min(depthLevelList)), partitionName=chosenPartitionName)
            self.plotwidget.showVoxelsVisited(visitedVoxels)
        else:
            print('No voxels visited')
        
        # aa = [self.plotwidget.G.node[voxel]['pathDistance'] for voxel in visitedVoxels]
        # print(aa)
    
    def onShowPartitionsButtonClicked(self):
        colorPools = [pg.glColor('r'), pg.glColor('g'), pg.glColor('b'), pg.glColor('y'), pg.glColor('c')]
        colorPools = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
        ii = 0
        for chosenPartitionName in self.plotwidget.partitionInfo.keys():
            visitedVoxels = self.plotwidget.partitionInfo[chosenPartitionName]['visitedVoxels']
            color = colorPools[ii]
            self.plotwidget.showVoxelsVisited(visitedVoxels, color=color)
            ii += 1

    def onLoadSegmentNodeInfoDictButtonClicked(self):
        directory = self.directory
        nodeInfoDictFileName = 'nodeInfoDict.pkl'
        nodeInfoDictFilePath = os.path.join(directory, nodeInfoDictFileName)
        with open(nodeInfoDictFilePath, 'rb') as f:
            self.plotwidget.nodeInfoDict = pickle.load(f)
            print('{} loaded from {}.'.format(nodeInfoDictFileName, directory))

        segmentInfoDictFileName = 'segmentInfoDict.pkl'
        segmentInfoDictFilePath = os.path.join(directory, segmentInfoDictFileName)
        with open(segmentInfoDictFilePath, 'rb') as f:
            self.plotwidget.segmentInfoDict = pickle.load(f)
            print('{} loaded from {}.'.format(segmentInfoDictFileName, directory))
    
    def onShowNodeButtonClicked(self):
        self.plotwidget.showNode()
    
    def onPerformFluidSimulationButtonClicked(self):
        self.plotwidget.performFluidSimulation()
    
    def onLoadFluidResultButtonClicked(self):
        directory = self.directory
        # print('c' in locals(), 'c' in globals())
        if 'c' in globals() and 'k' in globals() and c != 0 and k != 0:
            filename = 'fluidResult(c={}, k={}).pkl'.format(c, k)
        else:
            filename = 'fluidResult.pkl'

        if os.path.exists(directory + filename):
            with open(directory + filename, 'rb') as f:
                self.plotwidget.fluidResult = pickle.load(f)
                print('{} loaded'.format(filename))
        else:
            print('{} does not exist'.format(filename))
        
        self.plotwidget.updateFluidVolume()
        print('Fluid volume updated')
    
    def onShowPressureResultButtonClicked(self):
        self.plotwidget.showFluidResult('Pressure')
    
    def onShowVelocityResultButtonClicked(self):
        self.plotwidget.showFluidResult('Velocity')
    
    def onShowQuantityButtonClicked(self):
        if 'showQuantity' in globals():
            self.plotwidget.showQuantity(showQuantity)
    
    def onApplyPressureVelocityDistributionButtonClicked(self):
        self.plotwidget.applyPressureVelocityDistribution()
    
    def onShowSegmentButtonClicked(self):
        segmentIndexInput = int(self.segmentIndexBox.text())
        segmentPlotItem = self.plotwidget.items[self.plotwidget.segmentStartIndex + segmentIndexInput]
        segmentPlotItem.setData(color=pg.glColor('b'))
        
  
class myPlotObject(PlotObject):
    def customInit(self):
        self.directory = ''
        self.skeletonNodesStartIndex = 0
        self.segmentStartIndex = 1
        self.chosenVoxels = {}
        self.G = nx.Graph()
        self.shape = (10, 10, 10)
        self.voxelIndexArray = np.full(self.shape, 0)
        self.voxelSegmentIndexArray = np.full(self.shape, 0)
        self.partitionInfo = {}
        self.segmentInfoDict = {}
        self.nodeInfoDict = {}
        self.segmentList = []
        self.fluidResult = {}
        self.pressureVolume = np.array([]) # volume of the same size as the data volume that contains pressure from the fluid simulation
        self.velocityVolume = np.array([]) # volume of the same size as the data volume that contains velocity from the fluid simulation
        self.partitionNames = ['LCA', 'RCA', 'PCA', 'LACA', 'RACA']
        for partitionName in self.partitionNames:
            self.chosenVoxels[partitionName] = {}
            for voxelCategory in ['initialVoxels', 'boundaryVoxels']:
                self.chosenVoxels[partitionName][voxelCategory] = []
        
        if platform.system() == 'Windows':
            self._voxelNormalSize = 5
            self._voxelChosenSize = 8
            self._segmentNormalWidth = 3
        elif platform.system() == 'Darwin': # Mac
            self._voxelNormalSize = 7
            self._voxelChosenSize = 10
            self._segmentNormalWidth = 3
    
    def clearChosenList(self):
        skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonPlotItemColor = skeletonPlotItem.color        
        
        chosenVoxelsType = self.parent().buttonIDMap[self.parent().chosenVoxelsButtonGroup.checkedId()]
        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        if chosenVoxelsType == 'initialVoxels':
            self.chosenVoxels[chosenPartitionName][chosenVoxelsType] = []
        elif chosenVoxelsType == 'boundaryVoxels':
            boundaryVoxels = self.chosenVoxels[chosenPartitionName][chosenVoxelsType]
            self.chosenVoxels[chosenPartitionName][chosenVoxelsType] = []
            # manually reset previous boundary points to white
            if len(boundaryVoxels) != 0:
                boundaryVoxelsArray = np.array(boundaryVoxels, dtype=np.int16)
                skeletonPlotItemColor[self.voxelIndexArray[tuple(boundaryVoxelsArray.T)], :] = [1, 1, 1, 1] # reset to white
        else:
            print('Choose either initial points or boundary points!')
            return 
        
        if chosenPartitionName != 'unused':
            visitedVoxels = self.partitionInfo[chosenPartitionName]['visitedVoxels'] # obtain voxels in this partition
            if len(visitedVoxels) != 0: 
                for voxel in visitedVoxels:
                    del self.G.node[voxel]['depthVoxel']
                    del self.G.node[voxel]['depthLevel']
                
                visitedVoxelsArray = np.array(visitedVoxels, dtype=np.int16)
                skeletonPlotItemColor[self.voxelIndexArray[tuple(visitedVoxelsArray.T)], :] = [1, 1, 1, 1] # reset to white
                skeletonPlotItem.setData(color=skeletonPlotItemColor)
                self.partitionInfo[chosenPartitionName]['visitedVoxels'] = []
                self.partitionInfo[chosenPartitionName]['segmentIndexList'] = []
            else:
                print('No more voxels in this partition!')
        else:
            pass
    
    def showVoxelsVisited(self, visitedVoxels, color=None):
        skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonPlotItemColor = skeletonPlotItem.color
        skeletonPlotItemSize = skeletonPlotItem.size
        visitedVoxelsArray = np.array(visitedVoxels, dtype=np.int16)
        # depthVoxelList = [self.G.node[voxel]['depthVoxel'] for voxel in visitedVoxels]
        # colormap = mf.generateColormap(depthVoxelList)
        # depthLevelList = [self.G.node[voxel]['depthLevel'] for voxel in visitedVoxels]
        # colormap = mf.generateColormap(depthLevelList)
        # print(depthLevelList)
        # print(colormap)
        # skeletonPlotItemColor[self.voxelIndexArray[tuple(visitedVoxelsArray.T)], :] = colormap
        # if visitedVoxels is an empty list, the following line will color all the voxels yellow!
        if color is None:
            color = [1, 1, 0, 1]

        skeletonPlotItemColor[self.voxelIndexArray[tuple(visitedVoxelsArray.T)], :] = color # yellow
        skeletonPlotItemSize[:] = self._voxelNormalSize
        skeletonPlotItem.setData(size=skeletonPlotItemSize, color=skeletonPlotItemColor)
        # self.getPartitionSegments()
    
    def getPartitionSegments(self):
        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        voxelsVisited = self.partitionInfo[chosenPartitionName]['visitedVoxels']
        segmentIndexList = [self.voxelSegmentIndexArray[voxel] for voxel in voxelsVisited if self.G.degree(voxel) < 3]
        segmentIndexList = np.unique(segmentIndexList)
        # self.partitionInfo[chosenPartitionName]['segmentIndexList'] = segmentIndexList

        # for segmentIndex in segmentIndexList:
        #     segmentPlotItem = self.items[self.segmentStartIndex + segmentIndex]
        #     segmentPlotItem.setData(color=pg.glColor('r'))

        return segmentIndexList
    
    def showNode(self):
        nodeInfoDict = self.nodeInfoDict
        nodesToShow = []
        for node, nodeInfo in nodeInfoDict.items():
            if 'localBifurcationAmplitude' in nodeInfo:
                nodesToShow.append([node, nodeInfo['localBifurcationAmplitude']])
        
        nodesCoords = np.array([node for node, localBifurcationAmplitude in nodesToShow if localBifurcationAmplitude >= 90], dtype=np.int16)
        # print('nodesCoords length = {}'.format(len(nodesCoords)))
        skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonPlotItemColor = skeletonPlotItem.color
        skeletonPlotItemSize = skeletonPlotItem.size
        skeletonPlotItemColor[self.voxelIndexArray[tuple(nodesCoords.T)], :] = [1, 0, 0, 1] # red
        skeletonPlotItemSize[self.voxelIndexArray[tuple(nodesCoords.T)]] = self._voxelChosenSize
        skeletonPlotItem.setData(size=skeletonPlotItemSize, color=skeletonPlotItemColor)
    
    def performFluidSimulation(self):
        G = self.G
        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        entryPoints = self.chosenVoxels[chosenPartitionName]['initialVoxels']
        allVoxels = self.partitionInfo[chosenPartitionName]['visitedVoxels']
        segmentList = self.segmentList
        segmentIndexList = self.partitionInfo[chosenPartitionName]['segmentIndexList']
        if chosenPartitionName == 'PCA':
            # boundaryCondition = [15998, 0, 2]
            boundaryCondition = {'pressureIn': 15946} # Pascal
        elif chosenPartitionName == 'LACA' or chosenPartitionName == 'RACA':
            # boundaryCondition = [15998, 0, 1.5]
            boundaryCondition = {'pressureIn': 15946}
        else:
            # boundaryCondition = [15998, 0, 0.3]
            boundaryCondition = {'pressureIn': 15946}

        ##
        directory = self.directory
        with open(os.path.join(directory, 'segmentInfoDict.pkl'), 'rb') as f:
            segmentInfoDict = pickle.load(f)
        with open(os.path.join(directory, 'nodeInfoDict.pkl'), 'rb') as f:
            nodeInfoDict = pickle.load(f)
        # with open(directory + 'partitionInfo.pkl', 'rb') as f:
        #     partitionInfo = pickle.load(f)
        # with open(directory + 'chosenVoxelsForPartition.pkl', 'rb') as f:
        #     chosenVoxels = pickle.load(f)
        
        ADANFolder = directory + '../../../ADAN-Web/'
        fileName = 'resultADANDict.pkl'
        with open(os.path.join(ADANFolder, fileName), 'rb') as f:
            resultADANDict = pickle.load(f)
            print('{} loaded from {}'.format(fileName, ADANFolder))

        # pressureArray, velocityArray, result, GIndex = mf.fluidSimulation(G, entryPoints, allVoxels, segmentList, segmentIndexList, boundaryCondition, fluidMethod='HW', showResult='Pressure')
        pressureArray, velocityArray, result, GIndex, eqnInfoDictList = mf.fluidSimulation4(G, entryPoints, allVoxels, segmentList, segmentIndexList, segmentInfoDict, nodeInfoDict, boundaryCondition, resultADANDict, fluidMethod='HW')
        self.fluidResult[chosenPartitionName] = {}
        self.fluidResult[chosenPartitionName]['pressureArray'] = pressureArray
        self.fluidResult[chosenPartitionName]['velocityArray'] = velocityArray
        self.fluidResult[chosenPartitionName]['result'] = result
        self.fluidResult[chosenPartitionName]['GIndex'] = GIndex
        self.updateFluidVolume(chosenPartitionName)
        
    def updateFluidVolume(self, chosenPartitionName=None):
        if self.pressureVolume.shape != self.shape:
            self.pressureVolume = np.full(self.shape, 0, dtype=np.float)
            self.velocityVolume = np.full(self.shape, 0, dtype=np.float)
        
        if chosenPartitionName is None:
            for partitionName, info in self.fluidResult.items():
                pressureArray = info['pressureArray']
                velocityArray = info['velocityArray']
                for row in pressureArray:
                    self.pressureVolume[tuple(row[:3].astype(np.int16))] = row[3]
                
                for row in velocityArray:
                    self.velocityVolume[tuple(row[:3].astype(np.int16))] = row[3]
        else:
            pressureArray = self.fluidResult[chosenPartitionName]['pressureArray']
            velocityArray = self.fluidResult[chosenPartitionName]['velocityArray']
            for row in pressureArray:
                self.pressureVolume[tuple(row[:3].astype(np.int16))] = row[3]
                
            for row in velocityArray:
                self.velocityVolume[tuple(row[:3].astype(np.int16))] = row[3]
    
    def showFluidResult(self, quantity):
        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        if quantity == 'Pressure':
            dataArray = self.fluidResult[chosenPartitionName]['pressureArray']
        elif quantity == 'Velocity':
            dataArray = self.fluidResult[chosenPartitionName]['velocityArray']
        
        voxelCoords = dataArray[:, :3].astype(np.int16)
        values = dataArray[:, 3]

        # chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        # voxelsVisited = self.partitionInfo[chosenPartitionName]['visitedVoxels']
        # voxelCoords = np.array(voxelsVisited, dtype=np.int16)
        # values = [self.G.node[node]['depthVoxel'] for node in voxelsVisited]
        
        skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonPlotItemColor = skeletonPlotItem.color
        skeletonPlotItemSize = skeletonPlotItem.size
        color = mf.generateColormap(values)
        skeletonPlotItemColor.dtype = np.float
        skeletonPlotItemColor[self.voxelIndexArray[tuple(voxelCoords.T)], :] = color
        skeletonPlotItemColor[:, 3] = 1
        skeletonPlotItemSize[self.voxelIndexArray[tuple(voxelCoords.T)]] = self._voxelChosenSize
        skeletonPlotItem.setData(size=skeletonPlotItemSize, color=skeletonPlotItemColor)
    
    def loadChosenVoxels(self):
        self.parent().onLoadChosenVoxelsButtonClicked()
        for chosenPartitionName in self.partitionNames:
            self.parent().onRandomWalkBFSButtonClicked(chosenPartitionName=chosenPartitionName)

    def applyPressureVelocityDistribution(self):
        if self.segmentInfoDict == {} or self.nodeInfoDict == {}:
            self.parent().onLoadSegmentNodeInfoDictButtonClicked()
            self.loadChosenVoxels()

        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        if chosenPartitionName == 'unused':
            print('Choose a partition!')
        else:
            segmentInfoDict, nodeInfoDict = assignPressureVelocityDistribution(self.G, self.partitionInfo, self.segmentInfoDict, self.nodeInfoDict, self.segmentList, chosenPartitionName, option=2)
            self.segmentInfoDict = segmentInfoDict
            self.nodeInfoDict = nodeInfoDict
            segmentIndexList = self.partitionInfo[chosenPartitionName]['segmentIndexList']
            # for segmentIndex in segmentIndexList:
            #     if 'velocityRatio' not in segmentInfoDict[segmentIndex]:
            #         print('velocityRatio does not exist in segmentIndex={}'.format(segmentIndex))
            k = 1.852
            ckResult, terminalPressureResult, segmentInfoDict, nodeInfoDict = calculatePressureVelocity(self.G, self.chosenVoxels, self.partitionInfo, self.segmentInfoDict, self.nodeInfoDict, self.segmentList, chosenPartitionName, k=k, c=None)
            self.segmentInfoDict = segmentInfoDict
            self.nodeInfoDict = nodeInfoDict
            # for segmentIndex in segmentIndexList:
            #     if 'velocity' not in segmentInfoDict[segmentIndex]:
            #         print('velocity does not exist in segmentIndex={}'.format(segmentIndex))
            pressureArray, velocityArray = self.generatePressureVelocityArray(self.segmentList, chosenPartitionName)
            if chosenPartitionName not in self.fluidResult:
                self.fluidResult[chosenPartitionName] = {}
            
            self.fluidResult[chosenPartitionName]['pressureArray'] = pressureArray
            self.fluidResult[chosenPartitionName]['velocityArray'] = velocityArray
            self.updateFluidVolume(chosenPartitionName)
    
    def generatePressureVelocityArray(self, segmentList, chosenPartitionName):
        segmentIndexList = self.partitionInfo[chosenPartitionName]['segmentIndexList']
        pressureArray = np.array([]).reshape(-1, 4)
        for segmentIndex in segmentIndexList:
            segment = segmentList[segmentIndex]
            segmentCoords = np.array(segment)
            l = len(segment)
            headPressure = self.nodeInfoDict[segment[0]]['pressure']
            tailPressure = self.nodeInfoDict[segment[-1]]['pressure']
            pressures = np.linspace(headPressure, tailPressure, num=l)
            pressureArraySegment = np.hstack((segmentCoords, pressures.reshape(l, 1)))
            pressureArray = np.vstack((pressureArray, pressureArraySegment))
        
        # velocity result
        velocityArray = np.array([]).reshape(-1, 4)
        for segmentIndex in segmentIndexList:
            segment = segmentList[segmentIndex]
            segmentCoords = np.array(segment)
            l = len(segment)
            # if 'velocity' not in self.segmentInfoDict[segmentIndex]:
            #     print(segmentIndex, self.G.degree(self.segmentList[segmentIndex][0]), self.G.degree(self.segmentList[segmentIndex][0]))
            #     continue
            velocity = self.segmentInfoDict[segmentIndex]['velocity']
            velocities = np.full((l, 1), velocity)
            velocityArraySegment = np.hstack((segmentCoords, velocities))
            velocityArray = np.vstack((velocityArray, velocityArraySegment))
        
        return pressureArray, velocityArray
   
    def showQuantity(self, showQuantity):
        values = [info[showQuantity] for node, info in self.G.nodes(data=True) if showQuantity in info]
        if len(values) != 0:
            skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
            # skeletonPlotItemColor = skeletonPlotItem.color
            color = mf.generateColormap(values).astype(np.float)
            skeletonPlotItem.setData(color=color)
        
    def pointSelectionLogic(self):
        currentVoxelIndex = self.currentVoxelIndex
        currentVoxel = self.currentVoxel
        skeletonPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonPlotItemColor = skeletonPlotItem.color
        skeletonPlotItemSize = skeletonPlotItem.size

        chosenVoxelsType = self.parent().buttonIDMap[self.parent().chosenVoxelsButtonGroup.checkedId()]
        chosenPartitionName = self.parent().buttonIDMap[self.parent().partitionNamesButtonGroup.checkedId()]
        if chosenVoxelsType == 'unused' or chosenPartitionName == 'unused':# or self.pressureVolume.shape == self.shape:
            pressuremmHg = (self.pressureVolume[tuple(currentVoxel)] - 101000) / (13560*9.8) * 1000
            velocity = self.velocityVolume[tuple(currentVoxel)]
            radiusVoxel = self.G.node[currentVoxel]['radius']
            neighbors = list(self.G.neighbors(currentVoxel))
            meanRadiusVoxel = self.G[currentVoxel][neighbors[0]]['meanRadius']
            print('Current voxel: {}, pressure = {:.3f} mmHg, velocity = {:.3f} m/s, radius(Voxel) = {:.3f}, meanRadius(Voxel) = {:.3f}'.format(currentVoxel, pressuremmHg, velocity, radiusVoxel, meanRadiusVoxel))
            # print('Current voxel: {}'.format(currentVoxel))
            stop = True
            return stop

        chosenVoxelsList = self.chosenVoxels[chosenPartitionName][chosenVoxelsType]
        if chosenVoxelsType == 'initialVoxels':
            if currentVoxel in chosenVoxelsList:
                chosenVoxelsList.remove(currentVoxel)
                skeletonPlotItemColor[currentVoxelIndex, :] = [1, 1, 1, 1]
                skeletonPlotItemSize[currentVoxelIndex] = self._voxelNormalSize
            else:
                chosenVoxelsList.append(currentVoxel)
                skeletonPlotItemColor[currentVoxelIndex, :] = [0, 0, 1, 1] # blue
                skeletonPlotItemSize[currentVoxelIndex] = self._voxelChosenSize
        
        elif chosenVoxelsType == 'boundaryVoxels':
            if currentVoxel in chosenVoxelsList:
                chosenVoxelsList.remove(currentVoxel)
                skeletonPlotItemColor[currentVoxelIndex, :] = [1, 1, 1, 1]
                skeletonPlotItemSize[currentVoxelIndex] = self._voxelNormalSize
            else:
                chosenVoxelsList.append(currentVoxel)
                skeletonPlotItemColor[currentVoxelIndex, :] = [1, 0, 0, 1] # red
                skeletonPlotItemSize[currentVoxelIndex] = self._voxelChosenSize
        
        skeletonPlotItem.setData(size=skeletonPlotItemSize, color=skeletonPlotItemColor)
        stop = True
        return stop