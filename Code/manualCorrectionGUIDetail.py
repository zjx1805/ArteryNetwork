import sys, os
from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy import interpolate
from numpy.linalg import norm
import traceback
import networkx as nx
import pickle

class PlotObject(gl.GLViewWidget):
    """ 
    This is the customized Qt widget in the GUI and it displays 3D vessel volume (semi-transparent, white) as well as
    centerpoints and connections. You can rotate (left button), move (middle button) and select points (right button).
    There are currently three operations supported: remove, reconnect and grow. Different operations require selecting
    different points: remove needs only one, reconnect needs four, grow needs four. Normal segments are shown in red,
    segments that are part of a loop are shown in blue, erroneous segments are shown in cyan, and removed segments shown
    in green. Note that all segments in the `segmentList` should be simple branches, i.e., they do not contain
    bifurcation unless at the two ends. After each operation, they should remain simple branches, and thus
    merging/splitting operation may be applied to certain affected segments to ensure that they remain simple branches.
    Also after each operation, a function `checkCycle` will be run to check whether there are any loops in the graph,
    and color those segments that are part of a loop in blue. 
    
    The GUI also supports undo and redo operations. The information of each operation is saved in a dictionary called
    `event` and the function `processEvents` will parse the event info and perform the corresponding actions. In this
    way, an operation could be undo by sending the event to the function `reverseEvents`, which will perform exactly the
    opposite operations as `processEvents`. Each event, if executed successfully, will be appended to `eventList` and
    could be saved by the user. At a later time, the user can choose to load the `eventList` into the GUI to recover the
    previous progress and move on (just like the save&load functions in Word).

    To remove a segment, first press the `Remove segment` button on the right, then right click on any centerpoint of
    this segment (cannot be two endpoints), and the selected segment will become green. Meanwhile, if either of the two
    ends are connected to other segments, then those segments will be merged to form a simple branch. The segments to be 
    merged will be marked as removed (but they remain in the `segmentList`) and a new segment will be created and added 
    to `segmentList`. This is to ensure that when the operation is reversed, the original segments can be recovered. Note that it
    currently cannot remove a segment with only two centerpoints, because of the aforementioned limit. **This will be
    fixed in the future.**

    `Reconnect segments` requires choosing four points on two segments (two points for each). For example, if we have
    two segments as follows:  
    [a1, a2,..., a10, a11,..., a20], [b1, b2,..., b10, b11,..., b20]  
    And assume you sequentially selects a10, a11, b10, b11 (note that the order matters!), then these two segments will
    be merged in the following way:  
    [a1, a2,..., a10, a11, [...], b10, b11,..., b20]  
    The [...] part in the middle will be extrapolated by fitting the remaining parts (a1 to a10 and b10 to b20) via a
    spline interpolation and then attached to the voxel grid. And similarly, if a20 or b1 is connected to other
    segments, they will be merged accordingly. The affected segments will be marked as removed and new segments will be
    created as before.

    'Grow segments' is similar to `Reconnect segments` except that it only extends to first segment until it reaches b10. 
    The second segment will be split at b10 into two new segments and the original one will be marked as removed as usual.

    This widget is based on the framework I found [here](https://groups.google.com/d/msg/pyqtgraph/mZiiLO8hS70/740KYx-vAAAJ), 
    which enables a user to select a point in 3D using pyqtgraph.


    Override GLViewWidget with enhanced behavior (Credit: https://groups.google.com/d/msg/pyqtgraph/mZiiLO8hS70/740KYx-vAAAJ)

    """
    App = None

    def __init__(self, app=None):

        if self.App is None:
            if app is not None:
                self.App = app
            else:
                self.App = QtGui.QApplication([])
        super(PlotObject,self).__init__()
        self.Poss = np.array([])
        self.removeList = []
        self.skeletonNodesStartIndex = 0
        self.segmentStartIndex = 1
        self.cycle2SegmentDict = {}
        self.segment2CycleDict = {}
        self.chosenVoxelsList = []
        self.eventList = []
        self.G = nx.Graph()
        self.segmentIndexUsed = []
        self.onLoading = True

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
                print('center={}'.format(self.opts['center']))
                print('x={}, y={}'.format(x,y))
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
                currentSkeletonColor = skeletonColor[i, :]
                indexVolume = self.indexVolume
                affineTransform = self.affineTransform
                offset = self.offset
                trueCoord = C
                segmentIndex = indexVolume[tuple(trueCoord)]
                
                # Create empty event
                event = {'type': None, 'nodeIndex': i, 'trueCoord': trueCoord, 'segmentIndex': segmentIndex}
                success, event = self.processEvent(event)
                print('Segment {} selected, coord = {}'.format(segmentIndex, trueCoord))

                neighbours = list(self.G.neighbors(tuple(trueCoord)))
                segmentIndexListFromGraph = [self.G[tuple(trueCoord)][neighbour]['segmentIndex'] for neighbour in neighbours]
                segmentIndexUniqueFromGraph = list(np.unique(segmentIndexListFromGraph))
                if len(segmentIndexUniqueFromGraph) == 1:
                    if segmentIndex == segmentIndexUniqueFromGraph[0]:
                        pass
                    else:
                        print('segmentIndexFromIndexVolume={}, segmentIndexFromGraph={}'.format(segmentIndex, segmentIndexUniqueFromGraph[0]))
                else:
                    print('segmentIndexFromIndexVolume={}, segmentIndexFromGraph={}'.format(segmentIndex, segmentIndexUniqueFromGraph))
                
                skeletonCoords = self.items[self.segmentStartIndex + segmentIndex].pos.tolist()
                skeletonCoordsTuple = list(map(tuple, skeletonCoords))
                if tuple(trueCoord) not in skeletonCoordsTuple:
                    print('Selected voxel not in corresponding GLLinePlotItem')
                
                segment = self.segmentList[segmentIndex]
                if tuple(trueCoord) not in segment:
                    print('Selected voxel not in corresponding segmentList')
                
                if segment != skeletonCoordsTuple:
                    shorterLength = np.min([len(segment), len(skeletonCoordsTuple)])
                    for ii in range(shorterLength):
                        if segment[ii] != skeletonCoordsTuple[ii]:
                            break
                    print('segment in GLLinePlotItem and segmentList does not match (differ after {}-th point)'.format(ii))
                    print(segment)
                    print(skeletonCoordsTuple)
                
                if self.parent().removeSegmentButtonClicked:
                    event['type'] = 'remove'
                    try:
                        success, event = self.processEvent(event)
                    except Exception as e:
                        success = False
                        print(traceback.format_exc())
                    
                elif self.parent().reconnectSegmentButtonClicked:
                    self.chosenVoxelsList.append([tuple(trueCoord), segmentIndex, i])
                    if len(self.chosenVoxelsList) < 4:
                        print(self.chosenVoxelsList)
                        success = False
                    else:
                        event['type'] = 'reconnect'
                        event['chosenVoxelsList'] = self.chosenVoxelsList
                        try:
                            success, event = self.processEvent(event)
                        except Exception as e:
                            success = False
                            print(traceback.format_exc())
                
                elif self.parent().growSegmentButtonClicked:
                    self.chosenVoxelsList.append([tuple(trueCoord), segmentIndex, i])
                    if len(self.chosenVoxelsList) < 4:
                        print(self.chosenVoxelsList)
                        success = False
                    else:
                        event['type'] = 'grow'
                        event['chosenVoxelsList'] = self.chosenVoxelsList
                        try:
                            success, event = self.processEvent(event)
                        except Exception as e:
                            success = False
                            print(traceback.format_exc())
                
                elif self.parent().cutSegmentButtonClicked:
                    pass
                
                else:
                    self.chosenVoxelsList.append([tuple(trueCoord), segmentIndex, i])

                
                # If event returns success, add it to the eventList
                if success:
                    self.eventList.append(event)
                    print('Event {}, eventList has {} events'.format(success, len(self.eventList)))
                    self.checkCycle()
                    break
                else:
                    # print('Event {}, eventList has {} events'.format(success, len(self.eventList)))
                    break
    
    def reportCycleInfo(self):
        validCycleCounter = 0
        for cycleIndex, cycleInfo in self.cycle2SegmentDict.items():
            segmentsValidList = list(cycleInfo.values())
            if all(segmentsValidList) and len(segmentsValidList) != 0:
                validCycleCounter += 1
        
        print('{} cycles remaining (reportCycleInfo)'.format(validCycleCounter))
    
    def addExtraInfo(self, **kwds):
        args = ['segmentList', 'indexVolume', 'affineTransform', 'offset', 'skeletonNodesStartIndex', 'segmentStartIndex', 
                 'segment2CycleDict', 'cycle2SegmentDict', 'G', 'segmentListDict', 'segmentIndexUsed', 'resultFolder']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
            
        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])
    
    def mergeSegments(self, segment, event, mergeOnHead=True, mergeOnTail=True):
        assert segment[0] != segment[-1], 'Self loop detected'

        voxel = segment[0] # segment head
        if self.G.degree(voxel) == 2 and mergeOnHead:
            neighbours = list(self.G.neighbors(voxel))
            segmentIndex1a, segmentIndex1b = self.G[voxel][tuple(neighbours[0])]['segmentIndex'], self.G[voxel][tuple(neighbours[1])]['segmentIndex']
            if segmentIndex1a != segmentIndex1b:
                # remove segment1a/segment1b in GLLinePlotItem
                self.items[self.segmentStartIndex + segmentIndex1a].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
                self.items[self.segmentStartIndex + segmentIndex1b].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
                # remove segment1a/segment1b from segmentIndexUsed
                self.segmentIndexUsed.remove(segmentIndex1a)
                self.segmentIndexUsed.remove(segmentIndex1b)
                # remove segment1a/segment1b from indexVolume (this step is done by adding newSegment to indexVolume)
                # obtain segment1a/segment1b
                segment1a = self.segmentList[segmentIndex1a]
                segment1b = self.segmentList[segmentIndex1b]
                # place segment1a/segment1b such that segment1a's tail is connected to segment1b's head at voxel
                if segment1a[-1] != voxel:
                    segment1a = segment1a[::-1]
                if segment1b[0] != voxel:
                    segment1b = segment1b[::-1]
                
                segmentIndex = event['segmentIndex']
                segment = self.segmentList[segmentIndex]
                assert segment1a[-1] == voxel, '\nseg1a({}):{},\n seg1b({}):{},\n voxel:{},\n neighbours:{},\n seg({}):{}'.format(segmentIndex1a, segment1a, segmentIndex1b, segment1b, voxel, neighbours, segmentIndex, segment)
                assert segment1b[0] == voxel, '\nseg1a({}):{},\n seg1b({}):{},\n voxel:{},\n neighbours:{},\n seg({}):{}'.format(segmentIndex1a, segment1a, segmentIndex1b, segment1b, voxel, neighbours, segmentIndex, segment)
                
                newSegment = segment1a + segment1b[1:]
                newSegmentIndex = len(self.segmentList)
                # add new segment to segmentList
                self.segmentList.append(newSegment)
                # add new segment to segmentListDict
                self.segmentListDict[tuple(newSegment)] = newSegmentIndex
                self.segmentListDict[tuple(newSegment[::-1])] = newSegmentIndex
                # add new segment to segmentIndexUsed
                self.segmentIndexUsed.append(newSegmentIndex)
                # add new segment to graph
                self.G.add_path(newSegment, segmentIndex=int(newSegmentIndex))
                # add new segment to GLLinePlotItem
                newSegmentCoords = np.array(newSegment, dtype=np.int16)
                aa = gl.GLLinePlotItem(pos=newSegmentCoords, width=3, color=pg.glColor('r'))
                aa.translate(self.offset[0], self.offset[1], self.offset[2])
                self.addItem(aa)
                # add new segment to indexVolume
                self.indexVolume[tuple(newSegmentCoords.T)] = newSegmentIndex
                event['segmentIndex1aMerge'] = segmentIndex1a
                event['segmentIndex1bMerge'] = segmentIndex1b
                event['newSegmentIndex1Merge'] = newSegmentIndex
                event['mergeOnHead'] = mergeOnHead
            else:
                errorSegment = self.segmentList[segmentIndex1a]
                degreeList = [v for _, v in self.G.degree(errorSegment)]
                voxelLoc = errorSegment.index(voxel)
                possibleSegments = [idx for idx, seg in enumerate(self.segmentList) if seg[0]==voxel or seg[-1]==voxel]
                print('voxel={}, voxelLoc={}, degreeList={}, segmentList len={}'.format(voxel, voxelLoc, degreeList, len(self.segmentList)))
                print('segmentIndex={}, segmentIndex1a={}, segmentIndex1b={}, possibleSegments={}'.format(event['segmentIndex'], segmentIndex1a, segmentIndex1b, possibleSegments))
        
        voxel = segment[-1] # segment tail
        if self.G.degree(voxel) == 2 and mergeOnTail:
            neighbours = list(self.G.neighbors(voxel))
            segmentIndex2a, segmentIndex2b = self.G[voxel][tuple(neighbours[0])]['segmentIndex'], self.G[voxel][tuple(neighbours[1])]['segmentIndex']
            assert(segmentIndex2a != segmentIndex2b)
            # remove segment2a/segment2b in GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex2a].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
            self.items[self.segmentStartIndex + segmentIndex2b].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
            # remove segment2a/segment2b from segmentIndexUsed
            self.segmentIndexUsed.remove(segmentIndex2a)
            self.segmentIndexUsed.remove(segmentIndex2b)
            # remove segment1a/segment1b from indexVolume (this step is done by adding newSegment to indexVolume)
            # obtain segment2a/segment2b
            segment2a = self.segmentList[segmentIndex2a]
            segment2b = self.segmentList[segmentIndex2b]
            # place segment2a/segment2b such that segment1a's tail is connected to segment1b's head at voxel
            if segment2a[-1] != voxel:
                segment2a = segment2a[::-1]
            if segment2b[0] != voxel:
                segment2b = segment2b[::-1]
            
            segmentIndex = event['segmentIndex']
            segment = self.segmentList[segmentIndex]
            assert segment2a[-1] == voxel, 'seg2a({}):{}, seg2b({}):{}, voxel:{}, neighbours:{}, seg({}):{}'.format(segmentIndex2a, segment2a, segmentIndex2b, segment2b, voxel, neighbours, segmentIndex, segment)
            assert segment2b[0] == voxel, 'seg2a({}):{}, seg2b({}):{}, voxel:{}, neighbours:{}, seg({}):{}'.format(segmentIndex2a, segment2a, segmentIndex2b, segment2b, voxel, neighbours, segmentIndex, segment)
            
            newSegment = segment2a + segment2b[1:]
            newSegmentIndex = len(self.segmentList)
            # add new segment to segmentList
            self.segmentList.append(newSegment)
            # add new segment to segmentListDict
            self.segmentListDict[tuple(newSegment)] = newSegmentIndex
            self.segmentListDict[tuple(newSegment[::-1])] = newSegmentIndex
            # add new segment to segmentIndexUsed
            self.segmentIndexUsed.append(newSegmentIndex)
            # add new segment to graph
            self.G.add_path(newSegment, segmentIndex=int(newSegmentIndex))
            # add new segment to GLLinePlotItem
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=newSegmentCoords, width=3, color=pg.glColor('r'))
            aa.translate(self.offset[0], self.offset[1], self.offset[2])
            self.addItem(aa)
            # add new segment to indexVolume
            self.indexVolume[tuple(newSegmentCoords.T)] = newSegmentIndex
            event['segmentIndex2aMerge'] = segmentIndex2a
            event['segmentIndex2bMerge'] = segmentIndex2b
            event['newSegmentIndex2Merge'] = newSegmentIndex
            event['mergeOnTail'] = mergeOnTail

        return event
    
    def mergeSegmentsReverse(self, event, head=True, tail=True):
        if 'newSegmentIndex2Merge' in event and tail:
            segmentIndex2a = event['segmentIndex2aMerge']
            segmentIndex2b = event['segmentIndex2bMerge']
            newSegmentIndex2 = event['newSegmentIndex2Merge']
            segment2a = self.segmentList[segmentIndex2a]
            segment2b = self.segmentList[segmentIndex2b]
            newSegment = self.segmentList[newSegmentIndex2]
            # remove new segment from GLLinePlotItem
            self.removeItem(self.items[self.segmentStartIndex + newSegmentIndex2])
            # remove new segment from graph edges
            edgesToRemove = [(newSegment[ii], newSegment[ii + 1]) for ii in range(len(newSegment) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            # remove new segment from segmentListDict
            del self.segmentListDict[tuple(newSegment)]
            del self.segmentListDict[tuple(newSegment[::-1])]
            # remove new segment from segmentList
            assert newSegmentIndex2 == len(self.segmentList)-1, 'newSegmentIndex2={}, segmentList len={}'.format(newSegmentIndex2, len(self.segmentList))
            self.segmentList.pop(newSegmentIndex2)
            # remove new segment from segmentIndexUsed
            assert newSegmentIndex2 in self.segmentIndexUsed, 'newSegId2={}'.format(newSegmentIndex2)
            self.segmentIndexUsed.remove(newSegmentIndex2)
            # remove new segment from indexVolume (this step is done by restoring segment2a/segment2b)

            # restore segment2a/segment2b in GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex2a].setData(pos=np.array(segment2a, dtype=np.int16))
            self.items[self.segmentStartIndex + segmentIndex2b].setData(pos=np.array(segment2b, dtype=np.int16))
            # update edge segmentIndex for segment2a/segment2b in the graph
            self.G.add_path(segment2a, segmentIndex=int(segmentIndex2a))
            self.G.add_path(segment2b, segmentIndex=int(segmentIndex2b))
            # restore segment2a/segment2b in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex2a)
            self.segmentIndexUsed.append(segmentIndex2b)
            # restore segment2a/segment2b in indexVolume
            segment2aCoords = np.array(segment2a, dtype=np.int16)
            segment2bCoords = np.array(segment2b, dtype=np.int16)
            self.indexVolume[tuple(segment2aCoords.T)] = segmentIndex2a
            self.indexVolume[tuple(segment2bCoords.T)] = segmentIndex2b

        if 'newSegmentIndex1Merge' in event and head:
            segmentIndex1a = event['segmentIndex1aMerge']
            segmentIndex1b = event['segmentIndex1bMerge']
            newSegmentIndex1 = event['newSegmentIndex1Merge']
            segment1a = self.segmentList[segmentIndex1a]
            segment1b = self.segmentList[segmentIndex1b]
            newSegment = self.segmentList[newSegmentIndex1]
            # remove new segment from GLLinePlotItem
            self.removeItem(self.items[self.segmentStartIndex + newSegmentIndex1])
            # remove new segment from graph edges
            edgesToRemove = [(newSegment[ii], newSegment[ii + 1]) for ii in range(len(newSegment) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            # remove new segment from segmentListDict
            del self.segmentListDict[tuple(newSegment)]
            del self.segmentListDict[tuple(newSegment[::-1])]
            # remove new segment from segmentList
            assert newSegmentIndex1 == len(self.segmentList)-1, 'newSegmentIndex1={}, segmentList len={}'.format(newSegmentIndex1, len(self.segmentList))
            self.segmentList.pop(newSegmentIndex1)
            # remove new segment from segmentIndexUsed
            assert newSegmentIndex1 in self.segmentIndexUsed, 'newSegId1={}'.format(newSegmentIndex1)
            self.segmentIndexUsed.remove(newSegmentIndex1)
            # remove new segment from indexVolume (this step is done by restoring segment1a/segment1b)

            # restore segment1a/segment1b in GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex1a].setData(pos=np.array(segment1a, dtype=np.int16))
            self.items[self.segmentStartIndex + segmentIndex1b].setData(pos=np.array(segment1b, dtype=np.int16))
            # update edge segmentIndex for segment1a/segment1b in the graph
            self.G.add_path(segment1a, segmentIndex=int(segmentIndex1a))
            self.G.add_path(segment1b, segmentIndex=int(segmentIndex1b))
            # restore segment1a/segment1b in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex1a)
            self.segmentIndexUsed.append(segmentIndex1b)
            # restore segment1a/segment1b in indexVolume
            segment1aCoords = np.array(segment1a, dtype=np.int16)
            segment1bCoords = np.array(segment1b, dtype=np.int16)
            self.indexVolume[tuple(segment1aCoords.T)] = segmentIndex1a
            self.indexVolume[tuple(segment1bCoords.T)] = segmentIndex1b
    
    def splitSegments(self, segment, event, splitOnHead=True, splitOnTail=True):
        assert segment[0] != segment[-1]
        voxel = segment[0] # segment head
        if self.G.degree(voxel) == 3 and splitOnHead:
            neighbours = [neighbour for neighbour in self.G.neighbors(voxel) if neighbour != segment[1]]
            assert len(neighbours) == 2, 'neighbours={}'.format(neighbours)
            segmentIndexLeft, segmentIndexRight = self.G[voxel][tuple(neighbours[0])]['segmentIndex'], self.G[voxel][tuple(neighbours[1])]['segmentIndex']
            assert segmentIndexLeft == segmentIndexRight, '\nsegIdLeft={}, segIdRight={}'.format(segmentIndexLeft, segmentIndexRight)
            segmentIndex1 = segmentIndexLeft
            # remove segment1 from GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex1].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
            # remove segment1 from segmentIndexUsed
            self.segmentIndexUsed.remove(segmentIndex1)
            # remove segment1 from indexVolume (this step is done by adding segment1a/segment1b to indexVolume)

            # obtain segment1a/segment1b
            segment1 = self.segmentList[segmentIndex1]
            voxelLoc = segment1.index(voxel)
            segment1a = segment1[:(voxelLoc + 1)]
            segment1b = segment1[voxelLoc:]

            segmentIndex1a = len(self.segmentList)
            segmentIndex1b = len(self.segmentList) + 1
            # add segment1a/segment1b to segmentList
            self.segmentList.append(segment1a)
            self.segmentList.append(segment1b)
            # add segment1a/segment1b to segmentListDict
            self.segmentListDict[tuple(segment1a)] = segmentIndex1a
            self.segmentListDict[tuple(segment1a[::-1])] = segmentIndex1a
            self.segmentListDict[tuple(segment1b)] = segmentIndex1b
            self.segmentListDict[tuple(segment1b[::-1])] = segmentIndex1b
            # add segment1a/segment1b to segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex1a)
            self.segmentIndexUsed.append(segmentIndex1b)
            # add segment1a/segment1b to graph
            self.G.add_path(segment1a, segmentIndex=int(segmentIndex1a))
            self.G.add_path(segment1b, segmentIndex=int(segmentIndex1b))
            # add segment1a/segment1b to GLLinePlotItem
            segment1aCoords = np.array(segment1a, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=segment1aCoords, width=3, color=pg.glColor('r'))
            aa.translate(self.offset[0], self.offset[1], self.offset[2])
            self.addItem(aa)
            segment1bCoords = np.array(segment1b, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=segment1bCoords, width=3, color=pg.glColor('r'))
            aa.translate(self.offset[0], self.offset[1], self.offset[2])
            self.addItem(aa)
            # add segment1a/segment1b to indexVolume
            self.indexVolume[tuple(segment1aCoords.T)] = segmentIndex1a
            self.indexVolume[tuple(segment1bCoords.T)] = segmentIndex1b
            event['segmentIndex1aSplit'] = segmentIndex1a
            event['segmentIndex1bSplit'] = segmentIndex1b
            event['segmentIndex1Split'] = segmentIndex1
            event['splitOnHead'] = splitOnHead
        
        voxel = segment[-1] # segment tail
        if self.G.degree(voxel) == 3 and splitOnTail:
            neighbours = [neighbour for neighbour in self.G.neighbors(voxel) if neighbour != segment[-2]]
            assert len(neighbours) == 2, 'neighbours={}'.format(neighbours)
            segmentIndexLeft, segmentIndexRight = self.G[voxel][tuple(neighbours[0])]['segmentIndex'], self.G[voxel][tuple(neighbours[1])]['segmentIndex']
            assert segmentIndexLeft == segmentIndexRight, '\nsegIdLeft={}, segIdRight={}'.format(segmentIndexLeft, segmentIndexRight)
            segmentIndex2 = segmentIndexLeft
            # remove segment2 from GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex2].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))
            # remove segment2 from segmentIndexUsed
            self.segmentIndexUsed.remove(segmentIndex2)
            # remove segment2 from indexVolume (this step is done by adding segment2a/segment2b to indexVolume)

            # obtain segment2a/segment2b
            segment2 = self.segmentList[segmentIndex2]
            voxelLoc = segment2.index(voxel)
            segment2a = segment2[:(voxelLoc + 1)]
            segment2b = segment2[voxelLoc:]

            segmentIndex2a = len(self.segmentList)
            segmentIndex2b = len(self.segmentList) + 1
            # add segment2a/segment2b to segmentList
            self.segmentList.append(segment2a)
            self.segmentList.append(segment2b)
            # add segment2a/segment2b to segmentListDict
            self.segmentListDict[tuple(segment2a)] = segmentIndex2a
            self.segmentListDict[tuple(segment2a[::-1])] = segmentIndex2a
            self.segmentListDict[tuple(segment2b)] = segmentIndex2b
            self.segmentListDict[tuple(segment2b[::-1])] = segmentIndex2b
            # add segment2a/segment2b to segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex2a)
            self.segmentIndexUsed.append(segmentIndex2b)
            # add segment2a/segment2b to graph
            self.G.add_path(segment2a, segmentIndex=int(segmentIndex2a))
            self.G.add_path(segment2b, segmentIndex=int(segmentIndex2b))
            # add segment2a/segment2b to GLLinePlotItem
            segment2aCoords = np.array(segment2a, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=segment2aCoords, width=3, color=pg.glColor('r'))
            aa.translate(self.offset[0], self.offset[1], self.offset[2])
            self.addItem(aa)
            segment2bCoords = np.array(segment2b, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=segment2bCoords, width=3, color=pg.glColor('r'))
            aa.translate(self.offset[0], self.offset[1], self.offset[2])
            self.addItem(aa)
            # add segment2a/segment2b to indexVolume
            self.indexVolume[tuple(segment2aCoords.T)] = segmentIndex2a
            self.indexVolume[tuple(segment2bCoords.T)] = segmentIndex2b
            event['segmentIndex2aSplit'] = segmentIndex2a
            event['segmentIndex2bSplit'] = segmentIndex2b
            event['segmentIndex2Split'] = segmentIndex2
            event['splitOnTail'] = splitOnTail

        return event
    
    def splitSegmentsReverse(self, event, head=True, tail=True):
        if 'segmentIndex2Split' in event and tail:
            segmentIndex2a = event['segmentIndex2aSplit']
            segmentIndex2b = event['segmentIndex2bSplit']
            segmentIndex2 = event['segmentIndex2Split']
            segment2a = self.segmentList[segmentIndex2a]
            segment2b = self.segmentList[segmentIndex2b]
            segment2 = self.segmentList[segmentIndex2]
            # remove segment2a/segment2b from GLLinePlotItem (2b first, then 2a)
            self.removeItem(self.items[self.segmentStartIndex + segmentIndex2b])
            self.removeItem(self.items[self.segmentStartIndex + segmentIndex2a])
            # remove segment2a/segment2b from graph edges
            edgesToRemove = [(segment2a[ii], segment2a[ii + 1]) for ii in range(len(segment2a) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            edgesToRemove = [(segment2b[ii], segment2b[ii + 1]) for ii in range(len(segment2b) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            # remove segment2a/segment2b from segmentListDict
            del self.segmentListDict[tuple(segment2a)]
            del self.segmentListDict[tuple(segment2a[::-1])]
            del self.segmentListDict[tuple(segment2b)]
            del self.segmentListDict[tuple(segment2b[::-1])]
            # remove segment2a/segment2b from segmentList
            assert segmentIndex2b == len(self.segmentList)-1, 'segmentIndex2b={}, segmentList len={}'.format(segmentIndex2b, len(self.segmentList))
            self.segmentList.pop(segmentIndex2b)
            self.segmentList.pop(segmentIndex2a)
            # remove segment2a/segment2b from segmentIndexUsed
            assert segmentIndex2a in self.segmentIndexUsed, 'segId2a={}'.format(segmentIndex2a)
            self.segmentIndexUsed.remove(segmentIndex2a)
            self.segmentIndexUsed.remove(segmentIndex2b)
            # remove segment2a/segment2b from indexVolume (this step is done by restoring segment2)

            # restore segment2 in GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex2].setData(pos=np.array(segment2, dtype=np.int16))
            # update edge segmentIndex for segment2 in the graph
            self.G.add_path(segment2, segmentIndex=int(segmentIndex2))
            # restore segment2 in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex2)
            # restore segment2 in indexVolume
            segment2Coords = np.array(segment2, dtype=np.int16)
            self.indexVolume[tuple(segment2Coords.T)] = segmentIndex2

        if 'segmentIndex1Split' in event and head:
            segmentIndex1a = event['segmentIndex1aSplit']
            segmentIndex1b = event['segmentIndex1bSplit']
            segmentIndex1 = event['segmentIndex1Split']
            segment1a = self.segmentList[segmentIndex1a]
            segment1b = self.segmentList[segmentIndex1b]
            segment1 = self.segmentList[segmentIndex1]
            # remove segment1a/segment1b from GLLinePlotItem (1b first, then 1a)
            self.removeItem(self.items[self.segmentStartIndex + segmentIndex1b])
            self.removeItem(self.items[self.segmentStartIndex + segmentIndex1a])
            # remove segment1a/segment1b from graph edges
            edgesToRemove = [(segment1a[ii], segment1a[ii + 1]) for ii in range(len(segment1a) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            edgesToRemove = [(segment1b[ii], segment1b[ii + 1]) for ii in range(len(segment1b) - 1)]
            self.G.remove_edges_from(edgesToRemove)
            # remove segment1a/segment1b from segmentListDict
            del self.segmentListDict[tuple(segment1a)]
            del self.segmentListDict[tuple(segment1a[::-1])]
            del self.segmentListDict[tuple(segment1b)]
            del self.segmentListDict[tuple(segment1b[::-1])]
            # remove segment1a/segment1b from segmentList
            assert segmentIndex1b == len(self.segmentList)-1, 'segmentIndex1b={}, segmentList len={}'.format(segmentIndex1b, len(self.segmentList))
            self.segmentList.pop(segmentIndex1b)
            self.segmentList.pop(segmentIndex1a)
            # remove segment1a/segment1b from segmentIndexUsed
            assert segmentIndex1a in self.segmentIndexUsed, 'segId1a={}'.format(segmentIndex1a)
            self.segmentIndexUsed.remove(segmentIndex1a)
            self.segmentIndexUsed.remove(segmentIndex1b)
            # remove segment1a/segment1b from indexVolume (this step is done by restoring segment1)

            # restore segment1 in GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex1].setData(pos=np.array(segment1, dtype=np.int16))
            # update edge segmentIndex for segment1 in the graph
            self.G.add_path(segment1, segmentIndex=int(segmentIndex1))
            # restore segment1 in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex1)
            # restore segment1 in indexVolume
            segment1Coords = np.array(segment1, dtype=np.int16)
            self.indexVolume[tuple(segment1Coords.T)] = segmentIndex1
    
    def checkCycle(self):
        cycleSegmentIndexList = []
        errorIndexList = []
        cycleBasis = nx.cycle_basis(self.G)
        numOfCycles = len(cycleBasis)
        # for cycleIndex, cycle in enumerate(nx.cycle_basis(self.G)):
        for cycle in cycleBasis:
            voxelDegrees = np.array([v for _, v in self.G.degree(cycle)])
            nodesLoc = np.nonzero(voxelDegrees != 2)[0]
            
            for ii in range(len(nodesLoc)):
                if ii != len(nodesLoc) - 1:
                    segmentUsed = cycle[nodesLoc[ii]:(nodesLoc[ii + 1] + 1)]
                else:
                    segmentUsed = cycle[nodesLoc[ii]:]
                    segmentUsed += cycle[:(nodesLoc[0] + 1)]
                
                edges = [(segmentUsed[ii], segmentUsed[ii + 1]) for ii in range(len(segmentUsed) - 1)]
                edgeIndexList = [self.G[edge[0]][edge[1]]['segmentIndex'] for edge in edges]
                uniqueIndex = np.unique(edgeIndexList).tolist()
                if len(uniqueIndex) == 1:
                    cycleSegmentIndexList.append(uniqueIndex[0])
                else:
                    errorIndexList += uniqueIndex
                    print('Error! Multiple segmentIndex found a single branch: {}'.format(edgeIndexList))

        for segmentIndex in self.segmentIndexUsed:
            segmentPlotItem = self.items[self.segmentStartIndex + segmentIndex]
            if segmentIndex in cycleSegmentIndexList:
                segmentPlotItem.setData(color=pg.glColor('b'))
            else:
                if segmentIndex in self.removeList:
                    segmentPlotItem.setData(color=pg.glColor('g'))
                elif segmentIndex in errorIndexList:
                    segmentPlotItem.setData(color=pg.glColor('c'))
                else:
                    segmentPlotItem.setData(color=pg.glColor('r'))
        
        if len(errorIndexList) != 0:
            print('errorIndexList: {}'.format(errorIndexList))

        self.cycleSegmentIndexList = cycleSegmentIndexList
        print('{} cycles remaining'.format(numOfCycles))
    
    
    def processEvent(self, event):
        success = False
        skeletonNodesStartIndex = self.skeletonNodesStartIndex
        skeletonNodesPlotItem = self.items[skeletonNodesStartIndex]
        skeletonColor = skeletonNodesPlotItem.color
        currentNodeIndex = event['nodeIndex']
        if currentNodeIndex != 0:
            currentSkeletonColor = skeletonColor[currentNodeIndex, :]
            if np.all(currentSkeletonColor == [1, 0, 0, 1]):
                currentSkeletonColor = [0, 1, 0, 1]
            else:
                currentSkeletonColor = [1, 0, 0, 1]

            skeletonColor[currentNodeIndex, :] = currentSkeletonColor    
            skeletonNodesPlotItem.setData(color=skeletonColor)

        eventType = event['type']
        if eventType is None:
            success = False
        
        if eventType == 'remove':
            segmentIndex = event['segmentIndex']
            # ignore the segmentIndex from automatically generated removeList if it is not in segmentIndexUsed
            if segmentIndex not in self.segmentIndexUsed:
                success = False
                print('segmentIndex {} not in segmentIndexUsed'.format(segmentIndex))
                return success, event

            segment = self.segmentList[segmentIndex]
            segmentPlotItem = self.items[self.segmentStartIndex + segmentIndex]
            if segmentIndex in self.removeList:
                self.removeList.remove(segmentIndex)
                # add selected edge to graph
                self.G.add_path(segment, segmentIndex=int(segmentIndex))
                event = self.splitSegments(self.segmentList[segmentIndex], event)
                event['action'] = 'deselect'
                success = True
            else:
                # print(segmentPlotItem.color)
                if not segmentPlotItem.color == pg.glColor('r') or self.onLoading:
                    self.removeList.append(segmentIndex)
                    # segmentIndex is still kept in self.segmentIndexUsed but is also kept in self.removeList so that it will be colored green
                    # remove selected edge from graph
                    edgesToRemove = [(segment[ii], segment[ii+1]) for ii in range(len(segment) - 1)]
                    self.G.remove_edges_from(edgesToRemove)
                    event = self.mergeSegments(self.segmentList[segmentIndex], event)
                    event['action'] = 'select'
                    success = True
                else:
                    success = False
                    print('During remove, event failed. segmentPlotItem.color == pg.glColor("r"): {}, self.onLoading = {}'.format(segmentPlotItem.color == pg.glColor('r'), self.onLoading))

        elif eventType == 'reconnect':
            chosenVoxelsList = event['chosenVoxelsList']
            offset = self.offset
            segmentIndex1 = chosenVoxelsList[0][1]
            segmentIndex2 = chosenVoxelsList[2][1]
            segment1 = self.segmentList[segmentIndex1]
            segment2 = self.segmentList[segmentIndex2]
            event['segmentIndex1'] = segmentIndex1
            event['segmentIndex2'] = segmentIndex2
            event['segment1Inverted'] = False
            event['segment2Inverted'] = False

            # check if segment1 needed to be inverted
            segment1StartIndex = segment1.index(chosenVoxelsList[0][0])
            segment1EndIndex = segment1.index(chosenVoxelsList[1][0])
            if segment1StartIndex > segment1EndIndex:
                segment1 = segment1[::-1]
                segment1StartIndex = len(segment1) - segment1StartIndex - 1
                segment1EndIndex = len(segment1) - segment1EndIndex - 1
                event['segment1Inverted'] = True
            
            event['segment1StartIndex'] = segment1StartIndex
            event['segment1EndIndex'] = segment1EndIndex
            
            # check if segment2 needed to be inverted
            segment2StartIndex = segment2.index(chosenVoxelsList[2][0])
            segment2EndIndex = segment2.index(chosenVoxelsList[3][0])
            if segment2StartIndex > segment2EndIndex:
                segment2 = segment2[::-1]
                segment2StartIndex = len(segment2) - segment2StartIndex - 1
                segment2EndIndex = len(segment2) - segment2EndIndex - 1
                event['segment2Inverted'] = True
            
            event['segment2StartIndex'] = segment2StartIndex
            event['segment2EndIndex'] = segment2EndIndex
            
            tempSegment = segment1[:(segment1EndIndex + 1)] + segment2[segment2StartIndex:]
            weightPool = [20, len(tempSegment), round(2 * len(tempSegment))]
            connectionSuccess = False
            for tryIndex, weightSelected in enumerate(weightPool):
                newSegmentCoords = np.array(tempSegment)
                weights = np.ones((len(tempSegment),))
                weights[segment1EndIndex:(segment1EndIndex + 2)] = weightSelected
                _, _, value = splineInterpolation(newSegmentCoords, np.linspace(0, 1, 100), w=weights)
                pointDiff= np.diff(value,axis=0) # get distances between segments
                segLength = np.sqrt((pointDiff * pointDiff).sum(axis=1)) # segment magnitudes
                lengthOfSpline = np.sum(segLength) 
                if lengthOfSpline / 0.7 >= 100:
                    numOfPoints = int(lengthOfSpline / 0.7)
                    _, _, value = splineInterpolation(newSegmentCoords, np.linspace(0, 1, numOfPoints), w=weights)
                # valueView = (value + offset) * affineTransform
                # aa = gl.GLLinePlotItem(pos=valueView, width=3, color=pg.glColor('y'))
    
                # Discretize new segment
                tempSegmentCoordsDiscretized = np.round(value)
                _, idx = np.unique(tempSegmentCoordsDiscretized, axis=0, return_index=True)
                tempSegmentCoordsDiscretized = tempSegmentCoordsDiscretized[np.sort(idx)].astype(np.int16)
                tempSegmentCoordsDiscretizedTuple = list(map(tuple, tempSegmentCoordsDiscretized))
                # print('new segment discretized')
                if chosenVoxelsList[1][0] in tempSegmentCoordsDiscretizedTuple:
                    tempSegmentConnectHeadIndex = tempSegmentCoordsDiscretizedTuple.index(chosenVoxelsList[1][0])
                else:
                    if tryIndex < len(weightPool) - 1:
                        print('Reconnect: {}th try failed'.format(tryIndex+1))
                        continue
                    else:
                        tempSegmentConnectHeadIndex = -1 # means not found
                        # aa = gl.GLLinePlotItem(pos=value, width=3, color=pg.glColor('y'))
                        # aa.translate(offset[0], offset[1], offset[2])
                        # self.addItem(aa)
                        # aa = gl.GLScatterPlotItem(pos=tempSegmentCoordsDiscretized, size=6, color=pg.glColor('g'))
                        # aa.translate(offset[0], offset[1], offset[2])
                        # self.addItem(aa)
                        # print('new segment connection head not found')
                        # print('new segment: {},\nsegment1: {},\nsegment2: {}'.format(tempSegmentCoordsDiscretizedTuple, segment1, segment2))
                        # temp = np.round(value).astype(np.int16)
                        # temp = list(map(tuple, temp))
                        # print(temp)
                        self.chosenVoxelsList = []
                        return success, event
                
                if chosenVoxelsList[2][0] in tempSegmentCoordsDiscretizedTuple:
                    tempSegmentConnectTailIndex = tempSegmentCoordsDiscretizedTuple.index(chosenVoxelsList[2][0])
                else:
                    if tryIndex < len(weightPool) - 1:
                        print('Reconnect: {}th try failed'.format(tryIndex+1))
                        continue
                    else:
                        tempSegmentConnectTailIndex = -1 # means not found
                        # aa = gl.GLLinePlotItem(pos=value, width=3, color=pg.glColor('y'))
                        # aa.translate(offset[0], offset[1], offset[2])
                        # self.addItem(aa)
                        # aa = gl.GLScatterPlotItem(pos=tempSegmentCoordsDiscretized, size=6, color=pg.glColor('g'))
                        # aa.translate(offset[0], offset[1], offset[2])
                        # self.addItem(aa)
                        print('new segment connection tail not found')
                        self.chosenVoxelsList = []
                        return success, event
                
                break
            
            # check if new voxels have been occupied by other segments
            # print('connection head = {}, connection tail = {}'.format(tempSegmentConnectHeadIndex, tempSegmentConnectTailIndex))
            voxelIndicesMiddle = self.indexVolume[tuple(tempSegmentCoordsDiscretized[tempSegmentConnectHeadIndex:(tempSegmentConnectTailIndex+1), :].T)]
            uniqueSegmentIndices = np.unique(voxelIndicesMiddle)
            allowedIndices = [-1, segmentIndex1, segmentIndex2]
            for uniqueSegmentIndex in uniqueSegmentIndices:
                if uniqueSegmentIndex in allowedIndices:
                    pass
                else:
                    print('One of the voxel is out of allowed segment indices ({}) and is occupied by segment {}'.format(allowedIndices, uniqueSegmentIndex))
                    self.chosenVoxelsList = []
                    return success, event

            skeletonCoords = skeletonNodesPlotItem.pos

            # mark the voxels of segment1 that needs to be removed in GLScatterPlotItem
            if self.G.degree(segment1[0]) == 1:
                segment1ScatterRemoveHead = 0
            else:
                segment1ScatterRemoveHead = 1
            event['segment1ScatterRemoveHead'] = segment1ScatterRemoveHead
            
            if self.G.degree(segment1[-1]) == 1:
                segment1ScatterRemoveTail = len(segment1) - 1
            else:
                segment1ScatterRemoveTail = len(segment1) - 2
            event['segment1ScatterRemoveTail'] = segment1ScatterRemoveTail

            voxelsToRemove = np.array(segment1[segment1ScatterRemoveHead:(segment1ScatterRemoveTail+1)], dtype=np.int16)
            skeletonCoords = removeBFromA(skeletonCoords, voxelsToRemove)

            # mark the voxels of segment2 that needs to be removed in GLScatterPlotItem
            if self.G.degree(segment2[0]) == 1:
                segment2ScatterRemoveHead = 0
            else:
                segment2ScatterRemoveHead = 1
            event['segment2ScatterRemoveHead'] = segment2ScatterRemoveHead
            
            if self.G.degree(segment2[-1]) == 1:
                segment2ScatterRemoveTail = len(segment2) - 1
            else:
                segment2ScatterRemoveTail = len(segment2) - 2
            event['segment2ScatterRemoveTail'] = segment2ScatterRemoveTail

            voxelsToRemove = np.array(segment2[segment2ScatterRemoveHead:(segment2ScatterRemoveTail+1)], dtype=np.int16)
            skeletonCoords = removeBFromA(skeletonCoords, voxelsToRemove)

            # remove segment1/segment2 from GLLinePlotItem
            segment1PlotItem = self.items[self.segmentStartIndex + segmentIndex1]
            segment1PlotItem.setData(pos=np.array([]).reshape(-1, 3))
            segment2PlotItem = self.items[self.segmentStartIndex + segmentIndex2]
            segment2PlotItem.setData(pos=np.array([]).reshape(-1, 3))

            # remove segment1/segment2 from graph
            pathToRemove = [(segment1[ii], segment1[ii + 1]) for ii in range(len(segment1) - 1)]
            self.G.remove_edges_from(pathToRemove)
            pathToRemove = [(segment2[ii], segment2[ii + 1]) for ii in range(len(segment2) - 1)]
            self.G.remove_edges_from(pathToRemove)
            
            # remove segment1/segment2 from indexVolume
            segment1Coords = np.array(segment1, dtype=np.int16)
            segment2Coords = np.array(segment2, dtype=np.int16)
            self.indexVolume[tuple(segment1Coords.T)] = -1
            self.indexVolume[tuple(segment2Coords.T)] = -1

            # remove segment1/segment2 from segmentIndexUsed
            self.segmentIndexUsed.remove(segmentIndex1)
            self.segmentIndexUsed.remove(segmentIndex2)

            # add new voxels to GLScatterPlotItem
            voxelsToAdd = np.vstack((np.array(segment1[segment1ScatterRemoveHead:segment1EndIndex], dtype=np.int16), 
                                    tempSegmentCoordsDiscretized[tempSegmentConnectHeadIndex:tempSegmentConnectTailIndex],
                                    np.array(segment2[segment2StartIndex:(segment2ScatterRemoveTail + 1)], dtype=np.int16)
                                    ))

            # new segment
            newSegment = segment1[:segment1EndIndex] + tempSegmentCoordsDiscretizedTuple[tempSegmentConnectHeadIndex:tempSegmentConnectTailIndex] + segment2[segment2StartIndex:]
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            # add new segment to GLLinePlotItem
            aa = gl.GLLinePlotItem(pos=newSegmentCoords, width=3, color=pg.glColor('r'))
            aa.translate(offset[0], offset[1], offset[2])
            self.addItem(aa)
                            
            # add new segment coords to GLScatterPlotItem
            skeletonCoords = np.vstack((skeletonCoords, voxelsToAdd))
            skeletonColor = np.full((len(skeletonCoords), 4), 1)
            skeletonColor[:, 1:3] = 0 # red
            skeletonNodesPlotItem.setData(pos=skeletonCoords, color=skeletonColor)

            # add new segment to indexVolume
            newSegmentIndex = len(self.segmentList)
            self.indexVolume[tuple(voxelsToAdd.T)] = newSegmentIndex

            # add new segment to segmentList/segmentListDict
            self.segmentListDict[tuple(newSegment)] = newSegmentIndex
            self.segmentListDict[tuple(newSegment[::-1])] = newSegmentIndex
            event['newSegmentIndex'] = newSegmentIndex
            self.segmentList.append(newSegment)
            
            # add new segment to graph
            self.G.add_path(newSegment, segmentIndex=int(newSegmentIndex))

            # add new segment to segmentIndexUsed
            self.segmentIndexUsed.append(newSegmentIndex)

            # merge segments at segment1 tail and segment2 head
            event = self.mergeSegments(segment1, event, mergeOnHead=False, mergeOnTail=True)
            event = self.mergeSegments(segment2, event, mergeOnHead=True, mergeOnTail=False)
            
            # event['chosenVoxelsList'] = chosenVoxelsList
            self.clearChosenList()
            success = True

        elif eventType == 'grow':
            chosenVoxelsList = event['chosenVoxelsList']
            affineTransform = self.affineTransform
            offset = self.offset
            segmentIndex1 = chosenVoxelsList[0][1]
            segmentIndex2 = chosenVoxelsList[2][1]
            segment1 = self.segmentList[segmentIndex1]
            segment2 = self.segmentList[segmentIndex2]
            event['segmentIndex1'] = segmentIndex1
            event['segmentIndex2'] = segmentIndex2
            event['segment1Inverted'] = False
            event['segment2Inverted'] = False

            # check if segment1 needed to be inverted
            segment1StartIndex = segment1.index(chosenVoxelsList[0][0])
            segment1EndIndex = segment1.index(chosenVoxelsList[1][0])
            if segment1StartIndex > segment1EndIndex:
                segment1 = segment1[::-1]
                segment1StartIndex = len(segment1) - segment1StartIndex - 1
                segment1EndIndex = len(segment1) - segment1EndIndex - 1
                event['segment1Inverted'] = True
            
            event['segment1StartIndex'] = segment1StartIndex
            event['segment1EndIndex'] = segment1EndIndex
            
            # check if segment2 needed to be inverted
            segment2StartIndex = segment2.index(chosenVoxelsList[2][0])
            segment2EndIndex = segment2.index(chosenVoxelsList[3][0])
            if segment2StartIndex > segment2EndIndex:
                segment2 = segment2[::-1]
                segment2StartIndex = len(segment2) - segment2StartIndex - 1
                segment2EndIndex = len(segment2) - segment2EndIndex - 1
                event['segment2Inverted'] = True
            
            event['segment2StartIndex'] = segment2StartIndex
            event['segment2EndIndex'] = segment2EndIndex
            
            tempSegment = segment1[:(segment1EndIndex + 1)] + segment2[segment2StartIndex:]
            newSegmentCoords = np.array(tempSegment)
            weights = np.ones((len(tempSegment),))
            weights[segment1EndIndex:(segment1EndIndex + 2)] = 20
            _, _, value = splineInterpolation(newSegmentCoords, np.linspace(0, 1, 100), w=weights)
            pointDiff= np.diff(value,axis=0) # get distances between segments
            segLength = np.sqrt((pointDiff * pointDiff).sum(axis=1)) # segment magnitudes
            lengthOfSpline = np.sum(segLength) 
            if lengthOfSpline / 0.7 >= 100:
                numOfPoints = int(lengthOfSpline / 0.7)
                _, _, value = splineInterpolation(newSegmentCoords, np.linspace(0, 1, numOfPoints), w=weights)
            # valueView = (value + offset) * affineTransform
            # aa = gl.GLLinePlotItem(pos=valueView, width=3, color=pg.glColor('y'))

            # Discretize new segment
            tempSegmentCoordsDiscretized = np.round(value)
            _, idx = np.unique(tempSegmentCoordsDiscretized, axis=0, return_index=True)
            tempSegmentCoordsDiscretized = tempSegmentCoordsDiscretized[np.sort(idx)].astype(np.int16)
            tempSegmentCoordsDiscretizedTuple = list(map(tuple, tempSegmentCoordsDiscretized))
            # print('new segment discretized')
            if chosenVoxelsList[1][0] in tempSegmentCoordsDiscretizedTuple:
                tempSegmentConnectHeadIndex = tempSegmentCoordsDiscretizedTuple.index(chosenVoxelsList[1][0])
            else:
                tempSegmentConnectHeadIndex = -1 # means not found
                print('new segment connection head not found')
                self.chosenVoxelsList = []
                return success, event
            
            if chosenVoxelsList[2][0] in tempSegmentCoordsDiscretizedTuple:
                tempSegmentConnectTailIndex = tempSegmentCoordsDiscretizedTuple.index(chosenVoxelsList[2][0])
            else:
                tempSegmentConnectTailIndex = -1 # means not found
                print('new segment connection tail not found')
                self.chosenVoxelsList = []
                return success, event
            
            # check if new voxels have been occupied by other segments
            # print('connection head = {}, connection tail = {}'.format(tempSegmentConnectHeadIndex, tempSegmentConnectTailIndex))
            voxelIndicesMiddle = self.indexVolume[tuple(tempSegmentCoordsDiscretized[(tempSegmentConnectHeadIndex + 1):tempSegmentConnectTailIndex, :].T)]
            uniqueSegmentIndices = np.unique(voxelIndicesMiddle)
            allowedIndices = [-1]
            for uniqueSegmentIndex in uniqueSegmentIndices:
                if uniqueSegmentIndex in allowedIndices:
                    pass
                else:
                    print('One of the voxel is out of allowed segment indices ({}) and is occupied by segment {}'.format(allowedIndices, uniqueSegmentIndex))
                    self.chosenVoxelsList = []
                    return success, event

            skeletonCoords = skeletonNodesPlotItem.pos

            # add new voxels to GLScatterPlotItem
            voxelsToAdd = tempSegmentCoordsDiscretized[(tempSegmentConnectHeadIndex + 1):tempSegmentConnectTailIndex]
            skeletonCoords = np.vstack((skeletonCoords, voxelsToAdd))
            skeletonColor = np.full((len(skeletonCoords), 4), 1)
            skeletonColor[:, 1:3] = 0
            skeletonNodesPlotItem.setData(pos=skeletonCoords, color=skeletonColor)

            # add new segment to GLLinePlotItem
            newSegment = tempSegmentCoordsDiscretizedTuple[tempSegmentConnectHeadIndex:(tempSegmentConnectTailIndex + 1)]
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            aa = gl.GLLinePlotItem(pos=newSegmentCoords, width=3, color=pg.glColor('r'))
            aa.translate(offset[0], offset[1], offset[2])
            self.addItem(aa)

            # add new segment to segmentList
            newSegmentIndex = len(self.segmentList)
            self.segmentList.append(newSegment)
            event['newSegmentIndex'] = newSegmentIndex

            # add new segment to segmentIndexUsed
            self.segmentIndexUsed.append(newSegmentIndex)

            # add new segment to graph
            self.G.add_path(newSegment, segmentIndex=int(newSegmentIndex))

            # add new segment to indexVolume
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            self.indexVolume[tuple(newSegmentCoords.T)] = newSegmentIndex

            # split segments on both ends
            event = self.splitSegments(newSegment, event)
         
            # event['chosenVoxelsList'] = chosenVoxelsList
            self.clearChosenList()
            success = True
        
        elif eventType == 'cut':
            offset = self.offset
            chosenVoxelsList = event['chosenVoxelsList']
            voxelDegreeList = np.array([self.G.degree(voxelInfo[0]) for voxelInfo in chosenVoxelsList])
            voxelIndexList = np.array([voxelInfo[1] for voxelInfo in chosenVoxelsList])
            assert len(np.unique(voxelIndexList[:-1])), '{}'.format(np.unique(voxelIndexList[:-1]))
            segmentIndex = chosenVoxelsList[0][1]
            segment = self.segmentList[segmentIndex]
            event['segmentIndex'] = segmentIndex

            # check if segment needed to be inverted
            segmentStartIndex = segment.index(chosenVoxelsList[0][0])
            segmentEndIndex = segment.index(chosenVoxelsList[-1][0])
            event['segmentInverted'] = False
            if segmentStartIndex > segmentEndIndex:
                segment = segment[::-1]
                segmentStartIndex = len(segment) - segmentStartIndex - 1
                segmentEndIndex = len(segment) - segmentEndIndex - 1
                event['segmentInverted'] = True
            
            event['segmentStartIndex'] = segmentStartIndex
            event['segmentEndIndex'] = segmentEndIndex

            # remove old segment from graph
            edgesToRemove = [(segment[ii], segment[ii + 1]) for ii in range(len(segment) - 1)]
            self.G.remove_edges_from(edgesToRemove)

            # remove old segment from segmentIndexUsed
            self.segmentIndexUsed.remove(segmentIndex)

            # remove old segment from GLLinePlotItem
            self.items[self.segmentStartIndex + segmentIndex].setData(pos=np.array([], dtype=np.int16).reshape(-1, 3))

            # remove old segment from indexVolume
            segmentCoords = np.array(segment, dtype=np.int16)
            self.indexVolume[tuple(segmentCoords.T)] = -1

            # remove old segment from GLScatterPlotItem
            voxelsToRemove = np.array(segment[segmentStartIndex:(segmentEndIndex + 1)], dtype=np.int16)
            skeletonCoords = skeletonNodesPlotItem.pos
            skeletonCoords = removeBFromA(skeletonCoords, voxelsToRemove)

            # obtain new segment
            newSegment = segment[:segmentStartIndex]
            newSegmentIndex = len(self.segmentList)
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            event['newSegmentIndex'] = newSegmentIndex

            # add new segment to segmentList
            self.segmentList.append(newSegment)

            # add new segment to segmentIndexUsed
            self.segmentIndexUsed.append(newSegmentIndex)

            # add new segment to graph
            self.G.add_path(newSegment, segmentIndex=int(newSegmentIndex))

            # add new segment to indexVolume
            self.indexVolume[tuple(newSegmentCoords.T)] = newSegmentIndex

            # add new segment to GLScatterPlotItem
            skeletonCoords = np.vstack((skeletonCoords, newSegmentCoords))
            skeletonColor = np.full((len(skeletonCoords), 4), 1)
            skeletonColor[:, 1:3] = 0
            skeletonNodesPlotItem.setData(pos=skeletonCoords, color=skeletonColor)

            # add new segment to GLLinePlotItem
            aa = gl.GLLinePlotItem(pos=newSegmentCoords, width=3, color=pg.glColor('r'))
            aa.translate(offset[0], offset[1], offset[2])
            self.addItem(aa)
            
            # merge at segment tail
            self.mergeSegments(segment, event, mergeOnHead=False, mergeOnTail=True)

            self.clearChosenList()
            success = True

        else:
            pass
        
        return success, event

    def reverseEvent(self, event):
        eventType = event['type']
        if eventType == 'remove':
            segmentIndex = event['segmentIndex']
            segment = self.segmentList[segmentIndex]
            action = event['action']
            if action == 'select':
                self.G.add_path(segment, segmentIndex=int(segmentIndex))
                self.mergeSegmentsReverse(event)
                self.removeList.remove(segmentIndex)
            elif action == 'deselect':
                edgesToRemove = [(segment[ii], segment[ii + 1]) for ii in range(len(segment) - 1)]
                self.G.remove_edges_from(edgesToRemove)
                self.splitSegmentsReverse(event)
                self.removeList.append(segmentIndex)
    
        elif eventType == 'reconnect':
            # reverse segment merge at segment2 head and segment1 tail
            self.mergeSegmentsReverse(event, head=True, tail=False)
            self.mergeSegmentsReverse(event, head=False, tail=True)

            # get segment1 and segment2
            segmentIndex1 = event['segmentIndex1']
            segmentIndex2 = event['segmentIndex2']
            segment1Inverted = event['segment1Inverted']
            segment2Inverted = event['segment2Inverted']
            # print(segment1Inverted, segmentIndex1)
            # print(self.segmentList[segmentIndex1])
            segment1FromList = self.segmentList[segmentIndex1]
            if segment1Inverted:
                segment1 = segment1FromList[::-1]
            else:
                segment1 = segment1FromList
            
            segment2FromList = self.segmentList[segmentIndex2]
            if segment2Inverted:
                segment2 = segment2FromList[::-1]
            else:
                segment2 = segment2FromList
            
            # get new segment
            newSegmentIndex = event['newSegmentIndex']
            newSegment = self.segmentList[newSegmentIndex]

            # remove new segment from segmentIndexUsed
            # assert self.segmentIndexUsed[-1] == newSegmentIndex, '{}/{}'.format(self.segmentIndexUsed[-1], newSegmentIndex)
            self.segmentIndexUsed.remove(newSegmentIndex)

            # remove new segment from graph
            pathToRemove = [(newSegment[ii], newSegment[ii + 1]) for ii in range(len(newSegment) - 1)]
            self.G.remove_edges_from(pathToRemove)

            # remove new segment from segmentList/segmentListDict
            self.segmentList.pop(newSegmentIndex)
            del self.segmentListDict[tuple(newSegment)]
            del self.segmentListDict[tuple(newSegment[::-1])]

            # get added voxel from new segment
            segment1ScatterRemoveHead = event['segment1ScatterRemoveHead']
            segment2ScatterRemoveTail = event['segment2ScatterRemoveTail']
            if segment2ScatterRemoveTail == len(segment2) - 1:
                voxelsAdded = np.array(newSegment[segment1ScatterRemoveHead:], dtype=np.int16)
            elif segment2ScatterRemoveTail == len(segment2) - 2:
                voxelsAdded = np.array(newSegment[segment1ScatterRemoveHead:-1], dtype=np.int16)
            
            # remove new segment coords from indexVolume
            self.indexVolume[tuple(voxelsAdded.T)] = -1
            
            # remove new segment coords from GLScatterPlotItem (plot item is updated later)
            skeletonNodesPlotItem = self.items[self.skeletonNodesStartIndex]
            skeletonCoords = skeletonNodesPlotItem.pos
            skeletonCoords = removeBFromA(skeletonCoords, voxelsAdded)

            # remove new segment from GLLinePlotItem
            newSegmentPlotItem = self.items[self.segmentStartIndex + newSegmentIndex]
            self.removeItem(newSegmentPlotItem)

            # restore segment1/segment2 in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex1)
            self.segmentIndexUsed.append(segmentIndex2)

            # restore segment1/segment2 in graph
            self.G.add_path(segment1, segmentIndex=int(segmentIndex1))
            self.G.add_path(segment2, segmentIndex=int(segmentIndex2))

            # restore voxels from segment1/segment2 to GLScatterPlotItem
            segment1ScatterRemoveHead = event['segment1ScatterRemoveHead']
            segment1scatterRemoveTail = event['segment1ScatterRemoveTail']
            voxelsToAdd1 = np.array(segment1[segment1ScatterRemoveHead:(segment1scatterRemoveTail + 1)], dtype=np.int16)
            segment2ScatterRemoveHead = event['segment2ScatterRemoveHead']
            segment2scatterRemoveTail = event['segment2ScatterRemoveTail']
            voxelsToAdd2 = np.array(segment2[segment2ScatterRemoveHead:(segment2scatterRemoveTail + 1)], dtype=np.int16)
            skeletonCoords = np.vstack((skeletonCoords, voxelsToAdd1, voxelsToAdd2))
            skeletonColor = np.full((len(skeletonCoords), 4), 1)
            skeletonColor[:, 1:3] = 0
            skeletonNodesPlotItem.setData(pos=skeletonCoords, color=skeletonColor)

            # restore segment1/segment2 back to GLLinePlotItem
            segment1Coords = np.array(segment1FromList, dtype=np.int16)
            segment1PlotItem = self.items[self.segmentStartIndex + segmentIndex1]
            segment1PlotItem.setData(pos=segment1Coords)
            segment2Coords = np.array(segment2FromList, dtype=np.int16)
            segment2PlotItem = self.items[self.segmentStartIndex + segmentIndex2]
            segment2PlotItem.setData(pos=segment2Coords)
            
            # restore segment1/segment2 in indexVolume
            self.indexVolume[tuple(segment1Coords.T)] = segmentIndex1
            self.indexVolume[tuple(segment2Coords.T)] = segmentIndex2
        
        elif eventType == 'grow':
            # obtain new segment
            newSegmentIndex = event['newSegmentIndex']
            newSegment = self.segmentList[newSegmentIndex]

            # remove new segment from indexVolume
            newSegmentCoords = np.array(newSegment, dtype=np.int16)
            self.indexVolume[tuple(newSegmentCoords.T)] = -1

            # remove new segment from graph
            edgesToRemove = [(newSegment[ii], newSegment[ii + 1]) for ii in range(len(newSegment) - 1)]
            self.G.remove_edges_from(edgesToRemove)

            # reverse segment split on both ends
            event = self.splitSegmentsReverse(event)

            # remove new segment from segmentIndexUsed
            self.segmentIndexUsed.remove(newSegmentIndex)

            # remove new segment from segmentList
            assert newSegmentIndex == len(self.segmentList) - 1, '{}/{}'.format(newSegmentIndex, len(self.segmentList))
            self.segmentList.pop(newSegmentIndex)

            # remove new segment from GLLinePlotItem
            self.removeItem(self.items[self.segmentStartIndex + newSegmentIndex])

            # remove new segment from GLScatterPlotItem
            skeletonCoords = self.items[self.skeletonNodesStartIndex].pos
            voxelsAdded = newSegmentCoords[1:-1, :]
            skeletonCoords = removeBFromA(skeletonCoords, voxelsAdded)
            self.items[self.skeletonNodesStartIndex].setData(pos=skeletonCoords)
        
        elif eventType == 'cut':
            # reverse segment merge
            self.mergeSegmentsReverse(event)

            # obtain new segment
            newSegmentIndex = event['newSegmentIndex']
            newSegment = self.segmentList[newSegmentIndex]
            newSegmentCoords = np.array(newSegment, dtype=np.int16)

            # remove new segment from GLLinePlotItem
            self.removeItem(self.items[self.segmentStartIndex + newSegmentIndex])

            # remove new segment from GLScatterPlotItem
            skeletonCoords = self.items[self.skeletonNodesStartIndex].pos
            segmentStartIndex = event['segmentStartIndex']
            segmentEndIndex = event['segmentEndIndex']
            voxelsAdded = newSegmentCoords[:segmentStartIndex, :]
            skeletonCoords = removeBFromA(skeletonCoords, voxelsAdded)
            self.items[self.skeletonNodesStartIndex].setData(pos=skeletonCoords)

            # remove new segment from indexVolume
            self.indexVolume[tuple(newSegmentCoords.T)] = -1

            # remove new segment from graph
            edgesToRemove = [(newSegment[ii], newSegment[ii + 1]) for ii in range(len(newSegment) - 1)]
            self.G.remove_edges_from(edgesToRemove)

            # remove new segment from segmentIndexUsed
            self.segmentIndexUsed.remove(newSegmentIndex)

            # remove new segment from segmentList
            assert newSegmentIndex == len(self.segmentList) - 1, '{}/{}'.format(newSegmentIndex, len(self.segmentList))
            self.segmentList.pop(newSegmentIndex)

            # restore old segment in GLScatterPlotItem
            segmentIndex = event['segmentIndex']
            segmentInverted = event['segmentInverted']
            segmentFromList = self.segmentList[segmentIndex]
            if segmentInverted:
                segment = segmentFromList[::-1]
            else:
                segment = segmentFromList
            
            segmentCoords = np.array(segment, dtype=np.int16)
            segmentFromListCoords = np.array(segmentFromList, dtype=np.int16)
            skeletonCoords = np.vstack((skeletonCoords, segmentCoords))
            skeletonColor = np.full((len(skeletonCoords), 4), 1)
            skeletonColor[:, 1:3] = 0
            self.items[self.skeletonNodesStartIndex].setData(pos=skeletonCoords, color=skeletonColor)

            # restore old segment in indexVolume
            self.indexVolume[tuple(segmentCoords.T)] = segmentIndex

            # restore old segment in GLLinPlotItem
            self.items[self.segmentStartIndex + segmentIndex].setData(pos=segmentFromListCoords)

            # restore old segment in segmentIndexUsed
            self.segmentIndexUsed.append(segmentIndex)

            # restore old segment in graph
            self.G.add_path(segment, segmentIndex=int(segmentIndex))

            
        else:
            pass
        
        self.eventList.pop()
        print('Event reversed, eventList has {} events'.format(len(self.eventList)))
    
    def clearChosenList(self):
        chosenVoxelsList = self.chosenVoxelsList
        skeletonNodesPlotItem = self.items[self.skeletonNodesStartIndex]
        skeletonColor = skeletonNodesPlotItem.color
        skeletonColor[:, 0] = 1
        skeletonColor[:, 1] = 0
        skeletonColor[:, 2] = 0
        
        skeletonNodesPlotItem.setData(color=skeletonColor)
        self.chosenVoxelsList = []

def removeBFromA(A, B):
    '''
    Remove rows that are in B from A. (A and B must be integer arrays)
    Taken from: https://stackoverflow.com/questions/40055835/removing-elements-from-an-array-that-are-in-another-array
    '''
    dims = (np.maximum(B.max(0),A.max(0))+1).astype(int)
    # print(dims, A.shape, B.shape, A.dtype, B.dtype)
    # print(np.ravel_multi_index(A.T,dims))
    # print(np.ravel_multi_index(B.T,dims))
    out = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
    return out

def splineInterpolation(coords, pointLoc, smoothing=None, return_derivative=False, k=3, w=None):
    '''
    Use spline curve to fit the vessel skeleton and return the derivative at each end
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


class Vessel(QtGui.QWidget):
    def __init__(self):
        super(Vessel, self).__init__()
        self.init_ui()
        self.qt_connections()
        self.removeSegmentButtonClicked = False
        self.reconnectSegmentButtonClicked = False
        self.growSegmentButtonClicked = False
        self.cutSegmentButtonClicked = False

        # self.timer = pg.QtCore.QTimer()
        # self.timer.timeout.connect(self.moveplot)
        # self.timer.start(500)

    def init_ui(self):
        self.setWindowTitle('Vessel')
        hbox = QtGui.QHBoxLayout()
        self.setLayout(hbox)

        # self.plotwidget = pg.PlotWidget()
        # self.plotwidget = gl.GLViewWidget()
        self.plotwidget = PlotObject()
        
        hbox.addWidget(self.plotwidget, 4)
        
        vbox = QtGui.QVBoxLayout()
        self.removeSegmentButton = QtGui.QPushButton("Remove Segment")
        self.removeSegmentButton.setCheckable(True)
        self.reconnectSegmentButton = QtGui.QPushButton("Reconnect Segment")
        self.reconnectSegmentButton.setCheckable(True)
        self.growSegmentButton = QtGui.QPushButton("Grow Segment")
        self.growSegmentButton.setCheckable(True)
        self.cutSegmentButton = QtGui.QPushButton("Cut segment")
        self.undoLastEventButton = QtGui.QPushButton("Undo Last Event")
        self.clearChosenListButton = QtGui.QPushButton("Clear Chosen List")
        self.checkCycleButton = QtGui.QPushButton("Check Cycle")
        self.saveResultButton = QtGui.QPushButton("Save Result")
        self.showSegmentButton = QtGui.QPushButton("Show Segment")
        self.segmentIndexBox = QtGui.QLineEdit()
        

        vbox.addWidget(self.removeSegmentButton, 1)
        vbox.addWidget(self.reconnectSegmentButton, 1)
        vbox.addWidget(self.growSegmentButton, 1)
        vbox.addWidget(self.cutSegmentButton, 1)
        vbox.addWidget(self.undoLastEventButton, 1)
        vbox.addWidget(self.clearChosenListButton, 1)
        vbox.addWidget(self.checkCycleButton, 1)
        vbox.addWidget(self.saveResultButton, 1)
        vbox.addWidget(self.showSegmentButton, 1)
        vbox.addWidget(self.segmentIndexBox, 1)
        vbox.addStretch(1)
        hbox.addLayout(vbox, 1)

        self.setGeometry(30, 30, 1600, 900)
        self.show()

    def qt_connections(self):
        self.removeSegmentButton.clicked.connect(self.onRemoveClicked)
        self.reconnectSegmentButton.clicked.connect(self.onReconnectClicked)
        self.growSegmentButton.clicked.connect(self.onGrowClicked)
        self.cutSegmentButton.clicked.connect(self.onCutClicked)
        self.undoLastEventButton.clicked.connect(self.onUndoClicked)
        self.clearChosenListButton.clicked.connect(self.onClearClicked)
        self.checkCycleButton.clicked.connect(self.onCheckClicked)
        self.saveResultButton.clicked.connect(self.onSaveClicked)
        self.showSegmentButton.clicked.connect(self.onShowSegmentButtonClicked)

    def moveplot(self):
        self.updateplot()

    def updateplot(self):
        pass

    def onRemoveClicked(self, pressed):
        source = self.sender()
        if pressed:
            source.setDown(True)
        else:
            source.setDown(False)

        self.removeSegmentButtonClicked = source.isDown()
        self.updateplot()

    def onReconnectClicked(self, pressed):
        source = self.sender()
        if pressed:
            source.setDown(True)
        else:
            source.setDown(False)

        self.reconnectSegmentButtonClicked = source.isDown()
        self.updateplot()
    
    def onGrowClicked(self, pressed):
        source = self.sender()
        if pressed:
            source.setDown(True)
        else:
            source.setDown(False)

        self.growSegmentButtonClicked = source.isDown()
        self.updateplot()
    
    def onCutClicked(self):
        event = {'type': 'cut', 'nodeIndex': 0, 'chosenVoxelsList': self.plotwidget.chosenVoxelsList}
        try:
            # cutSegmentButtonClicked is manually set to True here so that its function can be executed and will be reset to False 
            # after event is processed
            self.cutSegmentButtonClicked = True
            success, event = self.plotwidget.processEvent(event)
            self.cutSegmentButtonClicked = False
            if success:
                self.plotwidget.eventList.append(event)
                print('Event {}, eventList has {} events'.format(success, len(self.plotwidget.eventList)))
                self.plotwidget.checkCycle()
        except Exception:
            print(traceback.format_exc())
    
    def onUndoClicked(self):
        lastEvent = self.plotwidget.eventList[-1]
        try:
            self.plotwidget.reverseEvent(lastEvent)
            self.plotwidget.checkCycle()
        except Exception:
            print(traceback.format_exc())
    
    def onClearClicked(self):
        try:
            self.plotwidget.clearChosenList()
        except Exception:
            print(traceback.format_exc())
    
    def onCheckClicked(self):
        try:
            self.plotwidget.checkCycle()
        except Exception:
            print(traceback.format_exc())
    
    def onSaveClicked(self):
        eventList = self.plotwidget.eventList
        segmentList = self.plotwidget.segmentList
        segmentIndexUsed = self.plotwidget.segmentIndexUsed
        removeList = self.plotwidget.removeList
        # Remove terminating branches of length 2
        shortTerminatingBranchCounter = 0
        for segmentIndex in segmentIndexUsed:
            segment = segmentList[segmentIndex]
            if len(segment) == 2 and segmentIndex not in removeList:
                G = self.plotwidget.G
                if (G.degree(segment[0]) == 1 and G.degree(segment[-1]) >= 3):
                    segmentHead, segmentTail = segment[-1], segment[0]
                elif (G.degree(segment[-1]) == 1 and G.degree(segment[0]) >= 3):
                    segmentHead, segmentTail = segment[0], segment[-1]
                else:
                    continue
                
                shortTerminatingBranchCounter += 1
                segmentIndex = G[segmentHead][segmentTail]['segmentIndex']
                # Create empty event
                event = {'type': 'remove', 'nodeIndex': 0, 'trueCoord': segmentTail, 'segmentIndex': segmentIndex}
                self.plotwidget.onLoading = True
                success, event = self.plotwidget.processEvent(event)
                self.plotwidget.onLoading = False
                if success:
                    self.plotwidget.eventList.append(event)
                    print('Event {}, eventList has {} events [short terminating branch removed]'.format(success, len(self.plotwidget.eventList)))
                    self.plotwidget.checkCycle()
                else:
                    print('Event failed')
        
        print('{} short terminating branches detected.'.format(shortTerminatingBranchCounter))
        segmentList = self.plotwidget.segmentList
        segmentIndexUsed = self.plotwidget.segmentIndexUsed
        removeList = self.plotwidget.removeList
        segmentUsed = [segmentList[ii] for ii in segmentIndexUsed if ii not in removeList]
        G = self.plotwidget.G
        resultFolder = self.plotwidget.resultFolder
        eventListFileName = 'eventList.pkl'
        eventListFilePath = os.path.join(resultFolder, eventListFileName)
        with open(eventListFilePath, 'wb') as f:
            pickle.dump(eventList, f, 2)

        print('{} saved to {}.'.format(eventListFileName, resultFolder)) 
        
        segmentListCleanedFileName = 'segmentListCleaned.npz'
        segmentListCleanedFilePath = os.path.join(resultFolder, segmentListCleanedFileName)
        np.savez_compressed(segmentListCleanedFilePath, segmentList=segmentUsed)
        print('{} saved to {}.'.format(segmentListCleanedFileName, resultFolder)) 

        graphCleanedFileName = 'graphRepresentationCleaned.graphml'
        graphCleanedFilePath = os.path.join(resultFolder, graphCleanedFileName)
        nx.write_graphml(G, graphCleanedFilePath)
        print('{} saved to {}.'.format(graphCleanedFileName, resultFolder)) 
        
    
    def onShowSegmentButtonClicked(self):
        segmentIndexInput = int(self.segmentIndexBox.text())
        segmentPlotItem = self.plotwidget.items[self.plotwidget.segmentStartIndex + segmentIndexInput]
        segmentPlotItem.setData(color=pg.glColor('b'))
    

def main():
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('Vessel')
    ex = Vessel()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()