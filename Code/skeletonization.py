import sys, os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.signal import convolve
from numpy.linalg import norm
import networkx as nx
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

def loadVolume(volumeFolderPath, volumeName):
    """
    Load nifti files (*.nii or *.nii.gz).

    Parameters
    ----------
    volumeFolderPath : str
        Folder of the volume file.
    volumeName : str
        Name of the volume file.
    
    Returns
    -------
    volume : ndarray
        Volume data in the form of numpy ndarray.
    affine : ndarray
        Associated affine transformation matrix in the form of numpy ndarray.
    """
    volumeFilePath = os.path.join(volumeFolderPath, volumeName)
    volumeImg = nib.load(volumeFilePath)
    volume = volumeImg.get_data()
    shape = volume.shape
    affine = volumeImg.affine
    print('Volume loaded from {} with shape = {}.'.format(volumeFilePath, shape))

    return volume, affine

def saveVolume(volume, affine, path, astype=None):
    """
    Save the given volume to the specified location in specified data type.

    Parameters
    ----------
    volume : ndarray
        Volume data to be saved.
    affine : ndarray
        The affine transformation matrix associated with the volume.
    path : str
        The absolute path where the volume is going to be saved.
    astype : numpy dtype, optional
        The desired data type of the volume data.
    """
    if astype is None:
        astype = np.uint8
    
    nib.save(nib.Nifti1Image(volume.astype(astype), affine), path)
    print('Volume saved to {} as type {}.'.format(path, astype))

def labelVolume(volume, minSize=1, maxHop=3):
    """
    Partition the volume into several connected components and attach labels.

    Parameters
    ----------
    volume : ndarray
        Volume to be partitioned.
    minSize : int, optional
        The connected component that is less than this size will be disgarded.
    maxHop : int, optional
        Controls how neighboring voxels are defined. See `label` doc for details.
    
    Returns
    -------
    labeled : ndarray
        The partitioned and labeled volume. Each connected component has a label (a positive integer) and the background
        is labeled as 0.
    labelResult : list
        In the form of [[label1, size1], [label2, size2], ...]
    """
    labeled, maxNum = label(volume, return_num=True, connectivity=maxHop)
    counts = np.bincount(labeled.ravel())
    countLoc = np.nonzero(counts)[0]
    sizeList = counts[countLoc]
    labelResult = list(zip(countLoc[sizeList >= minSize], sizeList[sizeList >= minSize]))
    # print(labelResult)
    # print('Total segments: {}'.format(np.count_nonzero(sizeList >= minSize)))
    return labeled, labelResult

def analyze(vesselVolumeMask, baseFolder):
    """
    Main function to provoke the skeletonization process. Note that here I am using the docker version of the code. If
    you have already downloaded the original C++ code and successfully compiled it, then you can run that compiled code
    instead of this one.
    """
    vesselVolumeMask = vesselVolumeMask.astype(np.uint8)
    vesselVolumeMask[vesselVolumeMask != 0] = 1
    vesselVolumeMask = np.swapaxes(vesselVolumeMask, 0, 2)
    shape = vesselVolumeMask.shape

    vesselVolumeMaskLabeled, vesselVolumeMaskLabelResult = labelVolume(vesselVolumeMask, minSize=1)
    directory = os.path.join(baseFolder, 'skeletonizationResult')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Directory {} created.'.format(directory))
    
    vesselVolumeMaskLabelInfoFilename = 'vesselVolumeMaskLabelInfo.npz'
    vesselVolumeMaskLabelInfoFilePath = os.path.join(directory, vesselVolumeMaskLabelInfoFilename)
    np.savez_compressed(vesselVolumeMaskLabelInfoFilePath, vesselVolumeMaskLabeled=vesselVolumeMaskLabeled, vesselVolumeMaskLabelResult=vesselVolumeMaskLabelResult)
    print('{} saved to {}.'.format(vesselVolumeMaskLabelInfoFilename, vesselVolumeMaskLabelInfoFilePath))
    
    # directory2 = directory + 'labelNum=' + str(labelNum) + '/'
    # if not os.path.exists(directory2):
    #     os.makedirs(directory2)
    # with open(directory2 + 'BB.txt', 'w') as f1:
    #     f1.write('1\n')
    #     f1.write('{} {} {}\n'.format(0, 0, 0))
    #     f1.write('{} {} {}'.format(*shape))
    #'''
    BBFilePath = os.path.join(directory, 'BB.txt')
    f1 = open(BBFilePath, 'w')
    f1.write('1\n')
    f1.write('{} {} {}\n'.format(0, 0, 0))
    f1.write('{} {} {}'.format(*shape))
    f1.close()
    
    vesselCoords = np.array(np.where(vesselVolumeMask)).T
    xyzFilePath = os.path.join(directory, 'xyz.txt')
    np.savetxt(xyzFilePath, vesselCoords, fmt='%1u')
    f2 = open(xyzFilePath, "r")
    contents = f2.readlines()
    f2.close()
    
    contents.insert(0, '{}\n'.format(len(vesselCoords)))
    
    f2 = open(xyzFilePath, "w")
    contents = "".join(contents)
    f2.write(contents)
    f2.close()
    #'''

    #'''
    currentPlatform = platform.system()
    print('Current platform is {}.'.format(currentPlatform))
    if currentPlatform == 'Windows':
        cmd = '"C:/Program Files/Docker/Docker/Resources/bin/docker.exe" run -v ' + '"' + directory + '"' + ':/write_directory -e THRESH=1e-12 -e CC_FLAG=1 -e CONVERSION_TYPE=1 amytabb/curveskel-tabb-medeiros-2018-docker'
    elif currentPlatform == 'Darwin':
        cmd = 'docker run -v ' + '"' + directory + '"' + ':/write_directory -e THRESH=1e-12 -e CC_FLAG=1 -e CONVERSION_TYPE=1 amytabb/curveskel-tabb-nih-aug2018-docker2'
    elif currentPlatform == 'Linux':
        cmd = '/usr/local/bin/docker run -v ' + '"' + directory + '"' + ':/write_directory -e THRESH=1e-12 -e CC_FLAG=1 -e CONVERSION_TYPE=1 amytabb/curveskel-tabb-medeiros-2018-docker'
        cmd = 'sudo docker run -v ' + '"' + directory + '"' + ':/write_directory -e THRESH=1e-12 -e CC_FLAG=1 -e CONVERSION_TYPE=1 amytabb/curveskel-tabb-medeiros-2018-docker'
        cmd = 'sudo docker run -v ' + '"' + directory + '"' + ':/write_directory -e THRESH=1e-12 -e CC_FLAG=1 -e CONVERSION_TYPE=1 amytabb/curveskel-tabb-nih-aug2018-docker2'
    
    print('cmd={}'.format(cmd))
    subprocess.call(cmd, shell=True)
    #'''

def combineSkeletonSegments(skeletonSegmentFolderPath):
    """
    Collect and combine the results from the skeletonization.

    Parameters
    ----------
    skeletonSegmentFolderPath : str
        The folder that contains the segments information (result_segments_xyz*.txt).
    
    Returns
    -------
    segmentList : list
        A list containing the segment information. Each sublist represents a segment and each element in the sublist
        represents a centerpoint coordinates.
    """
    segmentList = []
    files = glob.glob(os.path.join(skeletonSegmentFolderPath, 'result_segments_xyz*.txt'))
    for segmentFile in files:
        result = readSegmentFile(segmentFile)
        segmentList += result

    return segmentList

def readSegmentFile(segmentFile):
    """
    Parse the segment files (result_segments_xyz*.txt) and return segments information in a list.

    Parameters
    ----------
    segmentFile : str
        Path to the segment file.
    
    Returns
    -------
    segmentList : list
        A list containing the segment information. Each sublist represents a segment and each element in the sublist
        represents a centerpoint coordinates.
    """
    isFirstLine = True
    isSegmentLength = True
    segmentList = []
    with open(segmentFile) as f:
        for line in f:
            if isFirstLine:
                numOfSegments = int(line)
                isFirstLine = False
            else:
                if isSegmentLength:
                    segmentLength = int(line)
                    isSegmentLength = False
                    segmentCounter = 1
                    segment = []
                else:
                    if segmentCounter <= segmentLength:
                        voxel = tuple([int(x) for x in line.split(' ')])
                        segment.append(voxel[::-1])
                        segmentCounter += 1
                    else:
                        segmentCounter += 1
                        isSegmentLength = True
                        segmentList.append(segment)
                        assert(len(segment) == segmentLength)
    
    return segmentList

# def drawSegments(segmentList):
#     pass

def processSegments(segmentList, shape):
    """
    Re-partition the segments so that each segment is a simple branch, i.e., it does not contain bifurcation point
    unless at the two ends.

    Note that this function might be replaced by another more concise function `getSegmentList`.

    Parameters
    ----------
    segmentList : list
        A list containing the segment information. Each sublist represents a segment and each element in the sublist
        represents a centerpoint coordinates.
    shape : tuple
        Shape of the vessel volume (used for ploting).
    
    Returns
    -------
    G : NetworkX graph
        A graph in which each node represents a centerpoint and each edge represents a portion of a vessel branch.
    segmentList : list
        A list containing the segment information. Each sublist represents a segment and each element in the sublist
        represents a centerpoint coordinates.
    errorSegments : list
        A list that contains segments that cannot be fixed.
    """
    ## Import pyqtgraph ##
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    ## Init ##
    app = pg.QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 800
    w.setGeometry(0, 110, 1600, 900)
    offset = np.array(shape) / (-2.0)

           
    G = nx.Graph()
    colorList = [pg.glColor('r'), pg.glColor('g'), pg.glColor('b'), pg.glColor('c'), pg.glColor('m'), pg.glColor('y')]
    colorPointer = 0
    skeleton = np.full(shape, 0)
    for segment in segmentList:
        # G.add_path(list(map(tuple, segment)))
        G.add_path(segment)
        segmentCoords = np.array(segment)
        skeleton[tuple(segmentCoords.T)] = 1
        # segmentCoordsView = segmentCoords + offset
        # aa = gl.GLLinePlotItem(pos=segmentCoordsView, color=colorList[colorPointer], width=3)
        # w.addItem(aa)
        # colorPointer = colorPointer + 1 if colorPointer < len(colorList) - 1 else 0
        
    
    # skeletonCoords = np.array(np.where(skeleton)).T
    # skeletonCoordsView = (skeletonCoords + offset) * affineTransform
    # aa = gl.GLScatterPlotItem(pos=skeletonCoordsView, size=5)
    # w.addItem(aa)
    
    # w.show()

    voxelDegrees = np.array([v for _, v in G.degree(G.nodes())])
    maxVoxelDegree = np.amax(voxelDegrees)
    voxelDegreesZippedResult = list(zip(np.arange(maxVoxelDegree + 1), np.bincount(voxelDegrees)))
    print('Voxel degree distribution is \n{}'.format(voxelDegreesZippedResult))
    print('Number of cycles is {}'.format(len(nx.cycle_basis(G))))

    # Remove duplicate segments
    keepList = np.full((len(segmentList),), True)
    duplicateCounter = 0
    for idx, seg in enumerate(segmentList):
        for idx2, seg2 in enumerate(segmentList[idx + 1:]):
            if seg == seg2 or seg == seg2[::-1]:
                keepList[idx + idx2] = False
                duplicateCounter += 1
    
    segmentList = [seg for idx, seg in enumerate(segmentList) if keepList[idx]]
    print('{} duplicate segments removed!'.format(duplicateCounter))
    
    # Cut segments into sub-pieces if there are bifurcation points in the middle
    extraSegments = []
    keepList = np.full((len(segmentList),), True)
    for idx, segment in enumerate(segmentList):
        voxelDegrees = np.array([v for _, v in G.degree(segment)])
        if len(voxelDegrees) >= 3:
            if voxelDegrees[0] == 2 or voxelDegrees[-1] == 2 or (not np.all(voxelDegrees[1:-1] == 2)):
                keepList[idx] = False
                locs = np.nonzero(voxelDegrees != 2)[0]
                if voxelDegrees[0] == 2:
                    locs = np.hstack((0, locs))
                
                if voxelDegrees[-1] == 2:
                    locs = np.hstack((locs, len(voxelDegrees)))
                
                newSegments = []
                for ii in range(len(locs) - 1):
                    newSegments.append(segment[locs[ii]:(locs[ii + 1] + 1)])
                
                extraSegments += newSegments
    
    segmentList = [seg for idx, seg in enumerate(segmentList) if keepList[idx]]
    segmentList += extraSegments
    
    # Remove duplicate segments again
    keepList = np.full((len(segmentList),), True)
    duplicateCounter = 0
    for idx, seg in enumerate(segmentList):
        for idx2, seg2 in enumerate(segmentList[idx + 1:]):
            if seg == seg2 or seg == seg2[::-1]:
                keepList[idx + idx2] = False
                duplicateCounter += 1
    
    segmentList = [seg for idx, seg in enumerate(segmentList) if keepList[idx]]
    print('{} duplicate segments removed in the second stage!'.format(duplicateCounter))

    # Remove segment if it is completely contained in another segment
    # keepList = np.full((len(segmentList),), True)
    # sublistCounter = 0
    # for idx, seg in enumerate(segmentList):
    #     for idx2, seg2 in enumerate(segmentList[idx + 1:]):
    #         if contains(seg, seg2):
    #             keepList[idx] = False
    #             sublistCounter += 1
    #         elif contains(seg2, seg):
    #             keepList[idx + idx2] = False
    #             sublistCounter += 1
    
    # segmentList = [seg for idx, seg in enumerate(segmentList) if keepList[idx]]
    # print('{} sublist segments removed!'.format(sublistCounter))
    
    # Treat the segment if either end is not correct
    hasInvalidSegments = False
    for idx, segment in enumerate(segmentList):
        voxelDegrees = np.array([v for _, v in G.degree(segment)])
        if len(voxelDegrees) == 2:
            if voxelDegrees[0] == 2 or voxelDegrees[-1] == 2:
                # print('Degrees on either end is 2: {}'.format(voxelDegrees))
                hasInvalidSegments = True
        elif len(voxelDegrees) > 2:
            if voxelDegrees[0] == 2 or voxelDegrees[-1] == 2 or np.any(voxelDegrees[1:-1] != 2):
                # print('Degrees not correct: {}'.format(voxelDegrees))
                hasInvalidSegments = True
    
    if not hasInvalidSegments:
        drawSegments(segmentList, shape)
        print('No errors!')
        errorSegments = []
        return G, segmentList, errorSegments
    
    iterCounter = 1
    while hasInvalidSegments:
        print('\n\nIter={}'.format(iterCounter))
        keepList = np.full((len(segmentList),), True)
        extraSegments = []
        for idx, segment in enumerate(segmentList):
            if keepList[idx]:
                voxelDegrees = np.array([v for _, v in G.degree(segment)])
                if voxelDegrees[0] == 2 and voxelDegrees[-1] == 2:
                    print('Both end have 2 neighbours')
                elif voxelDegrees[0] == 2 or voxelDegrees[-1] == 2:
                    # print('Degrees on either end is 2: {}'.format(voxelDegrees))
                    # pass
                    # segmentCoords = np.array(segment)
                    if voxelDegrees[0] == 2:
                        otherSegmentInfo = [(idx2, seg) for idx2, seg in enumerate(segmentList) if (seg[0] == segment[0] or seg[-1] == segment[0]) and keepList[idx2] and idx != idx2]
                        if len(otherSegmentInfo) != 0:
                            if len(otherSegmentInfo) > 1:
                                # print(contains(segment, otherSegmentInfo[0][1]), contains(otherSegmentInfo[1][1], segment))
                                otherSegmentInfoTemp = []
                                for idx2, seg in otherSegmentInfo:
                                    if contains(segment, seg) or contains(segment[::-1], seg):
                                        keepList[idx] = False
                                        continue
                                    elif contains(seg, segment) or contains(seg[::-1], segment):
                                        keepList[idx2] = False
                                        otherSegmentInfoTemp.append((idx2, seg))
                                
                                otherSegmentInfo = otherSegmentInfoTemp
                                # otherSegmentInfo = [segInfo for segInfo in otherSegmentInfo if not (contains(segment, segInfo[1]) or contains(segInfo[1], segment))]
                                if len(otherSegmentInfo) > 1:
                                    print('More than one other segments found!')
                                    print('Current segment ({}) is {} ({})'.format(idx, segment, voxelDegrees))
                                    for otherSegmentIdx, otherSegment in otherSegmentInfo:
                                        otherSegmentVoxelDegrees = np.array([v for _, v in G.degree(otherSegment)])
                                        print('Idx = {}: {} ({})'.format(otherSegmentIdx, otherSegment, otherSegmentVoxelDegrees))
                                elif len(otherSegmentInfo) == 1:
                                    otherSegmentIdx, otherSegment = otherSegmentInfo[0]
                                else:
                                    print('No valid other segments found!')
                                    continue
                            else:
                                otherSegmentIdx, otherSegment = otherSegmentInfo[0]
                                if contains(segment, otherSegment) or contains(segment[::-1], otherSegment):
                                    keepList[idx] = False
                                    continue
                                elif contains(otherSegment, segment) or contains(otherSegment[::-1], segment):
                                    keepList[otherSegmentIdx] = False
                                    continue

                            newSegment = otherSegment + segment[1:] if otherSegment[-1] == segment[0] else otherSegment[::-1] + segment[1:]
                            if not validateSegment(G, newSegment):
                                newSegmentVoxelDegrees = np.array([v for _, v in G.degree(newSegment)])
                                print('Old degree is {} () and new degree is {} ()'.format(voxelDegrees, newSegmentVoxelDegrees))
                            else:
                                print('Two segments ({} and {}) merged together!'.format(idx, otherSegmentIdx))
            
                            extraSegments.append(newSegment)
                            keepList[idx] = False
                            keepList[otherSegmentIdx] = False
                        else:
                            print('Could not find other segments for segment({}) {} with degrees {}'.format(idx, segment, voxelDegrees))
                            possibleSegmentsInfo = [(idx2, seg) for idx2, seg in enumerate(segmentList) if (seg[0] == segment[0] or seg[-1] == segment[0]) and idx != idx2]
                            print('Possible segments: {}'.format(len(possibleSegmentsInfo)))
        
                    elif voxelDegrees[-1] == 2:
                        otherSegmentInfo = [(idx2, seg) for idx2, seg in enumerate(segmentList) if (seg[0] == segment[-1] or seg[-1] == segment[-1]) and keepList[idx2] and idx != idx2]
                        if len(otherSegmentInfo) != 0:
                            if len(otherSegmentInfo) > 1:
                                # print(contains(segment, otherSegmentInfo[0][1]), contains(otherSegmentInfo[1][1], segment))
                                otherSegmentInfoTemp = []
                                for idx2, seg in otherSegmentInfo:
                                    if contains(segment, seg) or contains(segment[::-1], seg):
                                        keepList[idx] = False
                                        continue
                                    elif contains(seg, segment) or contains(seg[::-1], segment):
                                        keepList[idx2] = False
                                        otherSegmentInfoTemp.append((idx2, seg))
                                
                                otherSegmentInfo = otherSegmentInfoTemp
                                # otherSegmentInfo = [segInfo for segInfo in otherSegmentInfo if not (contains(segment, segInfo[1]) or contains(segInfo[1], segment))]
                                if len(otherSegmentInfo) > 1:
                                    print('More than one other segments found!')
                                    print('Current segment ({}) is {} ({})'.format(idx, segment, voxelDegrees))
                                    for otherSegmentIdx, otherSegment in otherSegmentInfo:
                                        otherSegmentVoxelDegrees = np.array([v for _, v in G.degree(otherSegment)])
                                        print('Idx = {}: {} ({})'.format(otherSegmentIdx, otherSegment, otherSegmentVoxelDegrees))
                                elif len(otherSegmentInfo) == 1:
                                    otherSegmentIdx, otherSegment = otherSegmentInfo[0]
                                else:
                                    print('No valid other segments found!')
                                    continue
                            else:
                                otherSegmentIdx, otherSegment = otherSegmentInfo[0]
                                if contains(segment, otherSegment) or contains(segment[::-1], otherSegment):
                                    keepList[idx] = False
                                    continue
                                elif contains(otherSegment, segment) or contains(otherSegment[::-1], segment):
                                    keepList[otherSegmentIdx] = False
                                    continue

                            newSegment = segment[:-1] + otherSegment if otherSegment[0] == segment[-1] else segment[:-1] + otherSegment[::-1]
                            if not validateSegment(G, newSegment):
                                newSegmentVoxelDegrees = np.array([v for _, v in G.degree(newSegment)])
                                print('Old degree is {} () and new degree is {} ()'.format(voxelDegrees, newSegmentVoxelDegrees))
                            else:
                                print('Two segments ({} and {}) merged together!'.format(idx, otherSegmentIdx))
        
                            extraSegments.append(newSegment)
                            keepList[idx] = False
                            keepList[otherSegmentIdx] = False
                        else:
                            print('Could not find other segments for segment({}) {} with degrees {}'.format(idx, segment, voxelDegrees))
                            possibleSegmentsInfo = [(idx2, seg) for idx2, seg in enumerate(segmentList) if (seg[0] == segment[-1] or seg[-1] == segment[-1]) and idx != idx2]
                            print('Possible segments: {}'.format(len(possibleSegmentsInfo)))
                                
        segmentList = [segment for idx, segment in enumerate(segmentList) if keepList[idx]]
        segmentList += extraSegments
        hasInvalidSegments = False
        errorSegments = []
        for idx, segment in enumerate(segmentList):
            voxelDegrees = np.array([v for _, v in G.degree(segment)])
            if len(voxelDegrees) == 2:
                if voxelDegrees[0] == 2 or voxelDegrees[-1] == 2:
                    print('Degrees on either end is 2: {}'.format(voxelDegrees))
                    hasInvalidSegments = True
                    errorSegments.append(segment)
            elif len(voxelDegrees) > 2:
                if voxelDegrees[0] == 2 or voxelDegrees[-1] == 2 or np.any(voxelDegrees[1:-1] != 2):
                    print('Degrees not correct: {}'.format(voxelDegrees))
                    hasInvalidSegments = True
                    errorSegments.append(segment)

        print('hasInvalidSegments = {}'.format(hasInvalidSegments))
        iterCounter += 1
        if len(extraSegments) == 0:
            hasInvalidSegments = False
            print('While loop aborted because there is no change in segments!')
    
    for errorSegment in errorSegments:
        segmentList.remove(errorSegment)
    
    # np.savez_compressed(directory + 'segmentList.npz', segmentList=segmentList)
    # if partIdx != 10:
    #     nib.save(nib.Nifti1Image(skeleton.astype(np.int16), vesselImg.affine), directory + skeletonNamePartial + str(partIdx) + '.nii.gz')
    # else:
    #     nib.save(nib.Nifti1Image(skeleton.astype(np.int16), vesselImg.affine), directory + skeletonNameTotal + '.nii.gz')
    
    # nx.write_graphml(G, directory + 'graphRepresentation.graphml')

    # drawAbstractGraph(offset, segmentList)
    # drawAbstractGraph(offset, errorSegments)
    
    print(errorSegments)


    return G, segmentList, errorSegments

def sublist(ls1, ls2):
    '''
    >>> sublist([], [1,2,3])
    True
    >>> sublist([1,2,3,4], [2,5,3])
    True
    >>> sublist([1,2,3,4], [0,3,2])
    False
    >>> sublist([1,2,3,4], [1,2,5,6,7,8,5,76,4,3])
    False
    '''
    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True


def contains(lst1, lst2):
    lst1, lst2 = (lst2, lst1) if len(lst1) > len(lst2) else (lst1, lst2)
    if lst1[0] in lst2:
        startLoc = lst2.index(lst1[0])
    else:
        return False
    
    if lst1[-1] in lst2:
        endLoc = lst2.index(lst1[-1])
    else:
        return False
    
    if startLoc < endLoc:
        if lst1 == lst2[startLoc:(endLoc + 1)]:
            return True
        else:
            return False
    else:
        if lst1 == lst2[endLoc:(startLoc + 1)][::-1]:
            return True
        else:
            return False

def validateSegment(G, segment):
    """
    Check whether a segment is a simple branch.

    Parameters
    ----------
    G : NetworkX graph
        A graph in which each node represents a centerpoint and each edge represents a portion of a vessel branch.
    segment : list
        A list containing the coordinates of the centerpoints of a segment.
    
    Returns
    -------
    result : bool
        If True, the segment is a simple branch.
    """
    voxelDegrees = np.array([v for _, v in G.degree(segment)])
    if voxelDegrees[0] != 2 and voxelDegrees[-1] != 2:
        if len(voxelDegrees) == 2:
            result = True
        elif len(voxelDegrees) > 2:
            if np.all(voxelDegrees[1:-1] == 2):
                result = True
            else:
                result = False
        else:
            print('Error! Segment with length 1 found!')
            result = False
    else:
        result = False
    
    return result

def drawSegments(segmentList, shape):
    """
    Plot all the segments in `segmentList`. Try to assign different colors to the segments connected to the same node.

    Parameters
    ----------
    segmentList : list
        A list containing the segment information. Each sublist represents a segment and each element in the sublist
        represents a centerpoint coordinates.
    shape : tuple
        Shape of the vessel volume (used for ploting).
    """
    ## Import pyqtgraph ##
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    ## Init ##
    app = pg.QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 800
    w.setGeometry(0, 110, 1600, 900)
    offset = np.array(shape) / (-2.0)

    colorList = [pg.glColor('r'), pg.glColor('g'), pg.glColor('b'), pg.glColor('c'), pg.glColor('m'), pg.glColor('y')]
    colorNames = ['Red', 'Green', 'Blue', 'Cyan', 'Magneta', 'Yellow']
    numOfColors = len(colorList)
    nodeColorDict = {}
    for segment in segmentList:
        startVoxel = segment[0]
        endVoxel = segment[-1]
        if startVoxel in nodeColorDict and endVoxel in nodeColorDict: # and endVoxel in [voxel for voxel, _ in nodeColorDict[startVoxel]]:
            nodeColorDict[startVoxel].append([endVoxel, -1])
            nodeColorDict[endVoxel].append([startVoxel, -1])
        else:
            if startVoxel not in nodeColorDict:
                nodeColorDict[startVoxel] = [[endVoxel, -1]]
            else:
                nodeColorDict[startVoxel].append([endVoxel, -1])
            
            if endVoxel not in nodeColorDict:
                nodeColorDict[endVoxel] = [[startVoxel, -1]]
            else:
                nodeColorDict[endVoxel].append([startVoxel, -1])
    
        existingColorsInStart = [colorCode for _, colorCode in nodeColorDict[startVoxel]]
        existingColorsInEnd = [colorCode for _, colorCode in nodeColorDict[endVoxel]]
        availableColors = [colorCode for colorCode in range(numOfColors) if colorCode not in existingColorsInStart and colorCode not in existingColorsInEnd]
        # print('color in start: {} and color in end: {}'.format(existingColorsInStart, existingColorsInEnd))
        chosenColor = availableColors[0] if len(availableColors) != 0 else 0
        nodeColorDict[startVoxel][-1][1] = chosenColor
        nodeColorDict[endVoxel][-1][1] = chosenColor


        segmentCoords = np.array(segment)
        aa = gl.GLLinePlotItem(pos=segmentCoords, color=colorList[chosenColor], width=3)
        aa.translate(*offset)
        w.addItem(aa)
         
    w.show()
    pg.QtGui.QApplication.exec_()
    # sys.exit(app.exec_())

def main():
    start_time = timeit.default_timer()
    baseFolder = os.path.abspath(os.path.dirname(__file__))

    ## Load existing volume ##
    vesselVolumeMaskFolderPath = baseFolder
    vesselVolumeMaskFileName = 'vesselVolumeMask.nii.gz'
    vesselVolumeMask, vesselVolumeMaskAffine = loadVolume(vesselVolumeMaskFolderPath, vesselVolumeMaskFileName)
    
    ## Skeletonization ##
    # analyze(vesselVolumeMask, baseFolder)
    
    skeletonSegmentFolderPath = os.path.join(baseFolder, 'skeletonizationResult/segments_by_cc')
    segmentListRough = combineSkeletonSegments(skeletonSegmentFolderPath)

    shape = vesselVolumeMask.shape
    # drawSegments(segmentListRough, shape)

    G, segmentList, errorSegments = processSegments(segmentListRough, shape=shape)
    # drawSegments(segmentList, shape)
    G = nx.Graph()
    segmentIndex = 0
    for segment in segmentList:
        G.add_path(segment, segmentIndex=segmentIndex)
        segmentIndex += 1
    
    ## Save graph representation ##
    graphFileName = 'graphRepresentation.graphml'
    graphFilePath = os.path.join(baseFolder, graphFileName)
    nx.write_graphml(G, graphFilePath)
    print('{} saved to {}.'.format(graphFileName, graphFilePath))

    ## Save segmentList ##
    segmentListFileName = 'segmentListRough.npz'
    segmentListFilePath = os.path.join(baseFolder, segmentListFileName)
    np.savez_compressed(segmentListFilePath, segmentList=segmentList)
    print('{} saved to {}.'.format(segmentListFileName, segmentListFilePath))

    ## Save skeleton.nii.gz ##
    skeleton = np.zeros_like(vesselVolumeMask)
    for segment in segmentList:
        skeleton[tuple(np.array(segment).T)] = 1
    
    skeletonFileName = 'skeleton.nii.gz'
    skeletonFilePath = os.path.join(baseFolder, skeletonFileName)
    saveVolume(skeleton, vesselVolumeMaskAffine, skeletonFilePath, astype=np.uint8)


    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

if __name__ == "__main__":
    main() 

