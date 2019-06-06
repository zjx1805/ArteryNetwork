from __future__ import division
import numpy as np
import nibabel as nib
import nrrd
import timeit

A = (2*np.pi)**(-0.5)
# H = 1

def variationalRegionGrowing(dataArray, valueMap, H=2.25, maxSegmentSize=5000):
    """
    Implements the variational region growing algorithm described in this [paper](https://ieeexplore.ieee.org/document/7096420).

    Parameters
    ----------
    dataArray : ndarray
        The data volume to which the algorithm is applied.
    valueMap : ndarray
        An array of the same size as `dataArray` indicating initial settings.  
        The meaning of the numbers is as follows:  
        0: inside, 1: innerbnd, 2:outerbnd, 3: outside, 4: excluded
    H : float
        A parameter that controls the size of segmentation. Larger H leads to smaller segmentation.
    maxSegmentSize : int
        If the number of voxels segmented exceeds this limit, the program stops and returns the current result.

    Returns
    -------
    segmented : ndarray
        An array in which each line is the coordinate of the segmented voxel.
    segmentedMap : ndarray
        An array of the same size as dataArray, in which 1 means segmented voxels and 0 means background.
    valueMap : ndarray
        An array of the same size as `dataArray` indicating the current state.  
        The meaning of the numbers is as follows:  
        0: inside, 1: innerbnd, 2:outerbnd, 3: outside, 4: excluded
    """
    start_time = timeit.default_timer()
    # A = (2*np.pi)**(-0.5)
    dataShape = dataArray.shape
    # valueMap = np.full(dataShape, 3)
    # valueMap[dataArray <= threshold] = 4
    # valueMap[0:5, 0:5, 0:5] = 3
    segmented = np.array(np.where(valueMap == 0)).T
    segmentedMap = np.full(dataArray.shape, 0)
    segmentedMap[tuple(segmented.T)] = 1
    segmented, segmentedMap, valueMap, innerBnd, outerBnd, innerProb, outerProb = update(dataArray, segmented, segmentedMap, valueMap, H)
    allBnd = np.concatenate((innerBnd, outerBnd))
    innerLocs = np.array(np.where((valueMap == 0) | (valueMap == 1))).T
    outerLocs = np.array(np.where((valueMap == 2) | (valueMap == 3))).T
    innerSize = innerLocs.shape[0]
    outerSize = outerLocs.shape[0]
    # probabilityValue = np.ones((max(dataArray), 1)) * (-1)
    # innerProbabilityCounter = np.zeros((max(dataArray), 1))
    # outerProbabilityCounter = np.zeros((max(dataArray), 1))
    iterMax = 200
    iterNum = 1
    while iterNum <= iterMax:
        # flipedPoints = np.array([])
        # flipedPointsList = flipedPoints.tolist()
        # print('Iter {}, [{}, {}, {}, {}]'.format(iterNum, len(innerValues), innerSize, len(outerValues), outerSize))
        # for ii in list(range(len(allBnd))):  # point in allBnd.tolist():
        #     point = allBndList[ii]
        #     innerProbabilitySep = innerProbability[ii]
        #     outerProbabilitySep = outerProbability[ii]
            
        #     innerProbabilityNormalized = innerProbabilitySep / innerSize
        #     outerProbabilityNormalized = outerProbabilitySep / outerSize
        #     if outerProbabilityNormalized > innerProbabilityNormalized:
        #         ratio = 1
        #     elif outerProbabilityNormalized < innerProbabilityNormalized:
        #         ratio = -1
        #     else:
        #         ratio = 0
        #     deltaJ = (1 - 2 * segmentMap[tuple(point)]) * ratio
        #     # print(innerProbability, outerProbability, point, segmentMap[tuple(point)], ratio, deltaJ)
        #     if deltaJ <= 0: # and added < 10:
        #         flipedPointsList.append(point)
        innerProbSum = innerProb[tuple(allBnd.T)]
        outerProbSum = outerProb[tuple(allBnd.T)]
        innerProbNormalized = innerProbSum / innerSize
        outerProbNormalized = outerProbSum / outerSize
        # flipedPointsList = [point for i, point in enumerate(allBndList) 
        #     if (segmentMap[tuple(point)] == 0 and innerProbabilityNormalized[i] > outerProbabilityNormalized[i]) or 
        #     (segmentMap[tuple(point)] == 1 and innerProbabilityNormalized[i] < outerProbabilityNormalized[i]) or
        #     innerProbabilityNormalized[i] == outerProbabilityNormalized[i]]
        mask = np.logical_xor(segmentedMap[tuple(allBnd.T)], innerProbNormalized >= outerProbNormalized)
        flipedPoints = allBnd[mask, :]
        # print('Segmented size = {}'.format(len(segmented)))
        # print(len(segmented), maxSegmentSize, len(segmented) > maxSegmentSize)
        if len(flipedPoints) == 0: # or timeit.default_timer() - start_time >= 120 or len(segmented) >= maxSegmentSize:
            # print('Segmented points are: \n', segmented)
            # print(len(segmented), maxSegmentSize, len(segmented) > maxSegmentSize)
            print('Finished at iteration {}'.format(iterNum))
            print('Total segmented voxels: {}/{}'.format(segmented.shape[0], np.count_nonzero(dataArray)))
            return segmented, segmentedMap, valueMap
        elif timeit.default_timer() - start_time >= 120:
            print('Finished at iteration {} (Max time reached)'.format(iterNum))
            print('Total segmented voxels: {}/{}'.format(segmented.shape[0], np.count_nonzero(dataArray)))
            return segmented, segmentedMap, valueMap
        elif len(segmented) >= maxSegmentSize:
            print('Finished at iteration {} (Max segment size reached)'.format(iterNum))
            print('Total segmented voxels: {}/{}'.format(segmented.shape[0], np.count_nonzero(dataArray)))
            return segmented, segmentedMap, valueMap
        else:
            # flipedPoints = np.array(flipedPointsList)
            # print('Iter ', iterNum, ', flipedPoints = \n', flipedPoints)
            # print('Segmented points are: \n', segmented)
            segmented, segmentedMap, valueMap, innerBnd, outerBnd, innerProb, outerProb = update(dataArray, segmented, segmentedMap, valueMap, H, flipedPoints, innerBnd, outerBnd, innerProb, outerProb)
            # print(innerBnd.shape, outerBnd.shape)
            allBnd = np.concatenate((innerBnd, outerBnd), axis=0)
            # allBndList = allBnd.tolist()
            innerLocs = np.array(np.where((valueMap == 0) | (valueMap == 1))).T
            outerLocs = np.array(np.where((valueMap == 2) | (valueMap == 3))).T
            innerSize = innerLocs.shape[0]
            outerSize = outerLocs.shape[0]
            iterNum += 1
    print('Segmented points are: \n', segmented)
    print('Max iteration reached! Finished at iteration {}'.format(iterNum))
    print('Total segmented voxels: {}/{}'.format(segmented.shape[0], np.count_nonzero(dataArray)))
    return segmented, segmentedMap, valueMap
            

def update(dataArray, segmented, segmentedMap, valueMap, H, flipedPoints=None, innerBnd=None, outerBnd=None, innerProb=None, outerProb=None):
    # 0: inside, 1: innerbnd, 2:outerbnd, 3: outside, 4: excluded 
    dataShape = dataArray.shape # psuedo code
    segmentedList = segmented.tolist()
    
    if flipedPoints is None:
        innerBndList = []
        outerBndList = []
        innerProb = np.zeros(dataShape)
        outerProb = np.zeros(dataShape)
        for point in segmentedList:
            neighbours = get_neighbours(point, shape=dataShape)
            # valueMap[tuple(neighbours.T)] = 3
            valueMap[tuple(neighbours[valueMap[tuple(neighbours.T)] == 4, :].T)] = 3
            for neighbour in neighbours.tolist():
                if segmentedMap[tuple(neighbour)] == 0: # not in segmentedList:
                    if valueMap[tuple(point)] != 1: # not in innerBndList
                        innerBndList.append(point)
                        valueMap[tuple(point)] = 1
                    if valueMap[tuple(neighbour)] != 2: # not in outerBndList
                        outerBndList.append(neighbour)
                        valueMap[tuple(neighbour)] = 2

        innerBnd = np.array(innerBndList)
        outerBnd = np.array(outerBndList)
        innerValues = dataArray[(valueMap == 0) | (valueMap == 1)]
        outerValues = dataArray[(valueMap == 2) | (valueMap == 3)]
        for point in innerBndList + outerBndList:
            innerDiff = innerValues - dataArray[tuple(point)]
            outerDiff = outerValues - dataArray[tuple(point)]
            innerProb[tuple(point)] = np.sum(A * np.exp(-0.5 * H * innerDiff**2))
            outerProb[tuple(point)] = np.sum(A * np.exp(-0.5 * H * outerDiff**2))
    else:
        innerBndList = innerBnd.tolist()
        outerBndList = outerBnd.tolist()
        innerBndListOld = innerBndList.copy()
        outerBndListOld = outerBndList.copy()
        allBndList = innerBndList + outerBndList
        includedPoints = np.array([], dtype=np.int16).reshape(-1, 3)
        newInnerBndList = []
        newOuterBndList = []
        for point in flipedPoints.tolist():
            neighbours = get_neighbours(point, shape=dataShape)
            includedPointsLoc = neighbours[valueMap[tuple(neighbours.T)] == 4, :]
            valueMap[tuple(includedPointsLoc.T)] = 3
            includedPoints = np.concatenate((includedPoints, includedPointsLoc.reshape(-1, 3)))
            if valueMap[tuple(point)] == 1: # originally is inner bound
                innerBndList.remove(point)
                segmentedList.remove(point)
                segmentedMap[tuple(point)] = 0
                valueMap[tuple(point)] = 2
                outerBndList.append(point)
                for neighbour in neighbours.tolist():
                    neighbours2 = get_neighbours(neighbour, shape=dataShape)
                    includedPointsLoc = neighbours2[valueMap[tuple(neighbours2.T)] == 4, :]
                    valueMap[tuple(includedPointsLoc.T)] = 3
                    includedPoints = np.concatenate((includedPoints, includedPointsLoc.reshape(-1, 3)))
                    if valueMap[tuple(neighbour)] == 3: # is outside:
                        pass #not possible
                    elif valueMap[tuple(neighbour)] == 2: # in outerBnd:
                        # neighbours2 = get_neighbours(neighbour, shape=dataShape)
                        # valueMap[tuple(neighbours2[valueMap[tuple(neighbours2.T)] == 4, :].T)] = 3
                        if not any(valueMap[tuple(neighbours2.T)] == 1):
                            valueMap[tuple(neighbour)] = 3 # outerBnd -> outside
                            outerBndList.remove(neighbour)
                            innerProb[tuple(neighbour)] = 0
                            outerProb[tuple(neighbour)] = 0
                    elif valueMap[tuple(neighbour)] == 1: # in innerBnd:
                        pass
                    else: # in inside
                        valueMap[tuple(neighbour)] = 1 # inside -> innerBnd
                        newInnerBndList.append(neighbour)
                        innerBndList.append(neighbour)
                        # print('inside->innerBnd', neighbour)
            elif valueMap[tuple(point)] == 2: # flipped point is originally outer bound
                outerBndList.remove(point)
                segmentedList.append(point)
                segmentedMap[tuple(point)] = 1
                valueMap[tuple(point)] = 1
                # print('OuterBnd->InnerBnd', point)
                innerBndList.append(point)
                for neighbour in neighbours.tolist():
                    neighbours2 = get_neighbours(neighbour, shape=dataShape)
                    includedPointsLoc = neighbours2[valueMap[tuple(neighbours2.T)] == 4, :]
                    valueMap[tuple(includedPointsLoc.T)] = 3
                    includedPoints = np.concatenate((includedPoints, includedPointsLoc.reshape(-1, 3)))
                    if valueMap[tuple(neighbour)] == 3: # is outside:
                        valueMap[tuple(neighbour)] = 2 # outside -> outerBnd
                        newOuterBndList.append(neighbour)
                        outerBndList.append(neighbour)
                        # neighbours2 = get_neighbours(neighbour, shape=dataShape)
                        # if neighbor's neighbour is 4, 4->3
                        # valueMap[tuple(neighbours2[valueMap[tuple(neighbours2.T)] == 4, :].T)] = 3 
                    elif valueMap[tuple(neighbour)] == 2: # in outerBnd:
                        pass
                    elif valueMap[tuple(neighbour)] == 1: # in innerBnd:
                        # neighbours2 = get_neighbours(neighbour, shape=dataShape)
                        # valueMap[tuple(neighbours2[valueMap[tuple(neighbours2.T)] == 4, :].T)] = 3
                        # print('neighbour=',neighbour, 'neighbours2=',neighbours2, 'value=', valueMap[tuple(neighbours2.T)])
                        if not any(valueMap[tuple(neighbours2.T)] == 2):
                            valueMap[tuple(neighbour)] = 0 # innerBnd -> inside
                            # print('innerBnd->inside', neighbour)
                            innerBndList.remove(neighbour)
                            innerProb[tuple(neighbour)] = 0
                            outerProb[tuple(neighbour)] = 0
                    else: # in inside
                        pass # not possible
        
        innerAdded = dataArray[tuple(flipedPoints[valueMap[tuple(flipedPoints.T)] == 1, :].T)] # new 1s in segmented
        outerAdded = dataArray[tuple(flipedPoints[valueMap[tuple(flipedPoints.T)] == 2, :].T)] # removed 1s in segmented
        # print(includedPoints)
        addedPoints = dataArray[tuple(includedPoints.T)]
        for point in innerBndList + outerBndList:
            innerDiff = innerAdded - dataArray[tuple(point)] 
            outerDiff = outerAdded - dataArray[tuple(point)]
            addedDiff = addedPoints - dataArray[tuple(point)]
            innerCorrection = np.sum(A * np.exp(-0.5 * H * innerDiff**2)) 
            outerCorrection = np.sum(A * np.exp(-0.5 * H * outerDiff**2)) 
            addedCorrection = np.sum(A * np.exp(-0.5 * H * addedDiff**2))
            innerProb[tuple(point)] += innerCorrection
            innerProb[tuple(point)] -= outerCorrection
            outerProb[tuple(point)] -= innerCorrection
            outerProb[tuple(point)] += outerCorrection
            outerProb[tuple(point)] += addedCorrection
        
        innerValues = dataArray[(valueMap == 0) | (valueMap == 1)]
        outerValues = dataArray[(valueMap == 2) | (valueMap == 3)]
        for point in (newInnerBndList + newOuterBndList):
            innerDiff = innerValues - dataArray[tuple(point)]
            outerDiff = outerValues - dataArray[tuple(point)]
            innerProb[tuple(point)] = np.sum(A * np.exp(-0.5 * H * innerDiff**2))
            outerProb[tuple(point)] = np.sum(A * np.exp(-0.5 * H * outerDiff**2))
    
        innerBnd = np.array(innerBndList)
        outerBnd = np.array(outerBndList)
        segmented = np.array(segmentedList)
    
    return segmented, segmentedMap, valueMap, innerBnd, outerBnd, innerProb, outerProb

def get_neighbours(p, exclude_p=True, shape=None):
    ndim = len(p)
    # generate an (m, ndims) array containing all combinations of 0, 1, 2
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T
    # print(offset_idx)
    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)
    # offsets = offsets[np.sum(np.absolute(offsets), axis=1)<=1, :]
    # print(offsets)
    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets    # apply offsets to p
    # optional: exclude out-of-bounds indices
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]

    return neighbours

def test_StraightLine():
    volume = np.zeros((50, 50, 150), dtype=int)
    datamask = np.s_[20:22, 20:22, 20:40]
    volume[datamask] = 1
    valueMap = np.full(volume.shape, 3)
    valueMap[20:22, 20:22, 22:25] = 0
    segmented, segmentedMap, valueMap = variationalRegionGrowing(volume, valueMap)
    if all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) == len(segmented):
        print('Straight line test passed!')
    elif all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) != len(segmented):
        print('Straight line test partially failed: Segmented volume not complete!')
    elif not all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) == len(segmented):
        print('Straight line test partially failed: Wrong segments included!')
    else:
        print('Straight line test failed!')

def test_Sphere():
    x, y, z = np.mgrid[:50, :50, :50]
    volume = ((x - 25)**2 + (y - 25)**2 + (z - 25)**2 <= 100).astype(int)
    print(np.count_nonzero(volume))
    valueMap = np.full(volume.shape, 3)
    valueMap[25:27, 25:27, 25:27] = 0
    segmented, segmentMap, valueMap = variationalRegionGrowing(volume, valueMap)
    if all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) == len(segmented):
        print('Sphere test passed!')
    elif all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) != len(segmented):
        print('Sphere test partially failed: Segmented volume not complete!')
    elif not all(volume[tuple(segmented.T)]) and np.count_nonzero(volume) == len(segmented):
        print('Sphere test partially failed: Wrong segments included!')
    else:
        print('Sphere test failed!')

# test_StraightLine()
# test_Sphere()



