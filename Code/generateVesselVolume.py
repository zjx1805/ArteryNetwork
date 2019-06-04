import sys, os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import binary_closing, binary_opening
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from operator import itemgetter

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

def refineBrainVolumeMask(brainVolumeMask, rawVolume):
    """
    Manually added the CoW region to the brain volume.

    Parameters
    ----------
    brainVolumeMask : ndarray
        Volume mask of the brain.
    rawVolume : ndarray
        Raw MR image volume.
    
    Returns
    -------
    brainVolumeMaskRefined : ndarray
        The new brain volume mask with CoW region added.
    """
    brainVolumeMaskRefined = copy.deepcopy(brainVolumeMask)
    shape = brainVolumeMask.shape
    brainVolumeMaskRefined[brainVolumeMaskRefined != 0] = 1
    brainVolumeMaskRefined[150:350, 150:350, 0:120] = 1

    return brainVolumeMaskRefined

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

def maskVolume(volume, mask):
    """
    Apply the given volume mask to the given volume.

    Parameters
    ----------
    volume : ndarray
        Volume to be masked.
    mask : ndarray
        Mask to be applied.
    
    Returns
    -------
    newVolume : ndarray
        The resulting masked volume.
    """
    newVolume = copy.deepcopy(volume)
    newVolume[mask == 0] = 0

    return newVolume

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
    labelResult = list(zip(countLoc, sizeList))
    # labelResult = list(zip(countLoc[sizeList >= minSize], sizeList[sizeList >= minSize]))
    # print(labelResult)
    # print('Total segments: {}'.format(np.count_nonzero(sizeList >= minSize)))
    return labeled, labelResult

def main():
    start_time = timeit.default_timer()
    baseFolder = os.path.abspath(os.path.dirname(__file__))
    
    ## Create brain mask from raw volume ##
    rawVolumeFolderPath = baseFolder
    rawVolumeName = '401 3D MRA BRAIN.nii.gz'
    rawVolume, rawVolumeAffine = loadVolume(rawVolumeFolderPath, rawVolumeName)
    
    # brainVolumeRawMaskFolderPath = baseFolder
    # brainVolumeRawMaskName = 'brainVolumeMaskRaw.nii.gz'
    # brainVolumeMaskRaw, _ = loadVolume(brainVolumeRawMaskFolderPath, brainVolumeRawMaskName)

    # brainVolumeMaskRefined = refineBrainVolumeMask(brainVolumeMaskRaw, rawVolume)
    # brainVolumeMaskRefinedFilePath = os.path.join(baseFolder, 'brainVolumeMask.nii.gz')
    # saveVolume(brainVolumeMaskRefined, rawVolumeAffine, brainVolumeMaskRefinedFilePath, astype=np.uint8)

    # brainVolume = maskVolume(rawVolume, brainVolumeMaskRefined)
    # brainVolumeFolderPath = baseFolder
    # brainVolumeName = 'brainVolume.nii.gz'
    # brainVolumeFilePath = os.path.join(brainVolumeFolderPath, brainVolumeName)
    # saveVolume(brainVolume, rawVolumeAffine, brainVolumeFilePath, astype=np.float)

    ## Load existing volume ##
    brainVolumeFolderPath = baseFolder
    brainVolumeName = 'brainVolume.nii.gz'
    brainVolume, _ = loadVolume(brainVolumeFolderPath, brainVolumeName)

    brainVolumeMaskFolderPath = baseFolder
    brainVolumeMaskName = 'brainVolumeMask.nii.gz'
    brainVolumeMask, _ = loadVolume(brainVolumeMaskFolderPath, brainVolumeMaskName)

    vesselnessVolumeFolderPath = baseFolder
    vesselnessVolumeName = 'vesselnessFiltered.nii.gz'
    vesselnessVolume, _ = loadVolume(vesselnessVolumeFolderPath, vesselnessVolumeName)
    vesselnessVolume2 = copy.deepcopy(vesselnessVolume)

    ##
    # Set the vesselness of the voxels that are (1) <= certain voxels to the brain mask boundary and (2) vesselness <= X to 0
    recalculateBrainVolumeMaskDistanceTransform = False
    brainVolumeMaskDistanceTransformFilePath = os.path.join(baseFolder, 'brainVolumeMaskDistanceTransform.npz')
    if os.path.exists(brainVolumeMaskDistanceTransformFilePath) and recalculateBrainVolumeMaskDistanceTransform is False:
        brainVolumeMaskDistanceTransform = np.load(brainVolumeMaskDistanceTransformFilePath)['brainVolumeMaskDistanceTransform']
        print('brainVolumeMaskDistanceTransform.npz loaded from {}.'.format(brainVolumeMaskDistanceTransformFilePath))
    else:
        brainVolumeMaskDistanceTransform = distance_transform_edt(brainVolumeMask)
        np.savez_compressed(brainVolumeMaskDistanceTransformFilePath, brainVolumeMaskDistanceTransform=brainVolumeMaskDistanceTransform)
        print('brainVolumeMaskDistanceTransform.npz saved to {}.'.format(brainVolumeMaskDistanceTransformFilePath))
    
    minVesselness, maxVesselness = np.amin(vesselnessVolume), np.amax(vesselnessVolume)
    mask = np.logical_and(brainVolumeMaskDistanceTransform <= 10, vesselnessVolume2 <= minVesselness + 0.8 * (maxVesselness - minVesselness))
    vesselnessVolume2[mask] = 0
    mask = vesselnessVolume2 <= minVesselness + 0.7 * (maxVesselness - minVesselness)
    vesselnessVolume2[mask] = 0

    # Remove disconnected segments
    vesselnessVolume2[vesselnessVolume2 != 0] = 1
    vesselnessVolume2Labeled, vesselnessVolume2LabelResult = labelVolume(vesselnessVolume2, minSize=10, maxHop=3)

    for labelNum, labelSize in vesselnessVolume2LabelResult:
        if labelSize <= 150:
            vesselnessVolume2[vesselnessVolume2Labeled == labelNum] = 0
    
    # Binary closing to remove gaps
    # Create structure element for morphological operations
    # structureElement = np.zeros((4,4,4))
    # allIndexArray = np.array(np.meshgrid(*[range(axeLength) for axeLength in structureElement.shape])).T.reshape(-1,3)
    # centerLocArray = np.array([length - 1 for length in structureElement.shape]) / (2.0)
    # desiredIndexArray = np.array([index for index in allIndexArray if np.linalg.norm(index - centerLocArray) <= 2])
    # structureElement[tuple(desiredIndexArray.T)] = 1
    # vesselnessVolume2 = binary_closing(vesselnessVolume2, structure=structureElement, iterations=1).astype(int)

    print('Number of voxels in segmentation: {}'.format(np.count_nonzero(vesselnessVolume2)))
    
    # 
    vesselVolumeMaskFolderPath = baseFolder
    vesselVolumeMaskName = 'vesselVolumeMask.nii.gz'
    vesselVolumeMaskFilePath = os.path.join(vesselVolumeMaskFolderPath, vesselVolumeMaskName)
    saveVolume(vesselnessVolume2, rawVolumeAffine, vesselVolumeMaskFilePath, astype=np.uint8)

    # fig = plt.figure(1, figsize=(15, 8))
    # plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.3)
    # labelSizeList = [labelSize for labelNum, labelSize in vesselnessVolume2LabelResult]
    # plt.hist(labelSizeList)
    # # plt.plot(sorted(labelSizeList))

    # plt.show()
    # print(sorted(labelSizeList))

    
    
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed: {} sec'.format(elapsed))

if __name__ == "__main__":
    main()




