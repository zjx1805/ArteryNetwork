# ArteryNetwork

This repository contains the code for our [paper](link):

> Paper info

If you find the code useful for your research, please cite our paper:

> Paper reference

The entire processing pipeline described in our paper is implemented here and is divided into modules. This document will demonstrate, step by step, how to use the pipeline. Note that out of the 4 datasets used (ADAN/BraVa/GBM/Speck), two of them are publicly available (ADAN from [here](http://hemolab.lncc.br/adan-web/) and BraVa from [here](http://cng.gmu.edu/brava/home.php)), the Speck dataset can be obtained by contacting the [author](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27033), the GBM dataset is for private use only and thus cannot be shared (because it contains sensitive patient data).

# Prerequisites
* [3D Slicer](https://download.slicer.org/) (tested with version 4.10.0)
* Python (tested with Python 3.6.5)
* Numpy (tested with v1.14.3)
* Scipy (tested with v1.1.0)
* FSL (tested with v5.0.11): image registration
* Nibabel (tested with v2.3.0): reading and writing nifti files
* Skimage (tested with v0.13.1): image processing
* Pyqtgraph (tested with v0.10.0): displaying volume and GUI related
* Graphviz (tested with v1.3.1): creating graph plot
* Matplotlib (tested with v2.2.2): plotting
* NetworkX (tested with v2.1): graph-related
* Skeletonization ([Amy Tabb](https://data.nal.usda.gov/dataset/code-fast-and-robust-curve-skeletonization-real-world-elongated-objects)): skeletonization of vessel volume


# Pipeline

## MR images

In general, you should be able to see arteries clearly from MRA (Magnetic Resonance Angiography) images with high resolution (<= 600 um). In some cases, 3D T1 images would also do the job (like the Speck data).

## Pre-processing

We use [3D Slicer](https://download.slicer.org/) for displaying the MR image volumes and performing several pre-processing tasks. The following plugins are needed: `SwissSkullStripper`, `SlicerVMTK`, `SegmentEditorExtraEffects` (optional). They can be downloaded from the plugin store within 3D Slicer.

### MR image bias field correction (optional)

Depending on the quality of the MR images, a bias field correction might be necessary. This can be done using the built-in module `N4ITK MRI Bias correction` in 3D Slicer. Adjust the parameters as needed.

### MR image denoising (optional)

Depending on the quality of the MR images, basic image denoising operations might be necessary, e.g., thresholding, filtering. Some of them can be achieved by using the buili-in module `Simple Filters` in 3D Slicer.

### Multiple image registration (optional)

In case you have multiple sets of MR images of a same subject depicting different locations, you may use the `FSL` package to register them into the same space. See the help of `flirt` (linear registration) or `fnirt` (nonlinear registration) if necessary.

### Skull stripping

We use the `Swiss Skull Stripper` module with default parameters to extract the brain volume from the MR images. Note that although the default atlas should give you a nice brain volume, most of the CoW (Circle of Willis) region is not included. Besides, it might miss small areas at the surface of the brain. You can manually add these regions by loading the resulting volume into the `Segment Editor` module and use the `brush` tool to paint the regions you want to add.

## Vessel segmentation

In this step, we segment the brain volume and extract the vessels. If you already have a segmented vessel volume, you may skip this and proceed to the next step.

### Vesselness filtering

We use the `SlicerVMTK` module to perform vesselness filtering on the skull-stripped brain volume obtained from the previous step. Note that the quality of the output of this step is crucial to the steps afterwards, and thus please try your best to obtain a good filtered volume.

We recommend adjusting the parameters `Minimum vessel diameter`, `Maximum vessel diameter`, `Vessel contrast`, `Suppress plates` and `Suppress blobs` manually and for multiple times to see the effect. The filtering process would take a huge amount of memory and time. Based on our experience, a volume of 512\*512\*170 voxels in size would require about 10 GB of RAM and take about 3-5 minutes to run on a i7-6700K CPU. For a even larger volume (e.g., 880\*880\*640 voxels in size as of the Speck dataset), we recommend splitting the whole volume into smaller pieces, perform the veselness filtering on each of them, and then merge them together. Otherwise, the filtering process could take about an hour to complete.

The vesselness filtering algorithm essentially assigns a value to each voxel, representing the probability of it being a part of the vessel. It makes use of the Hessian matrix as well as the corresponding eigenvalues to detect the vessels, which are enlongated objects with roughly circular shapes. Thus it works well if the vessels are of regular shapes and can be seen quite clearly and produces poor result if the vessels are fuzzy, irregular in shape, or at the junctions (bifurcations). In general, if you can see the vessels quite clearly from the filtered volume (probably with simple thresholding), then you are fine and may proceed to the next step.

### Vessel smoothing

The filtered vessel volume obtained from the previous step might still have some rough boundaries or small gaps, and thus we try to fix them in this step. Specifically, we implement the variational region growing algorithm described in this [paper](https://ieeexplore.ieee.org/document/7096420) and apply it on the filterd vessel volume. It should be able to smoothen the surface of the vessels and bridge small gaps (several voxels wide depending on the parameters used) within the volume. The algorithm is implemented in `./code/variationalRegionGrowing.py`.

### Skeletonization

Since the vessels are usually enlongated objects with circular shape, we can simplify the vessel volume by reducing it into a 1D representation: centerlines with corresponding radius information. To do this, we make use of the skeletonziation algorithm described in this [paper](https://data.nal.usda.gov/dataset/code-fast-and-robust-curve-skeletonization-real-world-elongated-objects). In order to suppress the spurious vessel segments in noisy regions, the user-defined acceptance probability *t* was set to `1e^(-12)`. 

The code outputs the centerline information in different formats, and we use the files with names `result_segments_xyz*.txt` under the `segments_by_cc` folder. Each file contains the centerline information of several segments. A segment is defined as a sequence of centerpoint coordinates that represents a portion of an artery. The data in each text file should be interpreted as follows:

1. The first line of the file refers to the number of segments contained in this file.
2. Separate the remaining lines into N parts, where N is the number of segments. The first line of each part is an integer represeting the number of consecutive centerpoint coordinates in this segment. The coordinates are represented in the voxel space.

Ideally, each segment should be a simple branch, i.e., it does not contain bifurcation points unless at the two ends. However, due to some unknown reasons, some of the output segments are not simple branches. This is fixed using the function `processSegments` in the `skeletonization.py`. For each centerpoint, we assign a corresponding radius to it and the radius is obtained by performing distance transform to the segmented vessel volume. After that, we convert the entire network in the form of centerline/radius representation into a graph using the `NetworkX` package to faciliate further analysis.

### Manual correction

Even with the previous automatic processing steps, inevitably there is going to have artifects resulting from either segmentation error or skeletonization error. In this step, we are going to correct them manually using a GUI. The GUI is written in Python and uses PyQt as the framework. Note that since it is quite difficult to directly correct the segmentation error (because the volume is 3D), the GUI only tries to correct the centerlines and the corresponding radius information will be updated with the new centerlines. The ability to directly modify 3D segmentation volume might be added in the future if necessary.

Currently, the GUI supports the following operations:

* Remove: remove the selected simple branch
* Reconnect: connect two simple branches using spline interpolation
* Grow: extend the selected simple branch
* Cut: unfinished

The GUI can be found in `manualCorrectionGUI.py`.

## Graph analysis

### Compartment partitioning

We partition the artery network into several compartments based on physical locations: LMCA (starting from left middle cerebral artery)/RMCA (starting from right middle cerebral artery)/LPCA (starting from left posterior cerebral artery)/RPCA (starting from right posterior cerebral artery)/ACA (starting from anterior cerebral artery). The partitioning process is done using a GUI in `partitionCompartmentGUI.py`.

### Morporlogical properties

We calculate the various morphological properties listed in this [paper](https://www.ncbi.nlm.nih.gov/pubmed/23727319). This is done in `pythonFileName.py`.

## Blood flow simulation

We perform simplified blood flow simulation using the [Hazen–Williams equation](https://en.wikipedia.org/wiki/Hazen%E2%80%93Williams_equation) (H-W equation) on each of the compartments separately, or on the complete artery network with [Circle of Willis](https://en.wikipedia.org/wiki/Circle_of_Willis) (CoW) included. For the CoW, we create a simplified CoW structure from this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4420884/) and take their blood distribution numbers as a reference. The terminating pressure vs path length relationship used in our paper comes from the [ADAN](http://hemolab.lncc.br/adan-web/doc/index.html) dataset.

## File dependencies

Most of the functionalities in the pipeline are modularized and separated into different `.py` files, and each `.py` file requires different input files and produces different output files. Thus, we show these dependencies in this section.

----
**Filename:** `generateVesselVolume.py`  
**Description:** Create a vessel volume mask using the vesselness-filtered brain volume.  
**Requires:** 
* brainVolume.nii.gz: The extracted brain volume.  
* brainVolumeMask.nii.gz: The data mask of the extracted brain volume.  
* vesselnessFiltered.nii.gz: The vesselness-filtered brain volume.      

**Produces:**  
* vesselVolumeMask.nii.gz: The data mask of the resulting vessel volume.

----
**Filename:** `skeletonization.py`  
**Description:** Skeletonize the extracted vessel volume.  
**Requires:**  
* Result from the skeletonization (`result_segments_xyz*.txt`)

**Produces:**  
* graphRepresentation.graphml: The graph containing all the segments (vessel branches) and connections.
* segmentList.npz: A list containing all the segments (vessel branches) from the skeletonization.  
* skeleton.nii.gz: A Nifti file showing the skeletons (centerpoints).

----
**Filename:** `manualCorrectionGUI.py`  
**Description:** Manually correct the wrong connections.  
**Requires:**  
* graphRepresentation.graphml: The graph containing all the segments (vessel branches) and connections.
* segmentList.npz: A list containing all the segments (vessel branches) from the skeletonization.  
* skeleton.nii.gz: A Nifti file showing the skeletons (centerpoints).
* vesselVolumeMask.nii.gz: The data mask of the segmented vessel volume.

**Produces:**  
* removeList.npy: A list containing the segment indices that have been removed.
* eventList.pkl: A pickled data file that contains the information of every step performed within the GUI and can be used to restore previous progress by loading it into the GUI.  
* segmentListCleaned.npz: A list containing all the segments (vessel branches) after manual correction.  
* graphRepresentationCleaned.graphml: The graph containing all the segments (vessel branches) and connections corresponding to `segmentListCleaned.npz`.
* graphRepresentationCleanedWithEdgeInfo.graphml: The graph containing all the segments (vessel branches) and connections corresponding to `segmentListCleaned.npz` with basic branch properties (length, radius, etc.,) attached.
----

**Filename:** `partitionCompartmentGUI.py`  
**Description:** Partition the segmented vessels into different compartments.  
**Requires:**  
* segmentListCleaned.npz: A list containing all the segments (vessel branches) after manual correction.  
* graphRepresentationCleanedWithEdgeInfo.graphml: The graph containing all the segments (vessel branches) and connections corresponding to `segmentListCleaned.npz` with basic branch properties (length, radius, etc.,) attached.
* vesselVolumeMask.nii.gz: The data mask of the segmented vessel volume.

**Produces:**  
* chosenVoxelsForPartition.pkl: Contains the `initialVoxels` and `boundaryVoxels` for each compartment selected by the user.
* partitionInfo.pkl: Contains information about nodes (`visitedVoxels`) and segments (`segmentIndexList`) within each compartment.
* graphRepresentationCleanedWithAdvancedInfo.graphml: The same graph as `graphRepresentationCleanedWithEdgeInfo` and has additional depth information.

----

**Filename:** `graphRelated.py`  
**Description:** A collection of functions used to calculate morphological properties and creating plots.  
**Requires:**  
* chosenVoxelsForPartition.pkl: Contains the `initialVoxels` and `boundaryVoxels` for each compartment selected by the user.
* partitionInfo.pkl: Contains information about nodes (`visitedVoxels`) and segments (`segmentIndexList`) within each compartment.
* graphRepresentationCleanedWithAdvancedInfo.graphml: The same graph as `graphRepresentationCleanedWithEdgeInfo` and has additional depth information.
* segmentListCleaned.npz: A list containing all the segments (vessel branches) after manual correction.  
* nodeInfoDict.pkl: A dictionary containing information about all bifurcations.  
* segmentInfoDict.pkl: A dictionary containing information about all the segments in `segmentListCleaned.npz`.

**Produces:**  
* nodeInfoDict.pkl: A dictionary containing information about all bifurcations.  
* segmentInfoDict.pkl: A dictionary containing information about all the segments in `segmentListCleaned.npz`.

----

**Filename:** `xx.py`  
**Description:** aa  
**Requires:**  
* a
* b

**Produces:**  
* a
* b

