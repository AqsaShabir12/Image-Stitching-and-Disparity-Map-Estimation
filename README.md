#Image Stitching and Disparity Map Estimation**

This folder contains the implementation of two tasks as part of the CS554 Fall 2024 Homework. The project includes code for stitching images and calculating disparity maps for stereo image pairs.

**Group Members**:

- Kousar Kousar
- Aqsa Shabbir

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Image Stitching](#ImageStitching)
  - [Disparity Map Estimation](#DisparityMapEstimation)
- [Dataset](#dataset)
  - [data_image_stitching ](#data_image_stitching )
  - [data_disparity_estimation ](#data_disparity_estimation )
- [Results](#results)
  - [Image Stitching Results](#ImageStitchingResults)
  - [Disparity Map Estimation Results](#DisparityMapEstimationResults)

## Installation

To set up the environment, install the necessary dependencies as specified below:

```bash
pip3 install opencv-python numpy matplotlib
```

## Usage

### Image Stitching

The image_stitching.ipynb module implements an image stitching pipeline using SIFT-based feature matching and RANSAC-based homography estimation.

Steps:

1. Open the image_stitching.ipynb file.
2. Select the python jupyter notebook kernal of choice.
3. Run the module by passing the path of two images to be stitched.
<!--

````bash
# Apply
python3 image_stitching.ipynb
``` -->

### Disparity Map Estimation
The disparity_map_estimation.ipynb module estimates a dense disparity map between a pair of rectified images using normalized cross-correlation for patch matchings.

Steps:

1. Open the disparity_map_estimation.ipynb file.
2. Select the python jupyter notebook kernal of choice.
3. Run the module by specifying the paths of the left and right images.
<!--

````bash
# Apply
python3 disparity_map_estimation.ipynb
``` -->

### Dataset

#### Data Collection
Data for this project should be downloaded from 'http://www.cs.bilkent.edu.tr/~dibeklioglu/teaching/cs554/docs/homework_dataset.zip'.

#### Overview
This dataset includes:
1. Images for the stitching task in the data_image_stitching folder.
2. Image pairs for the disparity map estimation task in the data_disparity_estimation folder. The data_disparity_estimation folder contains ground truth disparity maps to evaluate performance.

### Results

#### Image Stitching Results
- **Method: SIFT-based feature matching, RANSAC-based homography estimation, and alpha blending.**

##### Example Result Image

![Image Stitching Results example 1](/Results/Pair1.png)
![Image Stitching Results example 2](/Results/Pair2.png)
![Image Stitching Results example 3](/Results/Pair3.png)
![Image Stitching Results example 4](/Results/Pair4.png)

#### Disparity Map Estimation Results
- **Method: Patch-based normalized cross-correlation for pixel-to-pixel disparity estimation**

##### Example Result Image

![Disparity Map Estimation Results example 1](/Results/disparity_cloth.png)
![Disparity Map Estimation Results example 2](/Results/disparity_plastic.png)
