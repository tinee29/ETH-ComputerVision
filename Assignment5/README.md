# Computer Vision Assignment 5 - Fall 2023

## Overview

This assignment focuses on **Structure from Motion (SfM)** and **Model Fitting** techniques. The tasks include:

1. **Structure from Motion (SfM)**:
   - Implement a 3D reconstruction pipeline using feature matching, camera pose estimation, and point triangulation.
   - Reconstruct a 3D scene from a set of 2D images.
   - Visualize the 3D point cloud and camera poses.

2. **Model Fitting**:
   - Implement **Least Squares** and **RANSAC** algorithms for line fitting.
   - Compare the robustness of both methods in the presence of noise and outliers.

---

## Structure from Motion (SfM)

### Tasks

1. **Feature Matching**: Extract and match keypoints across multiple images.
2. **Camera Pose Estimation**: Estimate the relative pose of cameras using the Essential Matrix.
3. **Triangulation**: Reconstruct 3D points from 2D correspondences.
4. **Incremental Reconstruction**: Add more images to the reconstruction and refine the 3D model.

### Results

- The 3D reconstruction is visualized as a point cloud, with varying point densities reflecting feature-rich areas in the images.
- Challenges include sparse point clouds in regions with insufficient features or image overlap.

---

## Model Fitting

### Tasks

1. **Least Squares**:
   - Fit a linear model to noisy data using the least squares method.
   - Sensitive to outliers, leading to less accurate parameter estimates.

2. **RANSAC**:
   - Robustly fit a linear model by iteratively selecting inliers and rejecting outliers.
   - More accurate parameter estimates, especially in the presence of outliers.

### Results

- **Least Squares**: Estimated parameters deviate from the ground truth due to outliers.
- **RANSAC**: Estimated parameters are closer to the ground truth, demonstrating robustness to outliers.

---

## How to Run the Code

1. Ensure the required libraries (`numpy`, `matplotlib`, etc.) are installed.
2. Place the images and keypoints in the `data/` directory.
3. Run the SfM script:

   ```bash
   python sfm.py
