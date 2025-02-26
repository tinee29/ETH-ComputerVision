# Computer Vision Assignments - Fall 2023

## Overview

This repository contains five computer vision assignments from the Fall 2023 course. Each assignment focuses on a distinct area of computer vision, covering techniques ranging from feature detection to advanced 3D reconstruction. The assignments are structured with clear tasks, scripts, and instructions to implement and evaluate key algorithms.

### Assignments Overview:

1. **Assignment 1: Harris Corner Detection and Descriptor Matching**
2. **Assignment 2: Image and Point Cloud Classification with PyTorch**
3. **Assignment 3: Bag-of-Words and CNN-based Image Classification**
4. **Assignment 4: CONDENSATION Tracker with Particle Filtering**
5. **Assignment 5: Structure from Motion (SfM) and Model Fitting**

---

## Assignment Summaries

### 1. Harris Corner Detection and Descriptor Matching

**Goal:** Implement Harris Corner Detection, local descriptor extraction, and descriptor matching between images.

**Tasks:**
- Detect corners using the Harris Corner Detector.
- Extract local descriptors around detected corners.
- Match descriptors between two images using multiple matching strategies.


**Dependencies:**
- OpenCV (`cv2`)
- NumPy

---

### 2. Image and Point Cloud Classification with PyTorch

**Goal:** Implement and evaluate classification models for images (MNIST, CIFAR-10) and 2D point clouds.

**Tasks:**
- Image Classification:
  - Implement MLP for MNIST and CNN for CIFAR-10.
  - Use pre-trained models (e.g., ResNet18) and improve performance with augmentation.
- Point Cloud Classification:
  - Implement Logistic Regression and MLP for binary classification.
  - Visualize decision boundaries.


**Dependencies:**
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

### 3. Bag-of-Words and CNN-based Image Classification

**Goal:** Implement a Bag-of-Words (BoW) classifier for binary classification and a simplified VGG network for CIFAR-10 multi-class classification.

**Tasks:**
- Bag-of-Words Classifier:
  - Extract HOG descriptors and create a visual vocabulary using K-Means.
  - Classify images via nearest-neighbor matching of BoW histograms.
- CNN Classifier:
  - Implement a simplified VGG model.
  - Train and test the model on the CIFAR-10 dataset.

**Dependencies:**
- NumPy
- OpenCV (`opencv-python`)
- scikit-learn
- tqdm
- PyTorch (for CNN classifier)

---

### 4. CONDENSATION Tracker with Particle Filtering

**Goal:** Implement an object tracking system using color histograms and particle filtering.

**Tasks:**
- Extract color histograms for object localization.
- Implement particle propagation and resampling.
- Evaluate performance by tuning parameters like noise and the number of particles.

**Dependencies:**
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib

---

### 5. Structure from Motion (SfM) and Model Fitting

**Goal:** Implement a 3D reconstruction pipeline using SfM and compare model fitting methods (Least Squares vs. RANSAC).

**Tasks:**
- Structure from Motion:
  - Feature matching, camera pose estimation, and triangulation.
  - Incrementally reconstruct a 3D scene.
- Model Fitting:
  - Implement Least Squares (sensitive to outliers) and RANSAC (robust to noise).

**Dependencies:**
- NumPy
- Matplotlib

---



