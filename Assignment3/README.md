# Computer Vision Assignment 4 - Fall 2023

**Professor:** Marc Pollefeys  
**Student:** Igor Martinelli  
**Email:** <maigor@ethz.ch>  
**ID:** 19-916-048  

## Overview

This assignment focuses on implementing two object recognition systems:

1. **Bag-of-Words (BoW) Image Classifier** for binary classification (car vs. non-car).
2. **CNN-based Image Classifier** for multi-class classification on the CIFAR-10 dataset.

---

## Bag-of-Words Classifier

### Tasks

1. **Local Feature Extraction**:
   - Implement a grid-based feature point detector (`grid_points`).
   - Compute Histogram of Oriented Gradients (HOG) descriptors for each feature point (`descriptors_hog`).

2. **Codebook Construction**:
   - Cluster local descriptors using K-Means to create a visual vocabulary (`create_codebook`).

3. **Bag-of-Words Vector Encoding**:
   - Compute BoW histograms for training images (`create_bow_histograms`).
   - Encode test images using the BoW representation (`bow_histogram`).

4. **Nearest Neighbor Classification**:
   - Classify test images by comparing their BoW histograms to training histograms (`bow_recognition_nearest`).

---

## CNN-based Classifier

### Tasks

1. **Simplified VGG Network**:
   - Implement a simplified version of the VGG network for CIFAR-10 classification (`models/vgg_simplified.py`).

2. **Training and Testing**:
   - Train the VGG network using the provided training script (`train_cifar10_vgg.py`).
   - Monitor training progress using TensorBoard.
   - Test the trained model on the CIFAR-10 test set (`test_cifar10_vgg.py`).

---

## How to Run the Code

### Bag-of-Words Classifier

1. Ensure the required libraries (`numpy`, `opencv-python`, `scikit-learn`, `tqdm`, etc.) are installed.
2. Place the dataset in the `data/data_bow/` directory.
3. Run the script:

   ```bash
   python bow_main.py
