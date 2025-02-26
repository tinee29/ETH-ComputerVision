# Computer Vision Assignment 2

## Overview

This assignment focuses on implementing and analyzing image classification and point cloud binary classification using PyTorch. The assignment is divided into two main parts:

1. **Image Classification with PyTorch (`image_classification.ipynb`)**
2. **Point Cloud Binary Classification with PyTorch (`pcl_binary_classification.ipynb`)**

---

## Image Classification with PyTorch

### Tasks

1. **Data Loading and Preprocessing**:
   - Load the MNIST and CIFAR10 datasets.
   - Normalize the data and apply transformations (e.g., random cropping, flipping for CIFAR10).
   - Create DataLoader objects for training and testing.

2. **Model Implementation**:
   - Implement a Multi-Layer Perceptron (MLP) for MNIST classification.
   - Implement a Convolutional Neural Network (CNN) for CIFAR10 classification.
   - Adapt the CNN to work with CIFAR10 by modifying the input channels and spatial dimensions.

3. **Training and Evaluation**:
   - Train the MLP and CNN models using appropriate loss functions (cross-entropy) and optimizers (Adam, SGD).
   - Evaluate the models on the test set and achieve high accuracy (e.g., >97% for MNIST, >90% for CIFAR10).

4. **Advanced Techniques**:
   - Use pre-trained models (e.g., ResNet18) for CIFAR10 classification.
   - Experiment with data augmentation, learning rate scheduling, and hyperparameter tuning to improve performance.

---

## Point Cloud Binary Classification with PyTorch

### Tasks

1. **Dataset Loading and Visualization**:
   - Load a 2D point cloud dataset and visualize the data points.

2. **Model Implementation**:
   - Implement a Logistic Regression model for binary classification.
   - Implement a Multi-Layer Perceptron (MLP) for improved classification.

3. **Training and Evaluation**:
   - Train the models using binary cross-entropy loss and the Adam optimizer.
   - Visualize the decision boundary to evaluate the model's performance.

---

## How to Run the Code

1. Ensure the required libraries (`torch`, `torchvision`, `numpy`, `matplotlib`, etc.) are installed.
2. Place the datasets in the `data/` directory.
3. Run the notebooks:
   - `image_classification.ipynb` for image classification tasks.
   - `pcl_binary_classification.ipynb` for point cloud classification tasks.

---

## Dependencies

- PyTorch
- Torchvision
- NumPy
- Matplotlib

---
