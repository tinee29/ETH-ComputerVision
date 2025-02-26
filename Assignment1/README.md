# Computer Vision Assignment 1 - Fall 2023

**Professor:** Marc Pollefeys  
**Student:** Igor Martinelli  
**Email:** <maigor@ethz.ch>  
**ID:** 19-916-048  

## Overview

This assignment focuses on implementing and analyzing key computer vision techniques, including the Harris Corner Detector, local descriptor extraction, and descriptor matching. The code is divided into three main components:

1. **Harris Corner Detection (`extract_harris.py`)**
2. **Local Descriptor Extraction (`extract_descriptors.py`)**
3. **Descriptor Matching (`match_descriptors.py`)**

The goal is to detect corners in grayscale images, extract local descriptors around these corners, and match descriptors between two images using different methods.

---

## How to Run the Code

1. Ensure the required libraries (`cv2`, `numpy`, etc.) are installed.
2. Place the images in the `images/` directory.
3. Run the script:

   ```bash
   python main.py
