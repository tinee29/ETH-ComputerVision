# Computer Vision Assignment 6 - Fall 2023

## Overview

This assignment implements a **CONDENSATION Tracker** using color histograms for object tracking in videos. The tracker uses a particle filter to estimate the position of an object across video frames. Key components include:

1. **Color Histogram Extraction**: Compute histograms for bounding boxes around the object.
2. **Particle Propagation**: Update particle states using no-motion or constant velocity models.
3. **Observation and Weighting**: Assign weights to particles based on histogram similarity.
4. **Estimation**: Compute the mean state of particles.
5. **Resampling**: Focus on the most likely particle positions.

The tracker is tested on three videos, with performance evaluated by tuning parameters like system noise, measurement noise, and the number of particles.

---

## How to Run the Code

1. Ensure the required libraries (`numpy`, `opencv-python`, `matplotlib`) are installed.
2. Place the videos in the `ex6_data/` directory.
3. Run the script:

   ```bash
   python condensation_tracker.py
