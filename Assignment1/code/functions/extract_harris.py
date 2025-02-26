import numpy as np
import cv2
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure


# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    Ix = signal.convolve2d(img, np.array([[-1, 0, 1]]), mode='same')
    Iy = signal.convolve2d(img, np.array([[-1], [0], [1]]), mode='same')

    # 2. Blur the computed gradients
    Ix2 = cv2.GaussianBlur(Ix ** 2, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(Iy ** 2, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    

    # 3. Compute elements of the local auto-correlation matrix "M"
    M11 = Ix2
    M22 = Iy2
    M12 = Ixy

    # 4. Compute Harris response function C
    det_M = M11 * M22 - M12 ** 2
    trace_M = M11 + M22
    C = det_M - k * (trace_M ** 2)

    # 5. Detection with threshold and non-maximum suppression
    corner_mask = (C > thresh)  # Thresholding
    local_maxima = ndimage.maximum_filter(C, size=(3, 3)) == C  # Non-maximum suppression
    corners = np.where(corner_mask & local_maxima)  # Find corner coordinates

    # Stack the coordinates to the correct output format
    corners = np.stack((corners[1], corners[0]), axis=-1)
    

    return corners, C

