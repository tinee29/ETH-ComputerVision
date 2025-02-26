import cv2
import numpy as np

def color_histogram(xMin, yMin, xMax, yMax, frame, hist_bin):
    xMin = max(1, xMin)
    xMax = min(xMax, frame.shape[1])
    yMin = max(1, yMin)
    yMax = min(yMax, frame.shape[0])

    xMin = round(xMin)
    xMax = round(xMax)
    yMin = round(yMin)
    yMax = round(yMax)

    bounding_box_img = frame[yMin-1:yMax, xMin-1:xMax, :]

    hist_R = cv2.calcHist([bounding_box_img], [0], None, [hist_bin], [0, 256])
    hist_G = cv2.calcHist([bounding_box_img], [1], None, [hist_bin], [0, 256])
    hist_B = cv2.calcHist([bounding_box_img], [2], None, [hist_bin], [0, 256])

    hist = np.vstack((hist_R, hist_G, hist_B))
    hist = hist / np.sum(hist)

    return hist