import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # Reshape desc1 and desc2 to enable broadcasting
    desc1 = desc1[:, np.newaxis, :]
    desc2 = desc2[np.newaxis, :, :]

    # Compute the squared differences element-wise
    squared_diff = (desc1 - desc2) ** 2

    # Sum along the feature dimension to get the SSD distances
    distances = np.sum(squared_diff, axis=2)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # Find the index of the minimum distance for each descriptor in desc1
        min_indices = np.argmin(distances, axis=1)

        # Create an array of indices for desc1 and corresponding matches in desc2
        matches = np.column_stack((np.arange(q1), min_indices))

    elif method == "mutual":
        # Compute one-way matches
        one_way_matches = np.argmin(distances, axis=1)
        
        # Check if either desc1 or desc2 is empty before proceeding
        if q1 == 0 or q2 == 0:
            return None

        # Compute one-way matches in the opposite direction
        reverse_distances = np.argmin(distances, axis=0)
        
        # Create an array of indices for desc2 and corresponding matches in desc1
        reverse_matches = np.column_stack((reverse_distances, np.arange(q2)))
        
        # Find mutual matches by checking both directions
        mutual_matches = []
        for i, match in enumerate(one_way_matches):
            if reverse_matches[match, 0] == i:
                mutual_matches.append([i, match])
        
        if mutual_matches:
            matches = np.array(mutual_matches)

    elif method == "ratio":
        # Find the indices of the two smallest distances for each descriptor in desc1
        sorted_indices = np.argsort(distances, axis=1)
        min_indices1 = sorted_indices[:, 0]
        min_indices2 = sorted_indices[:, 1]

        # Calculate the ratio of the two smallest distances
        ratio = distances[np.arange(q1), min_indices1] / distances[np.arange(q1), min_indices2]

        # Create a mask for matches based on the ratio threshold
        ratio_mask = ratio < ratio_thresh

        # Create an array of indices for desc1 and corresponding matches in desc2
        matches = np.column_stack((np.where(ratio_mask)[0], min_indices1[ratio_mask]))
        
    else:
        raise NotImplementedError
    return matches

