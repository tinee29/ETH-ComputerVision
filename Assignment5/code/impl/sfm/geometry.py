import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # Normalize coordinates (to points on the normalized image plane)
  normalized_kps1 = (np.linalg.inv(K) @ MakeHomogeneous(im1.kps.T)).T
  normalized_kps2 = (np.linalg.inv(K) @ MakeHomogeneous(im2.kps.T)).T
  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
      kp_idx1 = matches[i, 0]
      kp_idx2 = matches[i, 1]
      
      x1_hat = normalized_kps1[kp_idx1, :]
      x2_hat = normalized_kps2[kp_idx2, :]
      
      A = np.array([
            x1_hat[0] * x2_hat[0], x1_hat[0] * x2_hat[1], x1_hat[0],
            x1_hat[1] * x2_hat[0], x1_hat[1] * x2_hat[1], x1_hat[1],
            x1_hat[2] * x2_hat[0], x1_hat[2] * x2_hat[1], x1_hat[2]
        ])

      constraint_matrix[i, :] = A

  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1, :]

  # Reshape the vectorized matrix to its proper shape again
  E_hat = np.reshape(vectorized_E_hat, (3, 3), order='C')

  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singular values arbitrarily
  u, s, vh = np.linalg.svd(E_hat)
  E = u @ np.diag([1, 1, 0]) @ vh


  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
      kp1 = normalized_kps1[matches[i, 0]]
      kp2 = normalized_kps2[matches[i, 1]]
      assert abs(kp1 @ E @ kp2) < 0.01

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  valid_indices = []
  for i, point in enumerate(points3D):
      # Transform point to each camera's coordinate system
      point_homogeneous = np.append(point, 1)
      cam1_point = P1 @ point_homogeneous
      cam2_point = P2 @ point_homogeneous

      # Check if point is in front of both cameras
      if cam1_point[2] > 0 and cam2_point[2] > 0:
          valid_indices.append(i)

  # Keep only valid points and correspondences
  points3D = points3D[valid_indices]
  im1_corrs = im1_corrs[valid_indices]
  im2_corrs = im2_corrs[valid_indices]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  normalized_points2D = (np.linalg.inv(K) @ MakeHomogeneous(points2D.T)).T[:2, :]



  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

    image = images[image_name]
    points3D = np.zeros((0, 3))
    corrs = {}

    for reg_image_name in registered_images:
        # Avoid triangulating the image with itself
        if image_name == reg_image_name:
            continue

        reg_image = images[reg_image_name]
        pair_matches = GetPairMatches(image_name, reg_image_name, matches)


        # Triangulate points using the keypoints from both images
        new_points3D, im1_corrs, im2_corrs = TriangulatePoints(K, image, reg_image, pair_matches)

        # Update the points3D array with the new points
        start_idx = points3D.shape[0]
        points3D = np.vstack([points3D, new_points3D])

        # Update correspondences for both images
        corrs[image_name] = corrs.get(image_name, []) + (im1_corrs + start_idx).tolist()
        corrs[reg_image_name] = corrs.get(reg_image_name, []) + (im2_corrs + start_idx).tolist()

    return points3D, corrs

  
