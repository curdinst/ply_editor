import numpy as np
from scipy.spatial import KDTree

def scale_aligning_icp(source, target, init_tranformation, max_iterations=50, tolerance=1e-6):
    """
    Perform scale-aligning ICP to align the source point cloud to the target point cloud.
    
    Args:
        source (numpy.ndarray): Source point cloud of shape (N, 3).
        target (numpy.ndarray): Target point cloud of shape (M, 3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance for the transformation.

    Returns:
        numpy.ndarray: Aligned source point cloud.
        numpy.ndarray: Final transformation matrix (4x4).
        float: Final scale factor.
    """
    # Initialize transformation matrix and scale
    transformation = init_tranformation
    scale = 1.0

    # Add a column of ones to source and target for homogeneous coordinates
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print("scale: ", scale)
        # Apply current transformation and scale to the source
        transformed_source = (transformation @ source_h.T).T[:, :3] * scale

        # Find nearest neighbors in the target for each point in the transformed source
        tree = KDTree(target)
        distances, indices = tree.query(transformed_source)
        closest_points = target[indices]

        # Compute centroids of the matched points
        centroid_source = np.mean(transformed_source, axis=0)
        centroid_target = np.mean(closest_points, axis=0)

        # Center the points
        centered_source = transformed_source - centroid_source
        centered_target = closest_points - centroid_target

        # Estimate scale
        scale_numerator = np.sum(np.linalg.norm(centered_target, axis=1) ** 2)
        scale_denominator = np.sum(np.linalg.norm(centered_source, axis=1) ** 2)
        new_scale = np.sqrt(scale_numerator / scale_denominator)

        # Estimate rotation using Singular Value Decomposition (SVD)
        H = centered_source.T @ centered_target
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Estimate translation
        t = centroid_target - new_scale * (R @ centroid_source)

        # Update transformation matrix
        new_transformation = np.eye(4)
        new_transformation[:3, :3] = R
        new_transformation[:3, 3] = t

        # Check for convergence
        delta_transformation = np.linalg.norm(new_transformation - transformation)
        delta_scale = abs(new_scale - scale)
        if delta_transformation < tolerance and delta_scale < tolerance:
            break

        # Update transformation and scale
        transformation = new_transformation
        scale = new_scale

    # Apply final transformation and scale to the source
    aligned_source = (transformation @ source_h.T).T[:, :3] * scale

    return aligned_source, transformation, scale

def scale_only_aligning_icp(source, target, init_tranformation, init_scale, max_iterations=50, tolerance=1e-6):
    """
    Perform scale-aligning ICP to align the source point cloud to the target point cloud.
    
    Args:
        source (numpy.ndarray): Source point cloud of shape (N, 3).
        target (numpy.ndarray): Target point cloud of shape (M, 3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance for the transformation.

    Returns:
        numpy.ndarray: Aligned source point cloud.
        numpy.ndarray: Final transformation matrix (4x4).
        float: Final scale factor.
    """
    # Initialize transformation matrix and scale
    transformation = init_tranformation
    scale = init_scale

    # Add a column of ones to source and target for homogeneous coordinates
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print("scale: ", scale)
        # Apply current transformation and scale to the source
        transformed_source = (transformation @ source_h.T).T[:, :3] * scale

        # Find nearest neighbors in the target for each point in the transformed source
        tree = KDTree(target)
        distances, indices = tree.query(transformed_source)
        closest_points = target[indices]

        # Compute centroids of the matched points
        centroid_source = np.mean(transformed_source, axis=0)
        centroid_target = np.mean(closest_points, axis=0)

        # Center the points
        centered_source = transformed_source - centroid_source
        centered_target = closest_points - centroid_target

        # Estimate scale
        scale_numerator = np.sum(np.linalg.norm(centered_target, axis=1) ** 2)
        scale_denominator = np.sum(np.linalg.norm(centered_source, axis=1) ** 2)
        new_scale = np.sqrt(scale_numerator / scale_denominator)

        new_transformation = init_tranformation.copy()
        new_transformation[:3, 3] = new_scale * init_tranformation[:3, 3]

        # # Estimate rotation using Singular Value Decomposition (SVD)
        # H = centered_source.T @ centered_target
        # U, _, Vt = np.linalg.svd(H)
        # R = Vt.T @ U.T

        # # Ensure a proper rotation (det(R) = 1)
        # if np.linalg.det(R) < 0:
        #     Vt[-1, :] *= -1
        #     R = Vt.T @ U.T

        # # Estimate translation
        # t = centroid_target - new_scale * (R @ centroid_source)

        # # Update transformation matrix
        # new_transformation = np.eye(4)
        # new_transformation[:3, :3] = R
        # new_transformation[:3, 3] = t

        # Check for convergence
        # delta_transformation = np.linalg.norm(new_transformation - transformation)
        delta_scale = abs(new_scale - scale)
        if delta_scale < tolerance:
            break

        # Update transformation and scale
        # transformation = new_transformation
        scale = new_scale

    # Apply final transformation and scale to the source
    aligned_source = (transformation @ source_h.T).T[:, :3] * scale

    return aligned_source, transformation, scale