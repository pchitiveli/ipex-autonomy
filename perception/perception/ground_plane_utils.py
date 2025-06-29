import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

class GroundPlaneEstimator:
    def __init__(self, eps=0.1, min_samples=5, z_offset_threshold=0.02):
        self.eps = eps
        self.min_samples = min_samples
        self.z_offset_threshold = z_offset_threshold

    def cluster_points(self, points):
        """Cluster points using DBSCAN."""
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = db.labels_
        return labels

    def estimate_ground_plane(self, ground_points):
        """Estimate ground plane using RANSAC."""
        X = ground_points[:, :2]
        y = ground_points[:, 2]
        ransac = RANSACRegressor().fit(X, y)
        A, B = ransac.estimator_.coef_
        D = ransac.estimator_.intercept_
        return [A, B, -1, D]  # Plane equation: Ax + By - z + D = 0

    def classify_rocks(self, labels, points, ground_plane):
        """Classify clusters as rocks based on ground plane distance."""
        A, B, C, D = ground_plane
        rock_mask = np.zeros(len(points), dtype=bool)
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_mask = (labels == label)
            distances = self._calc_plane_distances(points[cluster_mask], ground_plane)
            if np.mean(distances) > self.z_offset_threshold:
                rock_mask[cluster_mask] = True
        return rock_mask

    def _calc_plane_distances(self, points, plane_coeffs):
        """Calculate distances from points to plane."""
        A, B, C, D = plane_coeffs
        return np.abs(A*points[:,0] + B*points[:,1] + C*points[:,2] + D) / np.sqrt(A**2 + B**2 + C**2)

def pointcloud2_to_array(msg):
    """Convert PointCloud2 message to numpy array."""
    dtype = np.dtype([(f.name, np.float32) for f in msg.fields])
    cloud_arr = np.frombuffer(msg.data, dtype=dtype)
    points = np.column_stack([cloud_arr['x'], cloud_arr['y'], cloud_arr['z']])
    return points