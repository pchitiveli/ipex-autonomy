#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, OccupancyGrid, Imu
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import Float32
from lac_interfaces.msg import Plane
from lac_interfaces.srv import CellData

import numpy as np
from ground_plane_utils import GroundPlaneEstimator, pointcloud2_to_array


class GroundPlaneNode(Node):
    def __init__(self):
        super().__init__("ground_plane_estimator")

        # Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("dbscan_eps", 0.1),
                ("min_samples", 5),
                ("z_offset_threshold", 0.02),
                ("map_resolution", 0.15),  # meters per cell
                ("map_width", 9.0),      # meters
                ("map_height", 9.0),     # meters
            ],
        )

        # Initialize map parameters
        self.resolution = self.get_parameter("map_resolution").value
        self.width = int(self.get_parameter("map_width").value / self.resolution)
        self.height = int(self.get_parameter("map_height").value / self.resolution)
        self.map_origin = [-self.get_parameter("map_width").value/2, 
                          -self.get_parameter("map_height").value/2]
        
        # Initialize height map and rock map
        self.height_map = np.full((self.height, self.width), np.nan)
        self.rock_map = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Initialize ground plane estimator
        self.estimator = GroundPlaneEstimator(
            dbscan_eps=self.get_parameter("dbscan_eps").value,
            min_samples=self.get_parameter("min_samples").value,
            z_offset_threshold=self.get_parameter("z_offset_threshold").value
        )
        
        # Publishers
        self.ground_plane_pub = self.create_publisher(Plane, "/ground_plane/coefficients", 10)
        self.ground_height_pub = self.create_publisher(Float32, "/ground_plane/height", 10)
        self.ground_pc_pub = self.create_publisher(PointCloud2, "/ground_plane/ground_point_cloud", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/ground_plane/map", 10)
        self.rock_map_pub = self.create_publisher(OccupancyGrid, "/rock/map", 10)

        # Subscribers
        self.pc_sub = self.create_subscription(PointCloud2, "/pc", self.pc_callback, 10)
        self.rock_pc_sub = self.create_subscription(PointCloud2, "/rock/pc", self.rock_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        
        # Store latest IMU data
        self.latest_imu = None

    def imu_callback(self, msg):
        self.latest_imu = msg

    def world_to_map(self, x, y):
        """Convert world coordinates to map cell indices"""
        map_x = int((x - self.map_origin[0]) / self.resolution)
        map_y = int((y - self.map_origin[1]) / self.resolution)
        return map_x, map_y

    def update_height_map(self, points, plane_coeffs):
        """Update height map with new ground plane points"""
        if self.latest_imu is None:
            return
            
        # Extract roll and pitch from IMU
        qx = self.latest_imu.orientation.x
        qy = self.latest_imu.orientation.y
        qz = self.latest_imu.orientation.z
        qw = self.latest_imu.orientation.w
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1-2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1-2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1-2*(qx**2 + qy**2)]
        ])
        
        # Apply rotation to points
        rotated_points = np.dot(points, R.T)
        
        # Update height map
        for point in rotated_points:
            x, y = self.world_to_map(point[0], point[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                # Calculate height using plane equation
                height = (-plane_coeffs[0]*point[0] - plane_coeffs[1]*point[1] - plane_coeffs[3]) / plane_coeffs[2]
                
                # Update cell if no previous height or new height is higher
                if np.isnan(self.height_map[y, x]) or height > self.height_map[y, x]:
                    self.height_map[y, x] = height

    def publish_height_map(self):
        """Publish height map as OccupancyGrid"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.map_origin[0]
        msg.info.origin.position.y = self.map_origin[1]
        
        # Convert heights to occupancy values (height in mm)
        valid_mask = ~np.isnan(self.height_map)
        occupancy = np.zeros(self.height_map.shape, dtype=np.int8)
        occupancy[valid_mask] = np.round(self.height_map[valid_mask] * 1000).astype(np.int8)
        
        msg.data = occupancy.flatten().tolist()
        self.map_pub.publish(msg)

    def pc_callback(self, msg):
        """Process incoming point cloud."""
        if self.latest_imu is None:
            return
            
        # Convert to numpy array
        points = pointcloud2_to_array(msg)

        if len(points) < 100:
            self.get_logger().warn("Insufficient points for processing")
            return

        # Estimate ground plane using RANSAC directly on all points
        try:
            plane_coeffs = self.estimator.estimate_ground_plane(points)
            self.update_height_map(points, plane_coeffs)
            self.publish_height_map()
            
            # Publish ground plane coefficients for other nodes
            self._publish_ground_plane(plane_coeffs, points)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process point cloud: {str(e)}")
            return

    def _publish_ground_plane(self, coefficients, ground_points):
        """Publish ground plane coefficients and average height."""
        # Publish coefficients
        coeff_msg = Plane()
        coeff_msg.a = float(coefficients[0])
        coeff_msg.b = float(coefficients[1])
        coeff_msg.c = float(coefficients[2])
        coeff_msg.d = float(coefficients[3])
        self.ground_plane_pub.publish(coeff_msg)

        # Calculate and publish average height
        avg_height = np.mean(ground_points[:, 2])
        height_msg = Float32()
        height_msg.data = float(avg_height)
        self.ground_height_pub.publish(height_msg)

    def _publish_rocks(self, original_msg, rock_points):
        """Publish rock points as new PointCloud2."""
        # Create new PointCloud2 message with rock points
        rock_msg = PointCloud2()
        rock_msg.header = original_msg.header
        rock_msg.height = 1
        rock_msg.width = rock_points.shape[0]
        rock_msg.fields = original_msg.fields
        rock_msg.is_bigendian = original_msg.is_bigendian
        rock_msg.point_step = original_msg.point_step
        rock_msg.row_step = rock_msg.point_step * rock_msg.width
        rock_msg.is_dense = True

        # Convert rock points to bytes
        dtype = np.dtype([(f.name, np.float32) for f in original_msg.fields])
        rock_arr = np.zeros(rock_points.shape[0], dtype=dtype)
        rock_arr["x"] = rock_points[:, 0]
        rock_arr["y"] = rock_points[:, 1]
        rock_arr["z"] = rock_points[:, 2]
        rock_msg.data = rock_arr.tobytes()

        self.rock_map_pub.publish(rock_msg)

    def _publish_ground_points(self, original_msg, ground_points):
        """Publish rock points as new PointCloud2."""
        # Create new PointCloud2 message with ground points
        ground_msg = PointCloud2()
        ground_msg.header = original_msg.header
        ground_msg.height = 1
        ground_msg.width = ground_points.shape[0]
        ground_msg.fields = original_msg.fields
        ground_msg.is_bigendian = original_msg.is_bigendian
        ground_msg.point_step = original_msg.point_step
        ground_msg.row_step = ground_msg.point_step * ground_msg.width
        ground_msg.is_dense = True

        # Convert rock points to bytes
        dtype = np.dtype([(f.name, np.float32) for f in original_msg.fields])
        ground_arr = np.zeros(ground_points.shape[0], dtype=dtype)
        ground_arr["x"] = ground_points[:, 0]
        ground_arr["y"] = ground_points[:, 1]
        ground_arr["z"] = ground_points[:, 2]
        ground_msg.data = ground_arr.tobytes()

        self.ground_pc_pub.publish(ground_msg)

    def publish_rock_map(self):
        """Publish rock map as OccupancyGrid"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.map_origin[0]
        msg.info.origin.position.y = self.map_origin[1]
        
        # Convert rock map to occupancy grid format
        # 1: rock present
        # 0: default/unknown state
        # -1: no rock present
        occupancy = self.rock_map.copy()
        msg.data = occupancy.flatten().tolist()
        
        self.rock_map_pub.publish(msg)

    def rock_callback(self, msg):
        """Process incoming rock point cloud and update rock map."""
        if self.latest_imu is None:
            return
            
        # Convert to numpy array
        points = pointcloud2_to_array(msg)

        if len(points) < 1:
            return

        try:
            # Reset rock map to default state (0)
            self.rock_map.fill(0)
            
            # Extract roll and pitch from IMU for point transformation
            qx = self.latest_imu.orientation.x
            qy = self.latest_imu.orientation.y
            qz = self.latest_imu.orientation.z
            qw = self.latest_imu.orientation.w
            
            # Convert quaternion to rotation matrix
            R = np.array([
                [1-2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1-2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1-2*(qx**2 + qy**2)]
            ])
            
            # Apply rotation to points
            rotated_points = np.dot(points, R.T)
            
            # Update rock map
            for point in rotated_points:
                x, y = self.world_to_map(point[0], point[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.rock_map[y, x] = 1  # Mark cell as containing rock
                    
            # Mark cells without rocks as -1
            self.rock_map[self.rock_map == 0] = -1
            
            # Publish rock map
            self.publish_rock_map()
            
        except Exception as e:
            self.get_logger().error(f"Failed to process rock point cloud: {str(e)}")
            return


def main(args=None):
    rclpy.init(args=args)
    node = GroundPlaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
