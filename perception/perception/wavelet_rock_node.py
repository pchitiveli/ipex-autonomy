import rclpy
from rclpy.node import Node

import numpy as np
import pywt
import open3d as o3d

from sensor_msgs.msg import PointCloud2

from util.perception_utils import PointCloudUtil


class WaveletRockIDNode(Node):
    def __init__(self):
        super().__init__("wavelet_rock_id_node")

        # listener for point cloud
        self.pc_listener = self.create_subscription(
            PointCloud2, "/point_cloud", self.image_callback, 10
        )

        # publisher for rock points
        self.rock_publisher = self.create_publisher(PointCloud2, "/rock_pc", 10)

    def image_callback(self, msg):
        print("received pc")
        pc = PointCloudUtil.ros_pc2_to_o3d_pc(msg)

        if pc is not None:
            rock_pc = self.identify_rocks(pc)
            rock_pc_ros = PointCloudUtil.o3d_pc_to_point_cloud2(
                self.get_clock().now().to_msg(), rock_pc
            )

            self.rock_publisher.publish(rock_pc_ros)

    def identify_rocks(self, pc):
        points = np.asarray(pc.points)

        coeffs = pywt.wavedec(points[:, 2], "db4", level=3)  # Daubechies-4 wavelet
        detail_coeffs = coeffs[1]

        # Step 3: Identify bumps using thresholding on wavelet detail coefficients
        threshold = np.std(detail_coeffs) * 2
        bump_indices = np.where(np.abs(detail_coeffs) > threshold)[0]

        # Extract bump points
        bump_points = points[bump_indices]

        bump_pcd = o3d.geometry.PointCloud()
        bump_pcd.points = o3d.utility.Vector3dVector(bump_points)

        colors = [[1.0, 0.0, 0.0]] * len(points)
        bump_pcd.colors = o3d.utility.Vector3dVector(colors)

        return bump_pcd


def main():
    rclpy.init()

    node = WaveletRockIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
