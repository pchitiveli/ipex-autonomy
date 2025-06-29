import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
# from visualization_msgs.msg import Marker

from util.perception_utils import PointCloudUtil


class SurfaceNormalNode(Node):
    def __init__(self):
        super().__init__("surface_normal_node")

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
            rock_pc = self.find_rocks_by_surface_normals(pc)
            rock_pc_ros = PointCloudUtil.o3d_pc_to_point_cloud2(
                self.get_clock().now().to_msg(), rock_pc
            )

            self.rock_publisher.publish(rock_pc_ros)

    def find_rocks_by_surface_normals(self, pc):
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100)
        )
        pc.orient_normals_towards_camera_location()

        normals = np.asarray(pc.normals)
        curvature = np.linalg.norm(normals - np.mean(normals, axis=0), axis=1)

        max_threshold = np.percentile(curvature, 100)
        min_threshold = np.percentile(curvature, 85)
        print(len(normals))
        rock_points = np.asarray(pc.points)[
            (curvature > min_threshold) & (curvature < max_threshold)
        ]
        print(len(rock_points))

        colors = [[1.0, 0, 0]] * len(rock_points)

        rock_pc = o3d.geometry.PointCloud()
        rock_pc.points = o3d.utility.Vector3dVector(rock_points)
        rock_pc.colors = o3d.utility.Vector3dVector(colors)

        return rock_pc


def main():
    rclpy.init()

    node = SurfaceNormalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
