import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

import numpy as np

from perception.perception_utils import PointCloudUtil

import open3d as o3d

class PointCloudAggregator(Node):
    def __init__(self):
        super().__init__("pc_aggregator_node")
        
        self.pc = o3d.geometry.PointCloud()

        self.pc_publisher = self.create_publisher(PointCloud2, "/total_pc", 10)
        self.pc_listener = self.create_subscription(PointCloud2, "/point_cloud", self.pc_callback, 10)
        
    def pc_callback(self, msg):
        # TODO: add ICP, intersection detection/handling, maybe pc normalization to reduce gaps (not sure yet)

        pc = PointCloudUtil.ros_pc2_to_o3d_pc(msg)
        self.pc = PointCloudUtil.add_point_clouds(pc, self.pc)
        
        ros_pc = PointCloudUtil.o3d_pc_to_point_cloud2(self.pc)
        self.pc_publisher.publish(ros_pc)
        
def main(args=None):
    node = PointCloudAggregator()
    rclpy.spin(node)
    node.destroy_node()
        
if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()