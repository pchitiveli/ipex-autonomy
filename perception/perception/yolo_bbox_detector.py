import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, PointCloud2
from lac_interfaces.msg import PoseStampedImagePair

from cv_bridge import CvBridge

import open3d as o3d

import ultralytics

from util.perception_utils import PointCloudUtil


class RockDetection(Node):
    IMG_HEIGHT = 720
    IMG_WIDTH = 1280

    def __init__(self):
        super().__init__("yolo_bbox_node")

        self.rockSpecifications = []
        self.model = ultralytics.YOLO(
            # "/workspace/ORB_SLAM3/LAC-Code/perception/models/model.pt"
            "/workspace/team_code/perception/models/model.pt"
        )
        # self.model.to('cuda')
        
        self.last_pos = np.array([0.0, 0.0, 0.0])
        self.last_rpy = np.array([0.0, 0.0, 0.0])
        
        self.acc_rock_pc = o3d.geometry.PointCloud()

        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # ROS info flow setup
        self.bridge = CvBridge()

        self.depth_pair_listener = self.create_subscription(
            PoseStampedImagePair,
            "/front_stereo_img_pair",
            self.image_callback,
            10,
        )

        self.rock_pc_pub = self.create_publisher(PointCloud2, "/rock/pc", 10)

        # TODO: debugging - for checking pc against image
        # self.id_rocks_pub = self.create_publisher(Image, "/id_rocks", 1)

    # =========================================================================================================
    # CALLBACKS
    # =========================================================================================================

    def image_callback(self, msg):
        encoding = msg.image_pair.left.encoding

        imgL = self.bridge.imgmsg_to_cv2(msg.image_pair.left, desired_encoding=encoding)
        imgR = self.bridge.imgmsg_to_cv2(msg.image_pair.right, desired_encoding=encoding)
        
        # read in position
        pos = np.array(
            [
                msg.position.x,
                msg.position.y,
                msg.position.z,
            ]
        )

        # read in roll, pitch, yaw
        rpy = np.array(
            [
                msg.orientation.roll,
                msg.orientation.pitch,
                msg.orientation.yaw,
            ]
        )
        
        # only detect rocks if robot has moved by at least 0.3m from last scan or yaw angle has changed by 5 rad
        should_detect = (np.linalg.norm(pos - self.last_pos) >= 0.3) or (np.abs(rpy[2] - self.last_rpy[2]) >= np.deg2rad(5))

        if imgL is not None and should_detect:
            if imgL.shape is not (self.IMG_WIDTH, self.IMG_HEIGHT):
                imgL = cv2.resize(imgL, (self.IMG_WIDTH, self.IMG_HEIGHT))
                imgR = cv2.resize(imgR, (self.IMG_WIDTH, self.IMG_HEIGHT))

            depth = PointCloudUtil.compute_depth_image(imgL, imgR)

            # sharpen left image to increase accuracy
            imgL = cv2.filter2D(imgL, -1, self.sharpen_kernel)

            imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)

            boxes = self.predict_rocks(imgL)
            rock_depth = self.get_rock_depth_map(boxes, depth)

            # TODO: USE THE FOLLOWING COMMENTED BLOCK FOR DEBUGGING
            # make and publish id rock image
            # id_img = imgL.copy()
            # for box in boxes:
            #     id_img = cv2.rectangle(
            #         id_img,
            #         (int(box[0]), int(box[1])),
            #         (int(box[2]), int(box[3])),
            #         (0, 255, 0),
            #         2,
            #     )
            # id_img_msg = self.bridge.cv2_to_imgmsg(id_img)
            # self.id_rocks_pub.publish(id_img_msg)

            # convert back to grayscale for pc generation
            imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
            rock_pc = PointCloudUtil.compute_point_cloud_from_depth(
                imgL, rock_depth, origin_pos=pos, origin_rpy=rpy, voxel_size=0.01
            )

            rock_pc.colors = o3d.utility.Vector3dVector(
                [[1.0, 0, 0]] * len(rock_pc.points)
            )
            # rock_pc, _ = rock_pc.remove_radius_outlier(nb_points=2, radius=0.01)
            rock_pc, _ = rock_pc.remove_statistical_outlier(
                nb_neighbors=15, std_ratio=1e-8
            )
            
            self.acc_rock_pc = PointCloudUtil.add_point_clouds(rock_pc, self.acc_rock_pc)

            self.rock_pc_pub.publish(
                PointCloudUtil.o3d_pc_to_point_cloud2(
                    self.get_clock().now().to_msg(), self.acc_rock_pc
                )
            )
            
            # update pose of last scan
            self.last_pos = pos
            self.last_rpy = rpy

    # =========================================================================================================
    # ROCK DETECTION
    # =========================================================================================================

    def predict_rocks(self, left):
        """
        Predicts the objects in a stereo image pair. Groups them up into pairs based on their location in the image.
        Parameters:
            left (cv2 img): left image
        Returns:
            boxes (list): list of bounding boxes generated in [x1, y1, x2, y2]
        """

        leftResults = self.model(left)[0]
        boxes = leftResults.boxes.xyxy.tolist()

        return boxes

    def get_rock_depth_map(self, boxes, depth):
        """
        Calculates depth map with only rocks and depth map with no rocks.

        Args:
            boxes (list): list of (x1, y1, x2, y2) bounding box coords
            depth (np.ndarray): depth image to segment
        Returns:
            rock_map (np.ndarray): depth map with only rocks
            filtered_map (np.ndarray): depth map with rock bounding boxes filtered out
        """
        rock_map = np.zeros(depth.shape)

        cols, rows = np.indices(depth.shape)
        for box in boxes:
            # segment depth map corresponding to rock into local np array
            local_rock = np.where(
                (rows > box[0]) & (cols > box[1]) & (rows < box[2]) & (cols < box[3]),
                depth,
                0,
            )

            # add local depth map to array of zeroes
            rock_map += local_rock

        return rock_map


def main():
    node = RockDetection()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
