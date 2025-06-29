#!/venv/bin/python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
import time
import json
import math
from numpy import random
import numpy as np

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from lac_interfaces.msg import ImagePair, StereoIMU, PoseStampedImagePair
from sensor_msgs.msg import Imu, Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
import launch
import launch_ros.actions

import csv
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

import subprocess


INITIAL_WIDTH = 1280
INITIAL_HEIGHT = 720
IMG_WIDTH = 480
IMG_HEIGHT = 270
FIELD_OF_VIEW = 70.0  # degrees
BASELINE = 0.162
DISPARITY_OCCLUSION_SLOPE = 0.6639344262
DISPARITY_OCCLUSION_YINT = -71.6066

# MAX_DISPARITY = 304 * (
#     float(IMG_WIDTH) / INITIAL_WIDTH
# )  # actually 315, but num_disp has to be divisible by 16
MAX_DISPARITY = 576
MAX_DISPARITY = int(MAX_DISPARITY - (MAX_DISPARITY % 16))

FOV_X = 70.0
FOV_Y = 39.375

FX = float(IMG_WIDTH) / (2 * np.tan(np.deg2rad(FOV_X)))
FY = float(IMG_HEIGHT) / (2 * np.tan(np.deg2rad(FOV_Y)))

CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0

BASELINE = 0.162

# CAMERA INFO SETUP -----------------------------------------------------------------------------------
l_camera_info_msg = CameraInfo()
l_camera_info_msg.header.frame_id = "fl_camera"
l_camera_info_msg.width = IMG_WIDTH
l_camera_info_msg.height = IMG_HEIGHT
l_camera_info_msg.distortion_model = "none" # sim has perfect cameras
l_camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # 0 distortion coefficients b/c perfect cams

# intrinsic camera matrix (p)
# [ fx, 0,  cx ]
# [ 0,  fy, cy ]
# [ 0,  0,  1 ]
#
# fx, fy = focal lengths for each axis
# cx, cy = principal point (center)

l_camera_info_msg.k = [FX, 0.0, CX,
                     0.0, FY, CY,
                     0.0, 0.0, 1.0]

# set rectification matrix to identity matrix bc perfect images
l_camera_info_msg.r = [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]

# Projection matrix (p)
# [ fx, 0,  cx, Tx ]
# [ 0,  fy, cy, Ty ]
# [ 0,  0,  1,  0 ]
#
# fx, fy = focal lengths for each axis
# cx, cy = principal point (center)
# Tx = Baseline * fx (multiplied by negative one for right image)

l_camera_info_msg.p = [FX, 0.0, CX, 0.0, # FX * (BASELINE / 2.0),
                     0.0, FY, CY, 0.0,
                     0.0, 0.0, 1.0, 0.0]

r_camera_info_msg = copy.deepcopy(l_camera_info_msg)
l_camera_info_msg.header.frame_id = "fr_camera"
r_camera_info_msg.p = [FX, 0.0, CX, -FX * BASELINE,
                     0.0, FY, CY, 0.0,
                     0.0, 0.0, 1.0, 0.0]

def get_entry_point():
    return 'AgentV1'

def euler_from_quaternion(x, y, z, w):
    # Roll (x-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (w * y - z * x))
    
    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return roll, pitch, yaw

class AgentNode(Node):
    def __init__(self):
        super().__init__("agent_node")

        # PUBLISHERS ------------------------------------------------------------------------------------------
        # Perception Publishers ------
        self.front_stereo_publisher = self.create_publisher(PoseStampedImagePair, "/front_stereo_img_pair", 10)
        # self.back_stereo_publisher = self.create_publisher(ImagePair, "/back_stereo_img_pair", 10)
        self.imu_publisher = self.create_publisher(Imu, "/imu", 10)
        self.stereo_imu_publisher = self.create_publisher(StereoIMU, "/stereo_imu", 10)
        self.left_img_publisher = self.create_publisher(Image, "/left_img",  10)
        

        # RTAB-Map SLAM Image Publishers ------
        self.fl_img_publisher = self.create_publisher(Image, "/stereo/left_image", 10)
        self.fr_img_publisher = self.create_publisher(Image, "/stereo/right_image", 10)
        self.left_camera_info_pub = self.create_publisher(CameraInfo, "/stereo/left_camera_info", 10)
        self.right_camera_info_pub = self.create_publisher(CameraInfo, "/stereo/right_camera_info", 10)

        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/rtabmap/initialpose", 10)

        self.pos = np.array([0.0, 0.0, 0.0])
        self.rpy = np.array([0.0, 0.0, 0.0])
        
        # read pose from rtabmap
        # self.pose_sub = self.create_subscription(PoseStamped, "visual_slam/tracking/vo_pose", self.pose_callback, 10)
        
        self.bridge = CvBridge()
        
    # ==========================================c================================================================
    # CALLBACKS
    # ==========================================================================================================
    
    def pose_callback(self, msg):
        if msg is not None:
            print("POSE RECEIVED")

            angles = np.array(euler_from_quaternion(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
            
            self.pos[0] = msg.pose.position.x
            self.pos[1] = msg.pose.position.y
            self.pos[2] = msg.pose.position.z
            
            self.rpy = angles
            
            print("POSE UPDATED")
    
    # ==========================================================================================================
    # PUBLISH METHODS
    # ==========================================================================================================

    def publish_slam_images(self, left_image, right_image):
        """
        Publishes images to Isaac ROS SLAM node
        
        Args:
            left_image (np.ndarray): opencv 1280x720 grayscale image
            right_image (np.ndarray): opencv 1280x720 grayscale image
        """
        
        # convert images to Image messages
        imgL = self.bridge.cv2_to_imgmsg(left_image, encoding="mono8")
        imgL.header.stamp = self.get_clock().now().to_msg()
        imgL.header.frame_id = "fl_camera"

        imgR = self.bridge.cv2_to_imgmsg(right_image, encoding="mono8")
        imgR.header.stamp = imgL.header.stamp
        imgR.header.frame_id = "fr_camera"

        l_camera_info_msg.header.stamp = imgL.header.stamp
        r_camera_info_msg.header.stamp = imgR.header.stamp
        
        self.fl_img_publisher.publish(imgL)
        self.fr_img_publisher.publish(imgR)
        self.left_camera_info_pub.publish(l_camera_info_msg)
        self.right_camera_info_pub.publish(r_camera_info_msg)

        print("Published SLAM Image Messages")
        
    def publish_front_img_pair(self, left_image, right_image):
        """
        Publishes front pose stamped stereo pair
        
        Args:
            left_image (np.ndarray): opencv 1280x720 grayscale image
            right_image (np.ndarray): opencv 1280x720 grayscale image
        """
        
        # store temp versions of position and orientation to prevent mid-cycle update
        pos = self.pos
        rpy = self.rpy
        
        # convert images to Image messages
        imgL = self.bridge.cv2_to_imgmsg(left_image, encoding="mono8")
        imgR = self.bridge.cv2_to_imgmsg(right_image, encoding="mono8")
        
        # instantiate pose stamped image pair
        img_pair = PoseStampedImagePair()
        img_pair.image_pair.left = imgL
        img_pair.image_pair.right = imgR
        img_pair.header.stamp = self.get_clock().now().to_msg()
        
        img_pair.position.x = pos[0]
        img_pair.position.y = pos[1]
        img_pair.position.z = pos[2]
        
        img_pair.orientation.roll = rpy[0]
        img_pair.orientation.pitch = rpy[1]
        img_pair.orientation.yaw = rpy[2]
        
        self.front_stereo_publisher.publish(img_pair)
        print("Published front stereo pair")
        
    def publish_imu(self, imu):
        """
        Publishes imu data
        """
        imu_data = Imu()
        imu_data.linear_acceleration.x = imu[0]
        imu_data.linear_acceleration.y = imu[1]
        imu_data.linear_acceleration.z = imu[2]
        imu_data.angular_velocity.x = imu[3]
        imu_data.angular_velocity.y = imu[4]
        imu_data.angular_velocity.z = imu[5]
        
        self.imu_publisher.publish(imu_data)
        
    def publish_stereo_imu(self, left_image, right_image, imu):
        """
        Publishes front stereo pair
        """
        # imgL = cv2.resize(left_image, (480, 270))
        # imgR = cv2.resize(left_image, (480, 270))
        
        imgL = self.bridge.cv2_to_imgmsg(left_image, encoding="mono8")
        imgR = self.bridge.cv2_to_imgmsg(right_image, encoding="mono8")
        
        imgL.header.stamp = self.get_clock().now().to_msg()
        imgR.header.stamp = imgL.header.stamp
        imgL.header.frame_id = "left_camera"
        imgR.header.frame_id = "right_camera"
        
        img_pair = ImagePair()
        img_pair.left = imgL
        img_pair.right = imgR
        
        imu_data = Imu()
        imu_data.linear_acceleration.x = imu[0]
        imu_data.linear_acceleration.y = imu[1]
        imu_data.linear_acceleration.z = imu[2]
        imu_data.angular_velocity.x = imu[3]
        imu_data.angular_velocity.y = imu[4]
        imu_data.angular_velocity.z = imu[5]
        imu_data.header.stamp = imgL.header.stamp
        imu_data.header.frame_id = "imu_link"
        
        stereo_imu_msg = StereoIMU()
        stereo_imu_msg.img_pair = img_pair
        stereo_imu_msg.imu = imu_data
        
        self.stereo_imu_publisher.publish(stereo_imu_msg)
        print("Published stereo imu pair")
        
    def publish_left_image(self, image):
        imgL = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
        self.left_img_publisher.publish(imgL)

    def publish_initial_pose(self, initial_pose):
        # convert into ros2 pose with covariance stamped
        pose = PoseWithCovarianceStamped()
        pose.pose.pose.position.x = initial_pose.location.x
        pose.pose.pose.position.y = initial_pose.location.y
        pose.pose.pose.position.z = initial_pose.location.z
        r = R.from_euler('xyz', [initial_pose.rotation.roll, initial_pose.rotation.pitch, initial_pose.rotation.yaw], degrees=False)
        quat = r.as_quat()

        pose.pose.pose.orientation.x = quat[0]
        pose.pose.pose.orientation.y = quat[1]
        pose.pose.pose.orientation.z = quat[2]
        pose.pose.pose.orientation.w = quat[3]

        pose.pose.covariance = [0.0] * 36

        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()

        # publish to initial pose topic
        self.initial_pose_pub.publish(pose)
        
class AgentV1(AutonomousAgent):

    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf):
        """
        Setup the agent parameters
        """
        
        # initialize rclpy from here
        rclpy.init()

        self.pos = np.array([0.0, 0.0, 0.0])
        self.rpy = np.array([0.0, 0.0, 0.0])

        self.gyro_queue = []
        self.gyro = np.array([0.0, 0.0, 0.0])
        
        # initialize publishing and listening node
        self.agent_node = AgentNode()
        
        # launch relevant nodes
        self.launch_nodes()
        
        self._active_side_cameras = False
        
        self.frame = 0
        self.linear_target_speed = 0
        self.angular_target_speed = 0
        self._linear_speed_increase = 0.02
        self._angular_speed_increase = 0.02
        self._max_linear_speed = 0.4
        self._max_angular_speed = 0.5

        self.last_time = 0
        self.first_epoch = True

    def use_fiducials(self):
        return False
    
    def launch_nodes(self):
        """
        Starts all nodes related to the agent.
        """

        # ros2 launch lac_launch agent_v1.launch.py
        # launch_file_command = ["ros2", "launch", "lac_launch", "agent_v1.launch.py"]
        # subprocess.Popen(launch_file_command)

        # ros2 run perception yolo_seg_detector
        # yolo_seg_command = ["ros2", "run", "perception", "yolo_seg_detector"]
        # subprocess.Popen(yolo_seg_command)

        # # ros2 run perception yolo_seg_detector
        # cam_info_pub_command = ["ros2", "run", "perception", "camera_info_publisher"]
        # subprocess.Popen(cam_info_pub_command)

        # ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
        # isaac_slam_command = ["ros2", "launch", "isaac_ros_visual_slam", "isaac_ros_visual_slam.launch.py"]
        # subprocess.Popen(isaac_slam_command)

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        
        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048) 
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """
        
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 1.0, 'width': '2440', 'height': '2048'
            },
        }
        return sensors

    def run_step(self, input_data):
        gyro = np.array(self.get_imu_data()[3:])

        # update rpy
        # if (self.first_epoch):
        #     self.last_time = (self.agent_node.get_clock().now().nanoseconds / 1e9) + self.agent_node.get_clock().now().seconds
        #     self.gyro_queue.append(self.gyro)
        #     self.first_epoch = False
        # else:
        #     now = (self.agent_node.get_clock().now().nanoseconds / 1e9) + self.agent_node.get_clock().now().seconds

        #     dt = now - self.last_time
        #     self.last_time = (self.agent_node.get_clock().now().nanoseconds / 1e9) + self.agent_node.get_clock().now().seconds

        #     # add to gyroscope queue
        #     if (len(self.gyro_queue) == 5):
        #         self.gyro_queue.pop(0)
        #     self.gyro_queue.append(gyro)

        #     self.rpy += dt * gyro # update roll, pitch, yaw

        # publish initial pose periodically
        self.agent_node.publish_initial_pose(self.get_initial_position())

        control = carla.VehicleVelocityControl(0, 0)
        
        """Get user input for driving"""
        
        self.set_light_state(carla.SensorPosition.Front, 1.0)
        
        """Execute one step of navigation"""
        
        self.set_front_arm_angle(np.deg2rad(60))
        # self.set_back_arm_angle(np.deg2rad(60))
        
        # publish front stereo image data
        front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        front_stereo_img_data = [front_left_data, front_right_data]

        if front_left_data is not None and front_right_data is not None:
            front_left_data = cv2.resize(np.asarray(front_left_data), (IMG_WIDTH, IMG_HEIGHT))
            front_right_data = cv2.resize(np.asarray(front_right_data), (IMG_WIDTH, IMG_HEIGHT))
            front_stereo_img_data = [front_left_data, front_right_data]

            cv2.imshow("Front Left", front_stereo_img_data[0])
            cv2.waitKey(1)

            if self.get_front_arm_angle() >= np.deg2rad(55):
                self.agent_node.publish_front_img_pair(front_stereo_img_data[0], front_stereo_img_data[1])
                self.agent_node.publish_slam_images(front_stereo_img_data[0], front_stereo_img_data[1])

        # if self.get_front_arm_angle() >= np.deg2rad(55):
        #     control = carla.VehicleVelocityControl(0.3, 0.3)
        # else:
        #     # collect and publish initial pose
        #     initial_pose = self.get_initial_position()
        #     self.agent_node.publish_initial_pose(initial_pose)

        print(self.get_mission_time())
        
        """Spin node once to update pose information"""
        
        # if rclpy.ok():
        #     rclpy.spin_once(self.agent_node, timeout_sec=0.1)

        return control

    def finalize(self):        
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))

        # end ros core
        rclpy.shutdown()
                

        
# if __name__ == "__main__":
#     rclpy.init()
#     agent_node = AgentNode()
#     rclpy.shutdown()