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
from geometry_msgs.msg import Pose, PoseStamped
import launch
import launch_ros.actions

import pygame
import pygame.locals as pykeys

import os
import csv
import numpy as np

import subprocess


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
        
        self.pos = np.array([0.0, 0.0, 0.0])
        self.rpy = np.array([0.0, 0.0, 0.0])

        # PUBLISHERS ------------------------------------------------------------------------------------------
        # Perception Publishers ------
        self.front_stereo_publisher = self.create_publisher(PoseStampedImagePair, "/front_stereo_img_pair", 10)
        # self.back_stereo_publisher = self.create_publisher(ImagePair, "/back_stereo_img_pair", 10)
        self.imu_publisher = self.create_publisher(Imu, "/imu", 10)
        self.stereo_imu_publisher = self.create_publisher(StereoIMU, "/stereo_imu", 10)
        self.left_img_publisher = self.create_publisher(Image, "/left_img",  10)

        # Isaac ROS SLAM Image Publishers ------
        self.fl_img_publisher = self.create_publisher(Image, "visual_slam/image_0", 10)
        self.fr_img_publisher = self.create_publisher(Image, "visual_slam/image_1", 10)
        
        # TODO: ONLY FOR DEBUGGING
        # self.true_pose_publisher = self.create_publisher(Pose, "/pose", 10)
        
        # read pose from localization
        self.pose_sub = self.create_subscription(PoseStamped, "/localization/raw_pose", self.pose_callback, 10)
        
        self.declare_parameter("front_stereo_topic", "/front_stereo_img_pair")
        self.declare_parameter("back_stereo_topic", "/back_stereo_img_pair")
        
        self.bridge = CvBridge()
        
    # ==========================================================================================================
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
        

class AgentV1(AutonomousAgent):

    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        
        # initialize rclpy from here
        rclpy.init()
        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1,1))
        
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
        
        # TODO: AMAN LOCALIZATION DATA COLLECTION
        # self.data_folder = os.path.expanduser('~/aman/data')
        # self.stereo_folder = os.path.join(self.data_folder, "stereo_pairs")
        # self.pose_file = os.path.join(self.data_folder, "pose.csv")
        
        # # save pose data
        # os.makedirs(self.stereo_folder, exist_ok=True)
        
        # if not os.path.exists(self.pose_file):
        #     with open(self.pose_file, 'w') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["fname", "x", "y", "z", "roll", "pitch", "yaw"])
        # TODO: END AMAN LOCALIZATION DATA COLLECTION


    def use_fiducials(self):
        return True
    
    def launch_nodes(self):
        """
        Starts all nodes related to the agent.
        """

        yolo_seg_command = ["ros2", "run", "perception", "yolo_seg_detector"]
        subprocess.Popen(yolo_seg_command)

        # pc_gen_command = ["ros2", "run", "perception", "pc_generation_node"]
        # subprocess.Popen(pc_gen_command)

        # localization_command = ["ros2", "run", "localization", "localization_node"]
        # subprocess.Popen(localization_command)

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
        """Get user input for driving"""
        
        self.set_light_state(carla.SensorPosition.Front, 1.0)
        control = carla.VehicleVelocityControl(0, 0)
        
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                print("key event triggered")
                
                # Mission complete
                if event.key == pykeys.K_ESCAPE:
                    self.agent.mission_complete()

                # Help menu
                if event.key == pykeys.K_i:
                    self.render_help = not self.render_help

        keys = pygame.key.get_pressed()

        if keys[pykeys.K_UP] or keys[pykeys.K_w]:
            self.linear_target_speed = min(self.linear_target_speed + self._linear_speed_increase, self._max_linear_speed)
        elif keys[pykeys.K_DOWN] or keys[pykeys.K_s]:
            self.linear_target_speed = max(self.linear_target_speed - self._linear_speed_increase, -self._max_linear_speed)
        else:
            self.linear_target_speed = max(self.linear_target_speed - 2 * self._linear_speed_increase, 0.0)

        if keys[pykeys.K_RIGHT] or keys[pykeys.K_d]:
            self.angular_target_speed = max(self.angular_target_speed - self._angular_speed_increase, -self._max_angular_speed)
        elif keys[pykeys.K_LEFT] or keys[pykeys.K_a]:
            self.angular_target_speed = min(self.angular_target_speed + self._angular_speed_increase, +self._max_angular_speed)
        else:
            self.angular_target_speed = max(self.angular_target_speed - 2 * self._angular_speed_increase, 0.0)

        control = carla.VehicleVelocityControl(self.linear_target_speed, self.angular_target_speed)
        
        """Execute one step of navigation"""
        
        if self.frame == 0:
            self.set_front_arm_angle(np.deg2rad(60))
            self.set_back_arm_angle(np.deg2rad(60))
        
        # publish front stereo image data
        front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        front_stereo_img_data = [front_left_data, front_right_data]
        
        true_pose = self.get_transform()
        if front_stereo_img_data[0] is not None and front_stereo_img_data[1] is not None:
            if true_pose is not None:
                pose_arr = np.array([
                    true_pose.location.x,
                    true_pose.location.y,
                    true_pose.location.z,
                    true_pose.rotation.pitch,
                    true_pose.rotation.yaw,
                    true_pose.rotation.roll
                ])
                
                # TODO: testing purposes purely - DIDN'T WORK - need to be in testing mode
                self.agent_node.publish_true_pose(pose_arr)
                
                # TODO: AMAN LOCALIZATION DATA COLLECTION
                # left_path = os.path.join(self.stereo_folder, f"{self.frame}_L.png")
                # right_path = os.path.join(self.stereo_folder, f"{self.frame}_R.png")
                
                # cv2.imwrite(left_path, front_stereo_img_data[0])
                # cv2.imwrite(right_path, front_stereo_img_data[1])

                # with open(self.pose_file, 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([self.frame] + pose_arr.tolist())
            
            cv2.imshow("Front Left", front_stereo_img_data[0])
            cv2.waitKey(1)
        
        print(np.rad2deg(self.get_front_arm_angle()))
        print(np.rad2deg(self.get_back_arm_angle()))
        if self.get_front_arm_angle() >= np.deg2rad(50) and self.get_back_arm_angle() >= np.deg2rad(50):
            # control = carla.VehicleVelocityControl(0.2, 0)

            # publish front stereo image data
            front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
            front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
            front_stereo_img_data = [front_left_data, front_right_data]
            
            if front_stereo_img_data[0] is not None and front_stereo_img_data[1] is not None and rclpy.ok():
                self.agent_node.publish_front_img_pair(front_stereo_img_data[0], front_stereo_img_data[1])
                self.agent_node.publish_stereo_imu(front_stereo_img_data[0], front_stereo_img_data[1], self.get_imu_data())
                
        else:
            control = carla.VehicleVelocityControl(0, 0)
        
        # front_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]  # Do something with this

        # check if condition for ending run has been met
        # mission_time = round(self.get_mission_time(), 2)
        print(self.get_mission_time())

        # if float(self.get_mission_time()) > 60.0:
        #     print("stop condition met")
        #     self.mission_complete()
        
        # TODO - FOR DATA COLLECTION PURPOSES MAINLY - REMOVE BEFORE FINAL (could have overflow)
        # self.frame += 1
        
        """Spin node once to get information"""
        
        if rclpy.ok():
            rclpy.spin_once(self.agent_node, timeout_sec=0.1)

        return control

    def finalize(self):
        # end ros core
        rclpy.shutdown()
        
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
                

        
# if __name__ == "__main__":
#     rclpy.init()
#     agent_node = AgentNode()
#     rclpy.shutdown()