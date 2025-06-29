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

import pygame
import pygame.locals as pykeys

import os
import csv

def get_entry_point():
    return 'AgentV1'
        

class AgentV1(AutonomousAgent):

    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        
        # launch relevant nodes
        # self.launch_nodes()
        
        self._active_side_cameras = False
        
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        self.frame = 0
        self.linear_target_speed = 0
        self.angular_target_speed = 0
        self._linear_speed_increase = 0.02
        self._angular_speed_increase = 0.02
        self._max_linear_speed = 0.4
        self._max_angular_speed = 0.5
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter('out/rockImageSamples/seg_images/raw_img/sharpened_vid.mp4', fourcc, 20, (1280, 720))
        
        pygame.init()
        
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
        
        # TODO: PRANAV IMG SEGMENTATION DATA COLLECTION
        self.data_folder = os.path.expanduser('~/nasa_lac/LunarAutonomyChallenge/out/rockImageSamples/seg_images/raw_img/new')
        self.train_folder = os.path.join(self.data_folder, "train")
        self.val_folder = os.path.join(self.data_folder, "val")
        
        self.train_semantic_folder = os.path.join(self.train_folder, "semantic")
        self.val_semantic_folder = os.path.join(self.val_folder, "semantic")
        
        self.train_real_folder = os.path.join(self.train_folder, "real")
        self.val_real_folder = os.path.join(self.val_folder, "real")
        # TODO: END PRANAV IMG SEGMENTATION DATA COLLECTION


    def use_fiducials(self):
        return True

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
        """Execute one step of navigation"""
        
        if self.frame == 0:
            self.set_front_arm_angle(np.deg2rad(60))
            self.set_back_arm_angle(np.deg2rad(60))
        
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
        
        # publish front stereo image data
        front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        front_stereo_img_data = [front_left_data, front_right_data]
        
        true_pose = self.get_transform()
        assert(true_pose is not None)
        
        # TODO: PRANAV IMG SEGMENTATION DATA COLLECTION
        if self.get_front_arm_angle() >= np.deg2rad(50) and self.get_back_arm_angle() >= np.deg2rad(50):
            if front_left_data is not None:
                # sharpen image
                front_left_data = cv2.filter2D(front_left_data, -1, self.sharpen_kernel)

                # save image
                front_left_data = cv2.cvtColor(front_left_data, cv2.COLOR_GRAY2BGR)
                self.video.write(front_left_data)
            # print(input_data['Semantic'])
            # semantic = input_data['Semantic']
            
            # if len(semantic) != 0:
            #     print("SEMANTIC!")
            #     front_left_semantic = semantic[carla.SensorPosition.FrontLeft]
            #     front_right_semantic = input_data['Semantic'][carla.SensorPosition.FrontRight]
                
            #     back_left_data = input_data['Grayscale'][carla.SensorPosition.BackLeft]
            #     back_right_data = input_data['Grayscale'][carla.SensorPosition.BackRight]
            #     back_left_semantic = input_data['Semantic'][carla.SensorPosition.BackLeft]
            #     back_right_semantic = input_data['Semantic'][carla.SensorPosition.BackRight]
                
            #     fl_real_path = os.path.join(self.train_real_folder, f"{self.frame}_FL.png")
            #     fl_semantic_path = os.path.join(self.train_semantic_folder, f"{self.frame}_FL.png")
            #     fr_real_path = os.path.join(self.train_real_folder, f"{self.frame}_FR.png")
            #     fr_semantic_path = os.path.join(self.train_semantic_folder, f"{self.frame}_FR.png")
                
            #     bl_real_path = os.path.join(self.train_real_folder, f"{self.frame}_BL.png")
            #     bl_semantic_path = os.path.join(self.train_semantic_folder, f"{self.frame}_BL.png")
            #     br_real_path = os.path.join(self.train_real_folder, f"{self.frame}_BR.png")
            #     br_semantic_path = os.path.join(self.train_semantic_folder, f"{self.frame}_BR.png")
                    
            #     if (self.frame % 5 == 0): # populate validation data - 80-20 split
            #         fl_real_path = os.path.join(self.val_real_folder, f"{self.frame}_FL.png")
            #         fl_semantic_path = os.path.join(self.val_semantic_folder, f"{self.frame}_FL.png")
            #         fr_real_path = os.path.join(self.val_real_folder, f"{self.frame}_FR.png")
            #         fr_semantic_path = os.path.join(self.val_semantic_folder, f"{self.frame}_FR.png")
                    
            #         bl_real_path = os.path.join(self.val_real_folder, f"{self.frame}_BL.png")
            #         bl_semantic_path = os.path.join(self.val_semantic_folder, f"{self.frame}_BL.png")
            #         br_real_path = os.path.join(self.val_real_folder, f"{self.frame}_BR.png")
            #         br_semantic_path = os.path.join(self.val_semantic_folder, f"{self.frame}_BR.png")
                    
            #     # save data
                # if front_left_data is not None:
                #     cv2.imwrite(fl_real_path, front_left_data)
            #     if front_left_semantic is not None:
            #         cv2.imwrite(fl_semantic_path, front_left_semantic)
                    
            #     if front_right_data is not None:
            #         cv2.imwrite(fr_real_path, front_right_data)
            #     if front_right_semantic is not None:
            #         cv2.imwrite(fr_semantic_path, front_right_semantic)
                    
            #     if back_left_data is not None:
            #         cv2.imwrite(bl_real_path, back_left_data)
            #     if back_left_semantic is not None:
            #         cv2.imwrite(bl_semantic_path, back_left_semantic)
                    
            #     if back_right_data is not None:
            #         cv2.imwrite(br_real_path, back_right_data)
            #     if back_right_semantic is not None:
            #         cv2.imwrite(br_semantic_path, back_right_semantic)
        # TODO: END PRANAV SEGMENTATION DATA COLLECTION
        
        # true_pose = self.get_transform()
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
                # self.agent_node.publish_true_pose(pose_arr)
                
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
            control = carla.VehicleVelocityControl(0.1, 0)

            # publish front stereo image data
            front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
            front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]
            front_stereo_img_data = [front_left_data, front_right_data]
            
            # if front_stereo_img_data[0] is not None and front_stereo_img_data[1] is not None and rclpy.ok():
            #     self.agent_node.publish_front_img_pair(front_stereo_img_data[0], front_stereo_img_data[1])
            #     self.agent_node.publish_stereo_imu(front_stereo_img_data[0], front_stereo_img_data[1], self.get_imu_data())
                
        else:
            control = carla.VehicleVelocityControl(0, 0)
        
        # front_data = input_data['Grayscale'][carla.SensorPosition.FrontRight]  # Do something with this

        # check if condition for ending run has been met
        # mission_time = round(self.get_mission_time(), 2)
        print(self.get_mission_time())

        # if float(self.get_mission_time()) > 60.0:
        #     print("stop condition met")
        #     self.mission_complete()
        
        # TODO - FOR DATA COLLECTION PURPOSES MAINLY - REMOVE BEFORE FINAL
        self.frame += 1

        return control

    def finalize(self):
        # end ros core
        # rclpy.shutdown()
        
        self.video.release()
        cv2.destroyAllWindows()
        
        # g_map = self.get_geometric_map()
        # map_length = g_map.get_cell_number()
        # for i in range(map_length):
        #     for j in range(map_length):
        #         g_map.set_cell_height(i, j, random.normal(0, 0.5))
        #         g_map.set_cell_rock(i, j, bool(random.randint(2)))
                

        
# if __name__ == "__main__":
#     rclpy.init()
#     agent_node = AgentNode()
#     rclpy.shutdown()