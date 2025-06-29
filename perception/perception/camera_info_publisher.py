import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo

import numpy as np

IMG_WIDTH = 1280
IMG_HEIGHT = 720

FOV_X = 70.0
FOV_Y = 39.375

FX = IMG_WIDTH / (2 * np.tan(np.radians(FOV_X)))
FY = IMG_HEIGHT / (2 * np.tan(np.radians(FOV_Y)))

CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__("camera_info_publisher_node")

        # CAMERA INFO SETUP -----------------------------------------------------------------------------------
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.width = 1280
        self.camera_info_msg.height = 720
        self.camera_info_msg.distortion_model = "none" # sim has perfect cameras
        self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # 0 distortion coefficients b/c perfect cams

        # intrinsic camera matrix (p)
        # [ fx, 0,  cx ]
        # [ 0,  fy, cy ]
        # [ 0,  0,  1 ]
        #
        # fx, fy = focal lengths for each axis
        # cx, cy = principal point (center)

        self.camera_info_msg.k = [float(FX), 0.0, float(CX),
                                  0.0, float(FY), float(CY),
                                  0.0, 0.0, 1.0]
        
        # set rectification matrix to identity matrix bc perfect images
        self.camera_info_msg.r = [1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0]

        # Projection matrix (p)
        # [ fx, 0,  cx, Tx ]
        # [ 0,  fy, cy, Ty ]
        # [ 0,  0,  1,  0 ]
        #
        # fx, fy = focal lengths for each axis
        # cx, cy = principal point (center)
        # Note: Tx = Ty = 0 when using monocular cameras

        self.camera_info_msg.p = [float(FX), 0.0, float(CX), 0.0,
                                  0.0, float(FY), float(CY), 0.0,
                                  0.0, 0.0, 1.0, 0.0]
        
        # PUBLISHER --------------------------------------------------------------------------------------------
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera_info", 10)

        # TIMER ------------------------------------------------------------------------------------------------
        frequency = 1 / 60.0
        self.timer = self.create_timer(frequency, self.timer_callback)
        
    def timer_callback(self):
        self.camera_info_pub.publish(self.camera_info_msg)

def main():
    rclpy.init()
    node = CameraInfoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()