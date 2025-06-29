import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from lac_interfaces.msg import ImagePair, StereoIMU, EulerPose
from geometry_msgs.msg import PoseStamped, Vector3
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class LocalizationNode(Node):
    def __init__(self, init_pose=None):
        super().__init__('localization_node')

        self.imu_pub = self.create_publisher(Imu, 'imu', 10)
        self.left_img_pub = self.create_publisher(Image, 'camera/left', 10)
        self.right_img_pub = self.create_publisher(Image, 'camera/right', 10)
        self.pose_publisher = self.create_publisher(EulerPose,'/localization/pose', 10)
        self.raw_pose_publisher = self.create_publisher(PoseStamped, 'localization/raw_pose', 10)

        self.pose_sub = self.create_subscription(PoseStamped, 'orbslam/pose', self.pose_callback, 10)
        self.pose_sub = self.create_subscription(StereoIMU, '/stereo_imu', self.stereo_imu_callback, 10)

        self.pose_list = []
        self.prev_pose = None
        self.curr_pose = init_pose
        # self.pose_msg_tmp = None

        # self.timer = self.create_timer(0.05, self.publish_data)

        self.bridge = CvBridge()


    def pose_callback(self, msg):
        if msg is None:
            return

        pose = {    
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
            "qx": msg.pose.orientation.x,
            "qy": msg.pose.orientation.y,
            "qz": msg.pose.orientation.z,
            "qw": msg.pose.orientation.w
        }

        # print(pose)

        self.pose_list.append(pose)
        self.get_logger().info(f"Updated pose list: {len(self.pose_list)} entries")

        position = Vector3()
        position.x = pose['x']
        position.y = pose['y']
        position.z = pose['z']

        # Create a Rotation object from the quaternion
        rotation = R.from_quat([
            pose['qx'],
            pose['qy'],
            pose['qz'],
            pose['qw']
        ])

        # Get Euler angles in radians
        euler_angles = rotation.as_euler('xyz', degrees=False)  # Set to degrees=True if you want degrees

        # You can convert to degrees if needed
        # euler_angles = np.degrees(euler_angles)  # Uncomment if you want degrees

        orientation = Vector3()
        orientation.x = euler_angles[0]
        orientation.y = euler_angles[1]
        orientation.z = euler_angles[2]

        # Prepare and publish the EulerPose message
        euler_msg = EulerPose()
        euler_msg.position = position
        euler_msg.orientation = orientation

        self.pose_publisher.publish(euler_msg)
        
        ### UPDATE RELATIVE POSE LIST
        if self.prev_pose is None:
            self.prev_pose = pose
            self.get_logger().info("First pose received, setting previous pose")
            return

        # Calculate the difference between the current and previous poses
        delta_position = np.array([
            pose['x'] - self.prev_pose['x'],
            pose['y'] - self.prev_pose['y'],
            pose['z'] - self.prev_pose['z']
        ])
        
        # Check if the pose difference is too large 
        threshold = 0.05  
        if np.linalg.norm(delta_position) > threshold:
            self.get_logger().info("Pose difference too large, resetting previous pose.")
            print("Got lost again :/")
            self.prev_pose = pose  
            return
        
        relative_position = delta_position
        relative_rotation = R.from_quat([
            pose['qx'], pose['qy'], pose['qz'], pose['qw']
        ]).inv() * R.from_quat([
            self.prev_pose['qx'], self.prev_pose['qy'], self.prev_pose['qz'], self.prev_pose['qw']
        ])
        
        p1 = np.eye(4)
        p2 = np.eye(4)

        p1[:3, 3] = [self.prev_pose['x'], self.prev_pose['y'], self.prev_pose['z']]        
        p2[:3, 3] = [pose['x'], pose['y'], pose['z']]

        R1 = R.from_quat([
            self.prev_pose['qx'], self.prev_pose['qy'], self.prev_pose['qz'], self.prev_pose['qw']
        ]).as_matrix()
        
        R2 = R.from_quat([
            pose['qx'], pose['qy'], pose['qz'], pose['qw']
        ]).as_matrix()
        
        p1[:3, :3] = R1
        p2[:3, :3] = R2

        relative_pose = np.linalg.inv(p1) @ p2
        
        # convert to correct coordinate frame
        '''
        robot:
        x,y,z
        
        x -> -z
        y -> -x
        z -> y
        
        or 
        
        x -> -y
        y -> z
        z -> -x
        '''
        
        R_transform = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])
        
        t_vec = relative_pose[:3, 3]
        tvec_robot = R_transform @ t_vec
        
        rotation = R.from_matrix(relative_pose[:3, :3])
        euler_angles = rotation.as_euler('xyz', degrees=False)
        euler_angles[0] = euler_angles[0] * -1
        euler_angles[1] = euler_angles[1] * 1
        euler_angles[2] = euler_angles[2] * -1
        print(euler_angles * 180 / np.pi)
        new_rotation = R.from_euler('yzx', euler_angles, degrees=False).as_matrix()
        
        relative_pose = np.eye(4)
        relative_pose[:3,:3] = new_rotation
        relative_pose[:3, 3] = tvec_robot

        # Update the current pose by adding the relative transformation
        if self.curr_pose is None:
            self.curr_pose = {
                "x": self.prev_pose['x'] + relative_position[0],
                "y": self.prev_pose['y'] + relative_position[1],
                "z": self.prev_pose['z'] + relative_position[2],
                "qx": relative_rotation.as_quat()[0],
                "qy": relative_rotation.as_quat()[1],
                "qz": relative_rotation.as_quat()[2],
                "qw": relative_rotation.as_quat()[3]
            }
        else:
            # relative_rotation = relative_rotation * R.from_quat([
            #     self.curr_pose['qx'], self.curr_pose['qy'], self.curr_pose['qz'], self.curr_pose['qw']
            # ])
            
            p_curr = np.eye(4)
            p_curr[:3, 3] = [self.curr_pose['x'], self.curr_pose['y'], self.curr_pose['z']]
            R_curr = R.from_quat([
                self.curr_pose['qx'], self.curr_pose['qy'], self.curr_pose['qz'], self.curr_pose['qw']
            ]).as_matrix()
            p_curr[:3, :3] = R_curr
            
            p_new = p_curr @ relative_pose
            q_new = R.from_matrix(p_new[:3,:3]).as_quat()
            
            self.curr_pose = {
                "x": p_new[0, 3],
                "y": p_new[1, 3],
                "z": p_new[2, 3],
                "qx": q_new[0],
                "qy": q_new[1],
                "qz": q_new[2],
                "qw": q_new[3]
            }
            
            # self.curr_pose = {
            #     "x": self.curr_pose['x'] + relative_position[0],
            #     "y": self.curr_pose['y'] + relative_position[1],
            #     "z": self.curr_pose['z'] + relative_position[2],
            #     "qx": relative_rotation.as_quat()[0],
            #     "qy": relative_rotation.as_quat()[1],
            #     "qz": relative_rotation.as_quat()[2],
            #     "qw": relative_rotation.as_quat()[3]
            # }

        # Publish the updated pose
        self.prev_pose = pose  # Update the previous pose
        
        # self.curr_pose = pose
        self.publish_raw_pose()
    
    
    def publish_raw_pose(self):
        # Publish the current pose as PoseStamped message
        print(f"RAW POSE PRINTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {self.curr_pose}")
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = self.curr_pose['x']
        pose_stamped.pose.position.y = self.curr_pose['y']
        pose_stamped.pose.position.z = self.curr_pose['z']
        pose_stamped.pose.orientation.x = self.curr_pose['qx']
        pose_stamped.pose.orientation.y = self.curr_pose['qy']
        pose_stamped.pose.orientation.z = self.curr_pose['qz']
        pose_stamped.pose.orientation.w = self.curr_pose['qw']
        pose_stamped.header.frame_id = "map"
        
        self.raw_pose_publisher.publish(pose_stamped)
        # if self.pose_msg_tmp is not None:
        #     self.raw_pose_publisher.publish(self.pose_msg_tmp)
    
        
    def stereo_imu_callback(self, msg):
        if msg is None:
            return
        
        print(f"Message Received {msg.imu.header.stamp}")
        img_pair = msg.img_pair
        imu_data = msg.imu
        
        print(f"imu_data {imu_data.linear_acceleration.x } ")
        
        image_l = img_pair.left
        image_r = img_pair.right
        
        self.imu_pub.publish(imu_data)
        self.left_img_pub.publish(image_l)
        self.right_img_pub.publish(image_r)
        
        
        

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()