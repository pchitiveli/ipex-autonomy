import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2, PointField
from lac_interfaces.msg import PoseStampedImagePair

from cv_bridge import CvBridge

import open3d as o3d

import torch
import copy

INITIAL_WIDTH = 1280
INITIAL_HEIGHT = 720
IMG_WIDTH = 480
IMG_HEIGHT = 270
FIELD_OF_VIEW = 70.0  # degrees
BASELINE = 0.162
DISPARITY_OCCLUSION_SLOPE = 0.6639344262
DISPARITY_OCCLUSION_YINT = -71.6066

MAX_DISPARITY = 304 * (
    float(IMG_WIDTH) / INITIAL_WIDTH
)  # actually 315, but num_disp has to be divisible by 16
MAX_DISPARITY = int(MAX_DISPARITY - (MAX_DISPARITY % 16))

FOV_X = 70.0
FOV_Y = 39.375

FX = float(IMG_WIDTH) / (2.0 * np.tan(np.deg2rad(FOV_X)))
FY = float(IMG_HEIGHT) / (2.0 * np.tan(np.deg2rad(FOV_Y)))

FX_MSG = float(IMG_WIDTH) / (2.0 * np.tan(np.deg2rad(FOV_X / 2.0)))
FY_MSG = float(IMG_HEIGHT) / (2.0 * np.tan(np.deg2rad(FOV_Y / 2.0)))

CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0

BASELINE = 0.162

# CAMERA INFO SETUP -----------------------------------------------------------------------------------
camera_info_msg = CameraInfo()
camera_info_msg.width = IMG_WIDTH
camera_info_msg.height = IMG_HEIGHT
camera_info_msg.distortion_model = "plumb_bob" # sim has perfect cameras
camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # 0 distortion coefficients b/c perfect cams

# intrinsic camera matrix (p)
# [ fx, 0,  cx ]
# [ 0,  fy, cy ]
# [ 0,  0,  1 ]
#
# fx, fy = focal lengths for each axis
# cx, cy = principal point (center)

camera_info_msg.k = [FX_MSG, 0.0, CX,
                     0.0, FY_MSG, CY,
                     0.0, 0.0, 1.0]

# set rectification matrix to identity matrix bc perfect images
camera_info_msg.r = [1.0, 0.0, 0.0,
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

camera_info_msg.p = [FX_MSG, 0.0, CX, 0.0,
                     0.0, FY_MSG, CY, 0.0,
                     0.0, 0.0, 1.0, 0.0]

camera_info_msg.roi.width = 480
camera_info_msg.roi.height = 270

class SlamPCPublisher(Node):
    def __init__(self):
        super().__init__("slam_pc_publisher")
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # ROS info flow setup
        self.bridge = CvBridge()

        self.img_pair_listener = self.create_subscription(
            PoseStampedImagePair,
            "/front_stereo_img_pair",
            self.image_callback,
            10,
        )

        self.pc_pub = self.create_publisher(PointCloud2, "/pc", 10)
        self.depth_pub = self.create_publisher(Image, "/front/depth", 10)
        self.rgb_pub = self.create_publisher(Image, "/front/rgb", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/front/camera_info", 10)

    # ==========================================================================================================
    # UTIL METHODS - MOVE TO AN ACTUAL UTIL PACKAGE BC HOLY SCUFFED
    # ==========================================================================================================

    def calculate_disparity_sgbm(self, imgL, imgR) -> np.array:
        """
        Calculates disparity map using semi-global block matching method.

        Args:
            imgL (np.ndarray): grayscale left image
            imgR (np.ndarray): grayscale right image
        Returns:
            disparity: numpy array of pixel disparities for each pixel in imgL
        """

        max_disp = int(
            MAX_DISPARITY * (imgL.shape[1] / IMG_WIDTH)
        )
        max_disp = int(max_disp - (max_disp % 16))

        imgL_padded = cv2.copyMakeBorder(
            imgL, 0, 0, max_disp, 0, cv2.BORDER_CONSTANT, value=0
        )
        imgR_padded = cv2.copyMakeBorder(
            imgR, 0, 0, max_disp, 0, cv2.BORDER_CONSTANT, value=0
        )

        # SGBM Parameters
        block_size = 3
        min_disp = 0
        num_channels = 1  # grayscale
        p1 = (
            8 * num_channels * (block_size**2)
        )  # common heuristic for calculating p1 - penalty for small disparity difference b/w neighboring pixels
        p2 = (
            32 * num_channels * (block_size**2)
        )  # common heuristic for calculating p2 - penalty for large disparity difference b/w neighboring pixels

        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            # minimum possible disparity value (0 for perfect sim, may be diff for rectified/distorted images)
            numDisparities=max_disp,
            # max disp - min disp. max disp is the max possible disparity value (for objects close up) - must be divisible by 16 in current opencv implementation for some reason
            blockSize=block_size,
            # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
            P1=p1,
            P2=p2,
            disp12MaxDiff=max_disp,
            # Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
            uniquenessRatio=10,
            speckleWindowSize=16,
            # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
            speckleRange=1,
            # Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
            preFilterCap=1,  # clamp value for x-derivative
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # divide by 16 to get actual disparity values
        disparity = stereo_sgbm.compute(imgL_padded, imgR_padded) / 16.0
        # disparity = disparity[:, MAX_DISPARITY:] # crop out the leftmost part of the disparity map
        return disparity[:, max_disp:]
    
    def disparity_to_depth_map(self, disparity: np.array, imgL: np.ndarray) -> np.array:
        """
        Converts disparity array to a depth map indicating the depth of each pixel.

        Args:
            disparity (np.array): disparity map
            imgL (np.ndarray): left image in grayscale form
        Return:
            depth_map (np.array): numpy array of depth
        """

        rows, cols = np.indices(disparity.shape)
        focal_length_x = imgL.shape[1] / (
            2 * np.tan(np.radians(FIELD_OF_VIEW / 2))
        )

        disparity = np.where(disparity > 1e-6, disparity, 0)
        depth_map = np.where(  # checks that image color is not pitch black at pixel location as well
            (disparity > 1e-6)
            & (imgL > 0)
            & (
                cols
                > (DISPARITY_OCCLUSION_SLOPE * rows)
                + (
                    DISPARITY_OCCLUSION_YINT
                    * (disparity.shape[0] / IMG_HEIGHT)
                )
            ),
            (focal_length_x * BASELINE) / disparity,
            0,
        )
        # filter out depth values greater than 20m
        # depth_map = np.where(depth_map > 20, float("inf"), depth_map)

        return depth_map

    def compute_depth_image(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
    ):
        """
        Computes depth image from input grayscale images.

        Args:
            imgL (np.ndarray) : left opencv image
            imgR (np.ndarray) : right opencv image
        """
        # calculate the disparities between pixels first
        disparity = None
        disparity = self.calculate_disparity_sgbm(imgL, imgR)

        # convert disparity map to a depth map for more useful info
        depth_map = self.disparity_to_depth_map(disparity, imgL)

        return depth_map
    
    def depth_map_to_points(
        self,
        depth_map: np.array,
        imgL: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Converts depth map numpy array into points and colors np arrays.

        Args:
            depth_map (np.array): depth numpy array
            imgL (np.array): left image in grayscale
            origin_pos (np.array): position of the camera in [x, y, z]
            origin_rpy (list-like): origin orientation of the camera in [roll, pitch, yaw]
        Return:
            points, colors - list-like, numpy arrays of matching points and colors
        """

        focal_length_x = imgL.shape[1] / (
            2 * np.tan(np.radians(FIELD_OF_VIEW / 2))
        )
        focal_length_y = imgL.shape[0] / (2 * np.tan(np.radians(39.375 / 2)))

        principal_point = (imgL.shape[1] / 2.0, imgL.shape[0] / 2.0)

        depth_map = torch.from_numpy(depth_map)  # convert depth map into tensor space
        imgL = (
            torch.from_numpy(imgL).float() / 255.0
        )  # convert image to tensor space and normalize image to within 0 to 1 pixel intensity

        cols, rows = torch.meshgrid(
            torch.arange(depth_map.shape[0]),
            torch.arange(depth_map.shape[1]),
            indexing="ij",
        )

        mask = depth_map != 0
        depth_map = depth_map[mask]
        x_indices = rows[mask]
        y_indices = cols[mask]

        Z = depth_map
        X = (x_indices - principal_point[0]) * depth_map / focal_length_x
        Y = (y_indices - principal_point[1]) * depth_map / focal_length_y
        points = torch.stack([X, Y, Z], dim=1)
        
        points = points.numpy()

        colors = imgL[mask]

        return points, colors.numpy()
    
    def compute_point_cloud_from_depth(
        self,
        imgL: np.ndarray,
        depth_map: np.ndarray,
        voxelize=True,
        voxel_size=0.02
    ):
        """
        Compute point cloud from stereo image input using Semi-Global Block Matching disparity calculation method.

        Args:
            imgL: grayscale left image
            imgR: grayscale right image
            origin_pos (list-like): camera position (x, y, z)
            origin_rpy (list-like): camera orientation (roll, pitch, yaw).
            disparity_method (str): allows for distinction between block matching and semi-global block matching
        Return:
            pc (open3d.geometry.PointCloud): point cloud of stereo image data
        """

        # profiler = cProfile.Profile()
        # profiler.enable()

        imgL = cv2.cvtColor(
            imgL,
            # convert grayscale opencv image to RGB for point cloud color assignment
            cv2.COLOR_GRAY2RGB,
        )

        points, colors = self.depth_map_to_points(
            depth_map,
            imgL,
        )

        # convert to open3d point cloud for downsampling
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        o3d_pc.colors = o3d.utility.Vector3dVector(colors)

        if voxelize:
            o3d_pc = o3d_pc.voxel_down_sample(
                voxel_size=voxel_size
            )  # downsample to voxel size of 2cm

        # o3d_pc, _ = o3d_pc.remove_radius_outlier(nb_points=20, radius=0.1)

        return o3d_pc
    
    def o3d_pc_to_point_cloud2(self, stamp_msg, o3d_pc) -> PointCloud2:
        """
        Converts an Open3D point cloud to a ROS2 PointCloud2 message.

        Args:
            stamp_msg (Header msg): timestamp as a header message
            o3d_pc (o3d.geometry.PointCloud): pointcloud to convert
        Return:
            PointCloud2 message representing the o3d point cloud.
        """

        points = np.asarray(o3d_pc.points, dtype=np.float32)
        colors = np.asarray(o3d_pc.colors, dtype=np.float32)

        msg = PointCloud2()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp_msg

        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = True  # no invalid values
        msg.is_bigendian = False
        msg.point_step = 16  # 16 bytes per point
        msg.row_step = msg.point_step * points.shape[0]  # byte length of row

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]

        # convert colors from 0-1 RGB to 0-1 intensity
        colors = colors * 255
        intensity = np.dot(colors, [0.2989, 0.5870, 0.1140]).astype(np.float32)

        # make numpy structured array of points
        structured_data = np.zeros(
            points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.uint32),
            ],
        )
        structured_data["x"] = points[:, 0]  # populate x-coords
        structured_data["y"] = points[:, 1]  # populate y coords
        structured_data["z"] = points[:, 2]  # populate z coords

        # populate intensity vals with 32-bit ints
        structured_data["intensity"] = intensity.flatten()

        msg.data = structured_data.tobytes()  # reads byte data when processing pc
        return msg
    
    def add_point_clouds(
        self, pc1: o3d.geometry.PointCloud, pc2: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """
        Add two point clouds together.

        Args:
            pc1 (open3d.geometry.PointCloud): first point cloud to add
            pc2 (open3d.geometry.PointCloud): second point cloud to add
        Return:
            aggregated point cloud from two component
        """

        pc1_points = np.asarray(pc1.points)
        pc1_colors = np.asarray(pc1.colors)
        pc2_points = np.asarray(pc2.points)
        pc2_colors = np.asarray(pc2.colors)

        points = np.concatenate((pc1_points, pc2_points), axis=0)
        colors = np.concatenate((pc1_colors, pc2_colors), axis=0)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        return pc

    # =========================================================================================================
    # CALLBACKS
    # =========================================================================================================

    def image_callback(self, msg):
        if msg is not None and msg.header.stamp is not None:
            # expecting grayscale images
            imgL = self.bridge.imgmsg_to_cv2(msg.image_pair.left, desired_encoding="mono8")
            imgR = self.bridge.imgmsg_to_cv2(msg.image_pair.right, desired_encoding="mono8")

            if imgL is not None:
                if imgL.shape is not (IMG_WIDTH, IMG_HEIGHT):
                    imgL = cv2.resize(imgL, (IMG_WIDTH, IMG_HEIGHT))
                    imgR = cv2.resize(imgR, (IMG_WIDTH, IMG_HEIGHT))
                
                # publish depth message
                depth = self.compute_depth_image(imgL, imgR)
                depth_mm = (depth * 1000).astype(np.uint16)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = 'fl_camera'

                # publish left image message
                rgb_img = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
                rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
                rgb_img_msg.header = depth_msg.header
                rgb_img_msg.header.frame_id = 'fl_camera'

                # publish custom point cloud
                # pc = self.compute_point_cloud_from_depth(
                #     imgL, depth, voxel_size=0.05
                # )
                # pc_msg = self.o3d_pc_to_point_cloud2(depth_msg.header.stamp, pc)
                # pc_msg.header.frame_id = "fl_camera"

                # update header of camera info
                camera_info_msg.header = depth_msg.header
                camera_info_msg.header.frame_id = 'fl_camera'
                
                # publish all at the same time
                # self.pc_pub.publish(pc_msg)
                self.rgb_pub.publish(rgb_img_msg)
                self.depth_pub.publish(depth_msg)
                self.camera_info_pub.publish(camera_info_msg)

def main():
    rclpy.init()
    node = SlamPCPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
