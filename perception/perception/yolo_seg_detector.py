import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, PointCloud2, PointField
from lac_interfaces.msg import PoseStampedImagePair

from cv_bridge import CvBridge

import open3d as o3d

import ultralytics
import torch

class RockDetection(Node):
    INITIAL_WIDTH = 1280
    INITIAL_HEIGHT = 720
    IMG_WIDTH = 480
    IMG_HEIGHT = 270
    FIELD_OF_VIEW = 70  # degrees
    BASELINE = 0.162
    DISPARITY_OCCLUSION_SLOPE = 0.6639344262
    DISPARITY_OCCLUSION_YINT = -71.6066

    MAX_DISPARITY = 304 * (
        float(IMG_WIDTH) / INITIAL_WIDTH
    )  # actually 315, but num_disp has to be divisible by 16
    MAX_DISPARITY = int(MAX_DISPARITY - (MAX_DISPARITY % 16))

    def __init__(self):
        super().__init__("yolo_seg_node")

        self.rockSpecifications = []
        self.model = ultralytics.YOLO(
            # "/workspace/team_code/perception/models/rock-seg-1.pt"
            "/workspace/team_code/perception/models/model.pt"
        )
        self.model.conf = 0.3
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
        self.ground_pc_pub = self.create_publisher(PointCloud2, "/filtered_pc", 10)
        self.pc_pub = self.create_publisher(PointCloud2, "/pc", 10)
        self.filtered_img_pub = self.create_publisher(PoseStampedImagePair, "/filtered_stereo_img_pair", 10)

        # TODO: debugging - for checking pc against image
        # self.id_rocks_pub = self.create_publisher(Image, "/id_rocks", 1)

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
            self.MAX_DISPARITY * (imgL.shape[1] / self.IMG_WIDTH)
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
            2 * np.tan(np.radians(self.FIELD_OF_VIEW / 2))
        )

        disparity = np.where(disparity > 1e-6, disparity, 0)
        depth_map = np.where(  # checks that image color is not pitch black at pixel location as well
            (disparity > 1e-6)
            & (imgL > 0)
            & (
                cols
                > (self.DISPARITY_OCCLUSION_SLOPE * rows)
                + (
                    self.DISPARITY_OCCLUSION_YINT
                    * (disparity.shape[0] / self.IMG_HEIGHT)
                )
            ),
            (focal_length_x * self.BASELINE) / disparity,
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
        origin_pos: np.array = np.array([0, 0, 0]),
        origin_rpy=[0, 0, 0],
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
            2 * np.tan(np.radians(self.FIELD_OF_VIEW / 2))
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

        X = depth_map
        Y = (x_indices - principal_point[0]) * depth_map / focal_length_x
        Z = (y_indices - principal_point[1]) * depth_map / focal_length_y
        points = torch.stack([X, -Y, -Z], dim=1)
        
        points = points.numpy()

        if not (origin_rpy == np.array([0, 0, 0])).all():
            roll, pitch, yaw = origin_rpy
            pitch = -pitch

            # Rotation matrices for roll, pitch, and yaw

            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)],
                ]
            )
            R_y = np.array(
                [
                    [np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)],
                ]
            )
            R_z = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )

            # Combined rotation matrix
            R = R_z @ R_y @ R_x
            R.astype(np.float64)
            
            print(points.T.shape)
            print(R.shape)
            
            points = R @ points.T
            points = points.T

        points = points + origin_pos

        colors = imgL[mask]

        return points, colors.numpy()
    
    def compute_point_cloud_from_depth(
        self,
        imgL: np.ndarray,
        depth_map: np.ndarray,
        origin_pos,
        origin_rpy,
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
            np.array(origin_pos),
            origin_rpy,
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
        msg.header.frame_id = (
            "map"  # TODO: swap out for actual frame id - prob odom evenutally
        )

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
                name="intensity", offset=12, datatype=PointField.UINT32, count=1
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
        print("IMAGE RECEIVED")

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
        # should_detect = (np.linalg.norm(pos - self.last_pos) >= 0.3) or (np.abs(rpy[2] - self.last_rpy[2]) >= np.deg2rad(5)) or (len(self.acc_rock_pc.points) == 0)

        if imgL is not None:
            if imgL.shape is not (self.INITIAL_WIDTH, self.INITIAL_HEIGHT):
                imgL = cv2.resize(imgL, (self.INITIAL_WIDTH, self.INITIAL_HEIGHT))
                imgR = cv2.resize(imgR, (self.INITIAL_WIDTH, self.INITIAL_HEIGHT))

            depth = self.compute_depth_image(imgL, imgR)

            # sharpen left and right images to increase accuracy
            imgL = cv2.filter2D(imgL, -1, self.sharpen_kernel)
            imgR = cv2.filter2D(imgL, -1, self.sharpen_kernel)

            imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
            imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)

            contours = self.predict_rocks(imgL)
            rock_depth, terrain_depth = self.get_segmented_depth_maps(contours, depth)

            # convert back to grayscale for pc generation
            imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
            rock_pc = self.compute_point_cloud_from_depth(
                imgL, rock_depth, origin_pos=pos, origin_rpy=rpy, voxel_size=0.01
            )

            rock_pc.colors = o3d.utility.Vector3dVector(
                [[1.0, 0, 0]] * len(rock_pc.points)
            )
            # rock_pc, _ = rock_pc.remove_radius_outlier(nb_points=2, radius=0.01)
            rock_pc, _ = rock_pc.remove_statistical_outlier(
                nb_neighbors=15, std_ratio=1e-8
            )
            
            # accumulate if rocks should be recorded
            # if should_detect:
            self.acc_rock_pc = self.add_point_clouds(rock_pc, self.acc_rock_pc)

            self.rock_pc_pub.publish(
                self.o3d_pc_to_point_cloud2(
                    self.get_clock().now().to_msg(), self.acc_rock_pc
                )
            )

            # terrain pc generation ---------
            pc = self.compute_point_cloud_from_depth(
                imgL, terrain_depth, origin_pos=pos, origin_rpy=rpy, voxel_size=0.05
            )
            pc, _ = pc.remove_radius_outlier(nb_points=20, radius=0.5)

            self.pc_pub.publish(
                self.o3d_pc_to_point_cloud2(
                    self.get_clock().now().to_msg(), pc
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
        Predicts the segmented contours from the left image.

        Parameters:
            left (cv2 img): left image
        Returns:
            contours (list-like): list of sets of contour points
        """

        contours = self.model(left)[0].boxes.xyxy
        contours = [np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2) for contour in contours if len(contour) > 0]
        return contours

    def get_segmented_depth_maps(self, contours, depth):
        """
        Segments depth map with only rocks and depth map with no rocks.

        Args:
            contours (list): list of {(x1, y1), (x2, y2), (x3, y3), ...} contour point coords
            depth (np.ndarray): depth image to segment
        Returns:
            rock_map (np.ndarray): depth map with only rocks
        """

        # Ensure maskL has the correct shape and type
        self.maskL = np.zeros(depth.shape[:2], dtype=np.uint8)

        # Ensure contours are properly formatted (N, 1, 2) for OpenCV
        # contours = [np.array(contour, dtype=np.int32).reshape(-1, 1, 2) for contour in contours if len(contour) > 0]

        # Draw filled contours
        self.maskL = cv2.drawContours(self.maskL, contours, -1, (255), thickness=cv2.FILLED)

        # Apply bitwise operation
        rock_map = cv2.bitwise_and(depth, depth, mask=self.maskL)

        # find rest of terrain
        anti_rock_mask = cv2.bitwise_not(self.maskL)
        terrain_map = cv2.bitwise_and(depth, depth, mask=anti_rock_mask)

        return rock_map, terrain_map


def main():
    rclpy.init()
    node = RockDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
