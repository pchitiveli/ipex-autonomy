import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import time

# from multiprocessing import Pool, Manager
import ultralytics
import cProfile
import torch


# ==========================================================================================================
# ROCK DETECTION METHODS
# ==========================================================================================================
class RockDetection:
    def __init__(self):
        self.rockSpecifications = []
        # self.model = ultralytics.YOLO("perception/models/model.pt")
        self.model = ultralytics.YOLO("models/model.pt")

    def __checkDepth(self, pos1, pos2):
        """
        Calculate the depth of an object in meters given the pixel locations of two points in the image.
        Args:
            pos1 (tuple): pixel location of point 1 in the image
            pos2 (tuple): pixel location of point 2 in the image
        Returns:
            z (float): depth of object in meters
        """
        d = pos1[0] - pos2[0]
        z = PointCloudUtil.FOCAL_LENGTH_X * PointCloudUtil.BASELINE / d
        return z

    def __getAngleOffset(self, pos1, pos2, z=None):
        """
        Calculate the angle offset of an object in the image.
        Args:
            pos1 (tuple): pixel location of point 1 in the image
            pos2 (tuple): pixel location of point 2 in the image
            z (float): depth of object in meters
        Returns:
            theta (float): angle offset of object in radians
        """
        if z is None:
            z = self.__checkDepth(pos1, pos2)
        x_1 = (
            (pos1[0] - PointCloudUtil.PRINCIPAL_POINT[0]) * z
        ) / PointCloudUtil.FOCAL_LENGTH_X
        theta = np.arctan(x_1 / z)
        return theta

    def __getCylindricalDiameter(self, boundingBoxWidth, z):
        """
        Calculate the diameter of an object in the image.
        Args:
            boundingBoxWidth (int): width of the bounding box of the object in pixels
            z (float): depth of object in meters
        Returns:
            diameter (float): diameter of object in meters
        """
        return boundingBoxWidth * z / PointCloudUtil.FOCAL_LENGTH_X

    def __getCylindicalHeight(self, boundingBoxHeight, z):
        """
        Calculate the height of an object in the image.
        Args:
            boundingBoxHeight (int): height of the bounding box of the object in pixels
            z (float): depth of object in meters
        Returns:
            height (float): height of object in meters
        """
        return boundingBoxHeight * z / PointCloudUtil.FOCAL_LENGTH_Y

    def within_rock(self, x, y, z):
        """
        Check if a point is within the bounds of a rock.
        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point
            z (float): z-coordinate of the point
        Returns:
            bool: whether the point is within the bounds of a rock
        """
        for i in range(len(self.rockSpecifications)):
            radius = self.rockSpecifications[i][2] / 2
            y_center = self.rockSpecifications[i][0] * np.tan(
                self.rockSpecifications[i][1]
            )
            distance = np.sqrt(
                (x - self.rockSpecifications[i][0]) ** 2 + (-y - y_center) ** 2
            )
            if distance > radius:
                continue
            return i
        return -1

    def __predictStereoImage(self, left, right, visualize=False):
        """
        Predicts the objects in a stereo image pair. Groups them up into pairs based on their location in the image.
        Parameters:
            leftPath (cv2 img): left image
            rightPath (cv2 img): right image
            visualize (bool): whether to visualize the results
        Returns:
            pairs (list): list of pairs of objects in the stereo image pair
        """
        # left = cv2.imread(leftPath)
        # right = cv2.imread(rightPath)

        leftResults = self.model(left)[0]
        rightResults = self.model(right)[0]

        leftBoxes = leftResults.boxes.xyxy.tolist()
        rightBoxes = rightResults.boxes.xyxy.tolist()

        leftImg = left
        rightImg = right

        pairs = []

        for i, leftBox in enumerate(leftBoxes):
            leftCenter = (
                leftBox[0] / 2 + leftBox[2] / 2,
                leftBox[1] / 2 + leftBox[3] / 2,
            )
            leftWidth = leftBox[2] - leftBox[0]
            leftHeight = leftBox[3] - leftBox[1]
            leftArea = leftWidth * leftHeight
            for j, rightBox in enumerate(rightBoxes):
                rightCenter = (
                    rightBox[0] / 2 + rightBox[2] / 2,
                    rightBox[1] / 2 + rightBox[3] / 2,
                )
                # within 150 pixels of each other (x - coordinate)
                if abs(rightCenter[0] - leftCenter[0]) > 150:
                    continue
                # within 10 pixels of each other (y - coordinate)
                # this is less since the rocks should not change their height pos
                if abs(rightCenter[1] - leftCenter[1]) > 10:
                    continue
                rightWidth = rightBox[2] - rightBox[0]
                rightHeight = rightBox[3] - rightBox[1]
                rightArea = rightWidth * rightHeight
                # if the area is different by more than 20%, then they are not a match
                if (leftArea - rightArea) / rightArea > 0.2:
                    continue
                pairs.append(
                    (
                        [leftCenter, leftWidth, leftHeight],
                        [rightCenter, rightWidth, rightHeight],
                    )
                )
                rightBoxes.pop(j)
                break

        if visualize:
            for i in range(len(pairs)):
                leftCenter, leftWidth, leftHeight = pairs[i][0]
                rightCenter, rightWidth, rightHeight = pairs[i][1]
                leftImg = cv2.rectangle(
                    leftImg,
                    (
                        int(leftCenter[0] - leftWidth / 2),
                        int(leftCenter[1] - leftHeight / 2),
                    ),
                    (
                        int(leftCenter[0] + leftWidth / 2),
                        int(leftCenter[1] + leftHeight / 2),
                    ),
                    (0, 255, 0),
                    2,
                )
                rightImg = cv2.rectangle(
                    rightImg,
                    (
                        int(rightCenter[0] - rightWidth / 2),
                        int(rightCenter[1] - rightHeight / 2),
                    ),
                    (
                        int(rightCenter[0] + rightWidth / 2),
                        int(rightCenter[1] + rightHeight / 2),
                    ),
                    (0, 255, 0),
                    2,
                )
                leftImg = cv2.putText(
                    leftImg,
                    str(i),
                    (int(leftCenter[0]), int(leftCenter[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                rightImg = cv2.putText(
                    rightImg,
                    str(i),
                    (int(rightCenter[0]), int(rightCenter[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imwrite("left.jpg", leftImg)
            cv2.imwrite("right.jpg", rightImg)

        return pairs

    def __getRockSpecifications(self, leftPos, leftWidth, leftHeight, rightPos):
        """
        Get the specifications of a rock given the bounding boxes of the rock in the stereo image pair.
        The specifications are: (depth, angleOffset, diameter, height)
        Args:
            leftPos (tuple): pixel location of the center of the left bounding box
            leftWidth (int): width of the left bounding box
            leftHeight (int): height of the left bounding box
            rightPos (tuple): pixel location of the center the right bounding box
        Returns:
            specifications (tuple): specifications of the rock
        """
        depth = self.__checkDepth(leftPos, rightPos)
        angleOffset = self.__getAngleOffset(leftPos, rightPos, depth)
        diameter = self.__getCylindricalDiameter(leftWidth, depth)
        height = self.__getCylindicalHeight(leftHeight, depth)
        return (depth.item(), angleOffset.item(), diameter.item(), height.item())

    def predictStereoImages(self, left, right, resetRocks=True):
        """
        Bundles everything together in one method
        Args:
            left (cv2 image): The left image
            right (cv2 image): The right image
        """
        if resetRocks:
            self.rockSpecifications = []
        pairs = self.__predictStereoImage(left, right)
        for pair in pairs:
            leftPos, leftWidth, leftHeight = pair[0]
            rightPos, rightWidth, rightHeight = pair[1]
            self.rockSpecifications.append(
                self.__getRockSpecifications(leftPos, leftWidth, leftHeight, rightPos)
            )


class PointCloudUtil:
    # CONSTANTS ------------------------------------------------------------------------------------------------

    # define constants
    INITIAL_WIDTH = 1280
    INITIAL_HEIGHT = 720
    IMG_WIDTH = 480
    IMG_HEIGHT = 270
    PRINCIPAL_POINT = (IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0)
    MAX_DISPARITY = 304 * (
        float(IMG_WIDTH) / INITIAL_WIDTH
    )  # actually 315, but num_disp has to be divisible by 16
    MAX_DISPARITY = int(MAX_DISPARITY - (MAX_DISPARITY % 16))
    FIELD_OF_VIEW = 70  # degrees
    FOCAL_LENGTH_X = IMG_WIDTH / (
        2 * np.tan(np.radians(FIELD_OF_VIEW / 2))
    )  # calculate focal length
    FOCAL_LENGTH_Y = IMG_HEIGHT / (
        2 * np.tan(np.radians(39.375 / 2))
    )  # calculate focal length
    BASELINE = 0.162

    MIN_SCANNING_DIST = 0.5  # meters
    MAX_SCANNING_DIST = 5  # meters

    NUM_PROCESSES = (
        1  # number of simultaneously running processes for point cloud generation
    )
    SLICE_WIDTH = (
        IMG_WIDTH - MAX_DISPARITY
    ) / NUM_PROCESSES  # calculation of the slice width

    # local map constants
    CELL_SIZE = 0.15
    HALF_CELL_SIZE = CELL_SIZE / 2
    RANSAC_THRESHOLD_POINTS = 10  # points required to calculate the RANSAC ground plane
    MAP_SIZE = 27
    CELL_COUNT = int(MAP_SIZE / CELL_SIZE)

    DISPARITY_OCCLUSION_SLOPE = 0.6639344262
    DISPARITY_OCCLUSION_YINT = -71.6066

    # ==========================================================================================================
    # BLOCK MATCHING METHODS
    # ==========================================================================================================

    @staticmethod
    def calculate_disparity_sgbm(imgL, imgR) -> np.array:
        """
        Calculates disparity map using semi-global block matching method.

        Args:
            imgL (np.ndarray): grayscale left image
            imgR (np.ndarray): grayscale right image
        Returns:
            disparity: numpy array of pixel disparities for each pixel in imgL
        """

        max_disp = int(
            PointCloudUtil.MAX_DISPARITY * (imgL.shape[1] / PointCloudUtil.IMG_WIDTH)
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

    @staticmethod
    def calculate_disparity_bm(imgL, imgR) -> np.array:
        """
        Calculates disparity map using block matching method. Does not produce as clean of a disparity map as SGBM.

        Args:
            imgL (np.ndarray): grayscale left image
            imgR (np.ndarray): grayscale right image
        Returns:
            disparity: numpy array of pixel disparities for each pixel in imgL
        """

        # BM Parameters
        block_size = 5

        stereo_bm = cv2.StereoBM_create(
            # max disp - min disp. max disp is the max possible disparity value (for objects close up) - must be divisible by 16 in current opencv implementation for some reason
            numDisparities=PointCloudUtil.MAX_DISPARITY,
            # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 5..21 range.
            blockSize=block_size,
        )

        # divide by 16 to get actual disparity values
        disparity = stereo_bm.compute(imgL, imgR) / 16.0
        # disparity = disparity[:, MAX_DISPARITY:] # crop out the leftmost part of the disparity map
        return disparity

    # ==========================================================================================================
    # POINT CLOUD GENERATION METHODS
    # ==========================================================================================================

    @staticmethod
    def disparity_to_depth_map(disparity: np.array, imgL: np.ndarray) -> np.array:
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
            2 * np.tan(np.radians(PointCloudUtil.FIELD_OF_VIEW / 2))
        )

        disparity = np.where(disparity > 1e-6, disparity, 0)
        depth_map = np.where(  # checks that image color is not pitch black at pixel location as well
            (disparity > 1e-6)
            & (imgL > 0)
            & (
                cols
                > (PointCloudUtil.DISPARITY_OCCLUSION_SLOPE * rows)
                + (
                    PointCloudUtil.DISPARITY_OCCLUSION_YINT
                    * (disparity.shape[0] / PointCloudUtil.IMG_HEIGHT)
                )
            ),
            (focal_length_x * PointCloudUtil.BASELINE) / disparity,
            0,
        )
        # filter out depth values greater than 20m
        # depth_map = np.where(depth_map > 20, float("inf"), depth_map)

        # KEEP FOLLOWING BLOCK FOR DEBUGGING PURPOSES!!!

        # plt.title("Disparity")
        # plt.imshow(disparity, cmap="viridis")
        # plt.figure()
        # plt.title("Depth")
        # plt.imshow(depth_map, cmap="viridis")
        # plt.show()

        return depth_map
    
    @staticmethod
    def compute_depth_image(
        imgL: np.ndarray,
        imgR: np.ndarray,
        disparity_method: str = "sgbm",
    ):
        # calculate the disparities between pixels first
        disparity = None
        if disparity_method == "sgbm":
            disparity = PointCloudUtil.calculate_disparity_sgbm(imgL, imgR)
        elif disparity_method == "bm":
            disparity = PointCloudUtil.calculate_disparity_bm(imgL, imgR)

        # convert disparity map to a depth map for more useful info
        depth_map = PointCloudUtil.disparity_to_depth_map(disparity, imgL)

        return depth_map

    @staticmethod
    def compute_point_cloud_from_depth(
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

        points, colors = PointCloudUtil.depth_map_to_points(
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

    @staticmethod
    def depth_map_to_points(
        depth_map: np.array,
        imgL: np.ndarray,
        origin_pos: np.array = np.array([0, 0, 0]),
        origin_rpy=[0, 0, 0],
    ) -> o3d.geometry.PointCloud:
        """
        Converts depth map numpy array into points and colors np arrays.

        Args:
            depth_map (np.array): depth numpy array
            imgL (np.array): left image in RGB
            rockDetector (RockDetection): RockDetection object
            index (int): index of depth slice if multiprocessing
            origin_pos (np.array): position of the camera in [x, y, z]
            origin_rpy (list-like): origin orientation of the camera in [roll, pitch, yaw]
        Return:
            points, colors - list-like, represent matching points and colors
        """

        focal_length_x = imgL.shape[1] / (
            2 * np.tan(np.radians(PointCloudUtil.FIELD_OF_VIEW / 2))
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

    @staticmethod
    def depth_map_point_to_3d(x_index: int, y_index: int, depth: float) -> np.array:
        """
        Converts a pixel on depth map to 3D point.

        Args:
            x_index (int): x pixel location
            y_index (int): y pixel location
            depth (float): depth at pixel location
        """

        X = depth
        Y = (
            -(x_index - PointCloudUtil.PRINCIPAL_POINT[0])
            * depth
            / PointCloudUtil.FOCAL_LENGTH_X
        )
        Z = (
            -(y_index - PointCloudUtil.PRINCIPAL_POINT[1])
            * depth
            / PointCloudUtil.FOCAL_LENGTH_Y
        )

        return np.array([X, Y, Z])

    @staticmethod
    def rotate_point(point, orientation) -> np.array:
        """
        Rotates a 3D point around the origin by the given orientation.

        Args:
            point (list-like): 3D point to be rotated [x, y, z]
            orientation (list-like): 3D orientation in [roll, pitch, yaw]
        Return:
            position of rotated point {x', y', z'}
        """

        point = np.array(point)

        if (orientation == np.array([0, 0, 0])).all():
            return point

        roll, pitch, yaw = orientation
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
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # Combined rotation matrix
        R = R_z @ R_y @ R_x

        # Rotate the point
        rotated_point = R @ point

        return rotated_point

    @staticmethod
    def add_point_clouds(
        pc1: o3d.geometry.PointCloud, pc2: o3d.geometry.PointCloud
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

    # ==========================================================================================================
    # ROS CONVERSIONS
    # ==========================================================================================================

    @staticmethod
    def add_point_to_point_cloud(
        pc1: o3d.geometry.PointCloud, point, color
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

        points = np.concatenate((pc1_points, np.array([point])), axis=0)
        colors = np.concatenate((pc1_colors, np.array([color])), axis=0)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        return pc

    @staticmethod
    def points_to_o3d_pc(points, colors):
        """
        Converts points and matching colors to an Open3D point cloud.

        Args:
            points (list-like): points in 3d space
            colors (list-like): colors in RGB normalized to 0-1 in same order as points
        """

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        return pc

    # @staticmethod
    # def o3d_pc_to_point_cloud2(stamp_msg, o3d_pc) -> PointCloud2:
    #     points = np.asarray(o3d_pc.points, dtype=np.float32)
    #     colors = np.asarray(o3d_pc.colors, dtype=np.float32)

    #     msg = PointCloud2()
    #     # msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.stamp = stamp_msg
    #     msg.header.frame_id = (
    #         "map"  # TODO: swap out for actual frame id - prob odom evenutally
    #     )

    #     msg.height = 1
    #     msg.width = points.shape[0]
    #     msg.is_dense = True  # no invalid values
    #     msg.is_bigendian = False
    #     msg.point_step = 16  # 16 bytes per point
    #     msg.row_step = msg.point_step * points.shape[0]  # byte length of row

    #     msg.fields = [
    #         PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(
    #             name="intensity", offset=12, datatype=PointField.UINT32, count=1
    #         ),
    #     ]

    #     # convert colors from 0-1 RGB to 0-1 intensity
    #     colors = colors * 255
    #     intensity = np.dot(colors, [0.2989, 0.5870, 0.1140]).astype(np.float32)

    #     # make numpy structured array of points
    #     structured_data = np.zeros(
    #         points.shape[0],
    #         dtype=[
    #             ("x", np.float32),
    #             ("y", np.float32),
    #             ("z", np.float32),
    #             ("intensity", np.uint32),
    #         ],
    #     )
    #     structured_data["x"] = points[:, 0]  # populate x-coords
    #     structured_data["y"] = points[:, 1]  # populate y coords
    #     structured_data["z"] = points[:, 2]  # populate z coords

    #     # populate intensity vals with 32-bit ints
    #     structured_data["intensity"] = intensity.flatten()

    #     msg.data = structured_data.tobytes()  # reads byte data when processing pc
    #     return msg

    # @staticmethod
    # def ros_pc2_to_o3d_pc(ros_pc: PointCloud2):
    #     try:
    #         gen = point_cloud2.read_points(
    #             ros_pc, field_names=("x", "y", "z"), skip_nans=True
    #         )
    #         points = list(gen)
    #         pc_points = [list(point) for point in points]
    #         pc_points = np.array(pc_points)

    #         if pc_points.shape[0] == 0:
    #             return None

    #         pc = o3d.geometry.PointCloud()
    #         pc.points = o3d.utility.Vector3dVector(pc_points)
    #         # print(pc)

    #         return pc
    #     except Exception as e:
    #         print(f"Conversion error: {e}")
    #         return None

    # ==========================================================================================================
    # RANSAC METHODS
    # ==========================================================================================================

    @staticmethod
    def calculate_plane_coefficients(point1, point2, point3):
        """
        Calculates plane from 3 points in 3D space.

        Args:
            point1 (list-like): point 1 (x, y, z)
            point2 (list-like): point 2 (x, y, z)
            point3 (list-like): point 3 (x, y, z)
        Return:
            normal_vector (np.array): normal vector of plane (x, y, z)
            plane_coefficients (np.array): coefficients of plane (A, B, C, D)
        """

        # Create vectors between the points
        vector1 = (np.array(point1) - np.array(point2)).astype(float)
        vector2 = (np.array(point3) - np.array(point2)).astype(float)
        # Calculate the normal vector to the plane using cross product
        normal_vector = np.cross(vector1, vector2)
        # Normalize the normal vector
        normal_magnitude = np.linalg.norm(normal_vector)
        if normal_magnitude == 0.0:
            return None, None  # no solution can be found
        else:
            normal_vector /= np.linalg.norm(normal_vector)

        # Calculate the constant term D in the plane equation Ax + By + Cz + D = 0
        D = -np.dot(normal_vector, point1)

        # Coefficients [A, B, C, D] of the plane equation
        plane_coefficients = np.concatenate((normal_vector, [D]))

        return normal_vector, plane_coefficients

    @staticmethod
    # @jit
    def fit_plane_ransac(
        data: np.ndarray, threshold: float = 0.01, iterations: int = 100
    ):
        """
        Runs RANSAC regression method for finding best candidate plane for point data.

        Args:
            data (np.ndarray): array of points in point cloud (x, y, z)
            threshold (float): maximum distance of point from candidate plane to be considered an inlier. Default is 0.01 meters
            iterations (int): maximum number of iterations. Default is 100
        Return:
            best_plane (list): plane coefficients for the best candidate plane in [A, B, C, D]
            best_inliers (list): inlier points for best plane in (x, y, z)
        """

        i = 0
        bestSupport = 0
        best_plane = np.zeros(4, dtype=np.float64)
        best_inliers = np.empty((0, 3), dtype=np.float64)
        BestStDeviation = 10e10
        for i in range(iterations):
            # print(f"Iteration {i} of RANSAC")  # <-- DEBUGGING

            # Randomly sample three points
            p1 = np.ascontiguousarray(np.array([0, 0]))
            p2 = np.ascontiguousarray(np.array([0, 0]))
            p3 = np.ascontiguousarray(np.array([0, 0]))

            normal_magnitude = 0.0
            vector1_magnitude = 0.0
            vector2_magnitude = np.array([0.0])

            sample_indices = np.random.choice(len(data), 3, replace=False)
            p1, p2, p3 = data[sample_indices]

            v1 = p1 - p2
            vector1_magnitude = np.linalg.norm(v1)
            v2 = p3 - p2
            vector2_magnitude = np.linalg.norm(v2)
            normal_vector = np.ascontiguousarray(np.cross(v1, v2).astype(np.float64))
            normal_magnitude = np.linalg.norm(normal_vector)

            bad_sample = (
                np.array_equal(p1, p2)
                or np.array_equal(p2, p3)
                or np.array_equal(p1, p3)
                or normal_magnitude < 1e-6
                or vector1_magnitude < 1e-6
                or vector2_magnitude < 1e-6
            )

            if bad_sample:
                continue

            # print("Samples found")

            # Calculate the constant term D in the plane equation Ax + By + Cz + D = 0
            D = -np.dot(normal_vector, p1)
            D = float(D)

            # Coefficients [A, B, C, D] of the plane equation
            plane_coefficients = np.empty(4, dtype=np.float64)
            plane_coefficients[:3] = normal_vector
            plane_coefficients[3] = np.float64(D)

            # return normal_vector, plane_coefficients

            # Fit a plane to the sampled points
            # normal_vector, plane_coefficients = PointCloudUtil.calculate_plane_coefficients(
            #     p1, p2, p3
            # )

            d = -np.dot(normal_vector, p1)
            # Currplane = np.concatenate((normalVector, [d]))

            # Calculate distances from all points to the plane
            distances = np.abs(np.dot(data, normal_vector) + d) / np.linalg.norm(
                normal_vector
            )
            # Count inliers (points within the threshold distance from the plane)
            inliers = data[distances < threshold]
            std = np.std(inliers)
            # Update best plane if the current one has more inliers
            if (len(inliers) > bestSupport) or (
                (len(inliers) >= bestSupport) and (std < BestStDeviation)
            ):
                bestSupport = len(inliers)
                best_plane = plane_coefficients
                best_inliers = inliers
                BestStDeviation = std
                # print(  # <-- DEBUGGING
                #     f"New best plane found with {bestSupport} inliers and standard deviation {BestStDeviation}"
                # )

        return best_plane, best_inliers

    # visualization method
    @staticmethod
    def create_visualization_plane_mesh(
        plane_coefficients, xlim=(-1, 1), ylim=(-1, 1)
    ) -> o3d.geometry.TriangleMesh:
        """
        Creates a visualization mesh for a plane.

        Args:
            plane_coefficients (list-like): coefficients of plane (A, B, C, D)
            xlim (list-like): limits on x axis (min_x, max_x)
            ylim (list-like): limits on y axis (min_y, max_y)
        Return:
            plane mesh (open3d.geometry.TriangleMesh)
        """

        a = plane_coefficients[0]
        b = plane_coefficients[1]
        c = plane_coefficients[2]
        d = plane_coefficients[3]

        # Generate a grid for the plane
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
        )
        zz = (-d - a * xx - b * yy) / c  # Solve for z

        # Create vertices
        vertices = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

        # Create triangles for mesh
        rows, cols = xx.shape
        triangles = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # store indexes of vertices in flattened plane points for triangle mesh creation
                idx1 = i * cols + j
                idx2 = i * cols + j + 1
                idx3 = (i + 1) * cols + j
                idx4 = (i + 1) * cols + j + 1
                triangles.append([idx1, idx2, idx3])
                triangles.append([idx3, idx2, idx4])

        # Create Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0, 1, 0])  # Green color for the plane

        return mesh


class MapBuilder:
    def __init__(self):
        self.__pc_map = np.array(  # initialize map of point clouds
            [[o3d.geometry.PointCloud()] * PointCloudUtil.CELL_COUNT]
            * PointCloudUtil.CELL_COUNT
        )
        self.__terrain_map = (
            np.array(  # initialize map of ground plane, height, and rock flag
                [[[None, 0.0, False]] * PointCloudUtil.CELL_COUNT]
                * PointCloudUtil.CELL_COUNT
            )
        )  # map[0] = ground plane coeffs, map[1] = height, map[2] = rock_flag

    # ==========================================================================================================
    # CELL MANIPULATION METHODS
    # ==========================================================================================================

    def get_cell_count(self):
        """
        Returns number of cells in square map.
        """
        return PointCloudUtil.CELL_COUNT

    def set_cell_point_cloud(
        self, x_index: float, y_index: float, pc: o3d.geometry.PointCloud
    ):
        """
        Sets cell's point cloud.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            pc (open3d.geometry.PointCloud): point cloud of environment in the cell
        """
        self.__pc_map[x_index, y_index] = pc

    def get_cell_point_cloud(self, x_index, y_index) -> o3d.geometry.PointCloud:
        """
        Gets cell's point cloud.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
        """
        return self.__pc_map[x_index, y_index]

    def add_point_cloud_to_cell(
        self, x_index: int, y_index: int, pc: o3d.geometry.PointCloud
    ):
        """
        Adds points to a cell's point cloud in the point cloud map.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            pc (open3d.geometry.PointCloud): point cloud to append to cell
        """
        self.set_cell_point_cloud(
            x_index,
            y_index,
            PointCloudUtil.add_point_clouds(
                self.get_cell_point_cloud(x_index, y_index), pc
            ),
        )

    def add_point_to_cell(self, x_index: int, y_index: int, point, color):
        """
        Adds point to a cell's point cloud in the point cloud map.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            point (array-like): [x, y, z]
            color (array-like): [r, g, b] normalized between 0 and 1
        """
        self.set_cell_point_cloud(
            x_index,
            y_index,
            PointCloudUtil.add_point_to_point_cloud(
                self.get_cell_point_cloud(x_index, y_index), point, color
            ),
        )

    def add_points_to_cell(self, x_index: int, y_index: int, points, colors):
        """
        Adds point to a cell's point cloud in the point cloud map.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            point (array-like): [x, y, z]
            color (array-like): [r, g, b] normalized between 0 and 1
        """
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        self.set_cell_point_cloud(
            x_index,
            y_index,
            PointCloudUtil.add_point_clouds(
                self.get_cell_point_cloud(x_index, y_index), pc
            ),
        )

    def set_ground_plane(self, x_index: int, y_index: int, plane):
        """
        Sets ground plane coeffs for cell.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            height (array-like): ground plane coefficients
        """
        self.__terrain_map[x_index, y_index][0] = plane

    def get_ground_plane(self, x_index: int, y_index: int):
        """
        Sets ground plane coeffs for cell.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
        """
        return self.__terrain_map[x_index, y_index][0]

    def set_height(self, x_index: int, y_index: int, height: float):
        """
        Sets height for cell.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            height (float): ground plane height of cell
        """
        self.__terrain_map[x_index, y_index][1] = height

    def get_height(self, x_index: int, y_index: int):
        """
        Gets height for cell.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
        """
        return self.__terrain_map[x_index, y_index][1]

    def set_rock_flag(self, x_index: int, y_index: int, rock_flag: bool):
        """
        Sets rock flag for mock map.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            rock_flag (bool): whether a rock exists in the cell
        """
        self.__terrain_map[x_index, y_index][2] = rock_flag

    def get_rock_flag(self, x_index: int, y_index: int):
        """
        Gets rock flag for mock map.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
        """
        return self.__terrain_map[x_index, y_index][2]

    @staticmethod
    def get_cell_indices(x_coord, y_coord):
        """
        Get cell indices in map.

        Args:
            x_coord (float): x coord in 3D space
            y_coord (float): y coord in 3D space
        """
        x_ind, y_ind = (
            int(x_coord / PointCloudUtil.CELL_SIZE),
            int(y_coord / PointCloudUtil.CELL_SIZE),
        )
        if (
            x_ind < 0
            or y_ind < 0
            or x_ind >= PointCloudUtil.CELL_COUNT
            or y_ind >= PointCloudUtil.CELL_COUNT
        ):
            return None, None
        else:
            return x_ind, y_ind

    # ==========================================================================================================
    # GROUND PLANE ESTIMATION
    # ==========================================================================================================

    @staticmethod
    def calculate_ground_plane_and_height(x_index, y_index, cell_points):
        """
        Calculates ground plane coefficients and median height.

        Args:
            x_index (int): x index of cell
            y_index (int): y index of cell
            cell_points (array-like): list of points in the cell
        Returns:
            [x index (float), y index (float), best plane coeffs (array-like), ground height (float)]
        """
        # regress best fit plane if there are more than 3 points in cell
        best_plane, _ = PointCloudUtil.fit_plane_ransac(
            np.asarray(cell_points), threshold=0.01, iterations=50
        )

        # calculate ground plane height at center of cell
        ground_height = (
            best_plane[3]
            - (
                best_plane[0]
                * ((x_index * PointCloudUtil.CELL_SIZE) + PointCloudUtil.HALF_CELL_SIZE)
            )
            - (
                best_plane[1]
                * ((y_index * PointCloudUtil.CELL_SIZE) + PointCloudUtil.HALF_CELL_SIZE)
            )
        )
        if best_plane[2] != 0:
            ground_height /= best_plane[2]

        # print(f"Best plane for ({x_index}, {y_index}) cell found")

        return [x_index, y_index, best_plane, ground_height]

    # def depth_map_to_point_cloud(
    #     self,
    #     depth_map: np.array,
    #     imgL: np.ndarray,
    #     rockSpecifications,
    #     origin_pos: np.array = np.array([0, 0, 0]),
    #     origin_rpy=[0, 0, 0],
    # ) -> o3d.geometry.PointCloud:
    #     """
    #     Converts depth map numpy array into an open3d point cloud.

    #     Args:
    #         depth_map (np.array): depth numpy array
    #         imgL (np.array): left image in RGB
    #         rockDetector (RockDetection): RockDetection object
    #         index (int): index of depth slice if multiprocessing
    #         origin_pos (np.array): position of the camera in [x, y, z]
    #         origin_rpy (list-like): origin orientation of the camera in [roll, pitch, yaw]
    #     Return:
    #         pc (open3d.geometry.PointCloud): point cloud represented by depth image
    #     """

    #     pc = o3d.geometry.PointCloud()
    #     points = []
    #     colors = []

    #     # loop through each row in the depth map
    #     for y in range(depth_map.shape[0]):
    #         # loop through each column in the depth map
    #         for x in range(depth_map.shape[1]):
    #             z = depth_map[y, x]  # find depth at the current pixel
    #             if z != float("inf"):  # if the depth is measurable, convert to 3D space
    #                 point = PointCloudUtil.depth_map_point_to_3d(
    #                     x_index=x,
    #                     # add the index to the x value to account for the slice width; otherwise the point clouds will overlap in the same space
    #                     y_index=y,
    #                     depth=z,
    #                 )

    #                 color = imgL[y][x] / 255.0

    #                 if rockSpecifications is not None:
    #                     inRock = -1
    #                     for i in range(len(rockSpecifications)):
    #                         radius = rockSpecifications[i][2] / 2
    #                         y_center = rockSpecifications[i][0] * np.tan(
    #                             rockSpecifications[i][1]
    #                         )
    #                         distance = np.sqrt(
    #                             (point[0] - rockSpecifications[i][0]) ** 2
    #                             + (-point[1] - y_center) ** 2
    #                         )
    #                         if distance > radius:
    #                             continue
    #                         inRock = i

    #                     if inRock != -1:
    #                         color = np.array([1, 0, 0])

    #                 # rotate based on given robot orientation
    #                 point = PointCloudUtil.rotate_point(point, origin_rpy)
    #                 # add the robot's position to the point so that the point cloud is oriented and positioned correctly
    #                 point = point + origin_pos

    #                 (x_index, y_index) = MapBuilder.get_cell_indices(point[0], point[1])

    #                 if x_index is not None and y_index is not None:
    #                     points.append(point)
    #                     colors.append(color)
    #                     # self.add_point_to_cell(x_index, y_index, point, color)

    #     pc.points = o3d.utility.Vector3dVector(points)
    #     pc.colors = o3d.utility.Vector3dVector(colors)
    #     return pc

    def compute_sorted_point_cloud(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        rockDetector,
        origin_pos,
        origin_rpy,
        disparity_method: str = "sgbm",
    ):
        """
        Compute point cloud from stereo image input using Semi-Global Block Matching disparity calculation method.

        Args:
            imgL: grayscale left image
            imgR: grayscale right image
            origin_pos (list-like): camera position (x, y, z)
            origin_rpy (list-like): camera orientation (roll, pitch, yaw).
            rockDetector (RockDetection): RockDetection object
            disparity_method (str): allows for distinction between block matching and semi-global block matching
        Return:
            pc (open3d.geometry.PointCloud): point cloud of stereo image data
        """

        start = time.perf_counter()
        # calculate the disparities between pixels first
        disparity = None
        if disparity_method == "sgbm":
            disparity = PointCloudUtil.calculate_disparity_sgbm(imgL, imgR)
        elif disparity_method == "bm":
            disparity = PointCloudUtil.calculate_disparity_bm(imgL, imgR)

        end = time.perf_counter()
        print(f"Disparity Calculation: {end - start}s")

        start = time.perf_counter()
        # convert disparity map to a depth map for more useful info
        depth_map = PointCloudUtil.disparity_to_depth_map(disparity, imgL)
        end = time.perf_counter()
        print(f"Depth Calculation: {end - start}s")

        imgL = cv2.cvtColor(
            imgL,
            # convert grayscale opencv image to RGB for point cloud color assignment
            cv2.COLOR_GRAY2RGB,
        )

        pc = PointCloudUtil.depth_map_to_points(
            depth_map,
            imgL,
            # rockDetector,
            np.array(origin_pos),
            origin_rpy,
        )

        # self.__compute_ground_planes(pc)

        return pc

    def __compute_ground_planes(self, pc):
        """
        Computes and assigns ground plane and average ground height to cell from point cloud.
        """

        gp_start = time.perf_counter()

        # indices = np.where(
        #     [
        #         [len(pc.points) >= RANSAC_THRESHOLD_POINTS for pc in row]
        #         for row in self.__pc_map
        #     ]
        # )

        # result_indices = list(zip(indices[0], indices[1]))

        # for x, y in result_indices:
        #     if len(self.__pc_map[x][y].points) >= RANSAC_THRESHOLD_POINTS:
        #         _, _, ground_plane, ground_height = (
        #             MapBuilder.calculate_ground_plane_and_height(
        #                 x, y, np.asarray(self.__pc_map[x][y].points)
        #             )
        #         )

        #         self.set_ground_plane(x, y, ground_plane)
        #         self.set_height(x, y, ground_height)

        points = np.asarray(pc.points)
        x_coords = points[:, 0]

        for x_ind in range(PointCloudUtil.CELL_COUNT):
            x_min = x_ind * PointCloudUtil.CELL_SIZE
            x_max = (x_ind + 1) * PointCloudUtil.CELL_SIZE
            x_indices = np.where((x_coords >= x_min) & (x_coords <= x_max))
            cell_points_x = points[x_indices]
            if len(cell_points_x) == 0:
                continue

            for y_ind in range(PointCloudUtil.CELL_COUNT):
                y_min = y_ind * PointCloudUtil.CELL_SIZE
                y_max = (y_ind + 1) * PointCloudUtil.CELL_SIZE
                y_indices = np.where(
                    (cell_points_x[:, 1] >= y_min) & (cell_points_x[:, 1] <= y_max)
                )
                cell_points = cell_points_x[y_indices]

                if len(cell_points) == 0:
                    continue

                if len(cell_points) >= PointCloudUtil.RANSAC_THRESHOLD_POINTS:
                    _, _, ground_plane, ground_height = (
                        MapBuilder.calculate_ground_plane_and_height(
                            x_ind, y_ind, np.asarray(cell_points)
                        )
                    )

                    self.set_ground_plane(x_ind, y_ind, ground_plane)
                    self.set_height(x_ind, y_ind, ground_height)

        gp_end = time.perf_counter()
        print(f"Ground Plane Computation and Assignment: {gp_end - gp_start}s")


# ==========================================================================================================
# MAIN METHOD
# ==========================================================================================================

if __name__ == "__main__":
    # imgL = cv2.imread("/workspace/ORB_SLAM3/LAC-Code/perception/test_imgs/2_915_0.png")
    # imgR = cv2.imread("/workspace/ORB_SLAM3/LAC-Code/perception/test_imgs/2_915_1.png")
    
    imgL = cv2.imread("test_imgs/2_915_0.png")
    imgR = cv2.imread("test_imgs/2_915_1.png")

    imgL = cv2.resize(imgL, (PointCloudUtil.IMG_WIDTH, PointCloudUtil.IMG_HEIGHT))
    imgR = cv2.resize(imgR, (PointCloudUtil.IMG_WIDTH, PointCloudUtil.IMG_HEIGHT))

    rockDetector = RockDetection()
    rockDetector.predictStereoImages(imgL, imgR)

    print("Finished Rock Detection")

    # cv2.imshow("Resized", imgL)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    profiler = cProfile.Profile()
    profiler.enable()

    map_builder = MapBuilder()

    start_time = time.perf_counter()

    points, colors = (
        map_builder.compute_sorted_point_cloud(  # use this method when calling from other files
            imgL,
            imgR,
            rockDetector.rockSpecifications,
            # None,
            origin_pos=np.array([0, 0, 0]),
            origin_rpy=(0, 0, 0),
            disparity_method="sgbm",
        )
    )

    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time} seconds")

    vis_pc = PointCloudUtil.points_to_o3d_pc(points, colors)

    profiler.disable()
    profiler.dump_stats("profile_results.prof")

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # vis_pc = o3d.geometry.PointCloud()  # initialize visualized point cloud

    # cell_count = map_builder.get_cell_count()
    for i in range(map_builder.get_cell_count()):
        # define x limits for plane
        xlim = [
            i * PointCloudUtil.CELL_SIZE,
            (i * PointCloudUtil.CELL_SIZE) + PointCloudUtil.CELL_SIZE,
        ]

        for j in range(map_builder.get_cell_count()):
            # cell_pc = map_builder.get_cell_point_cloud(i, j)
            # # visualize point cloud in cell
            # if len(np.asarray(cell_pc.points)) > 0:
            #     vis_pc = PointCloudUtil.add_point_clouds(
            #         vis_pc, cell_pc
            #     )  # add cell point cloud to aggregated visualized pc

            # define y limits for plane
            ylim = [
                j * PointCloudUtil.CELL_SIZE,
                (j * PointCloudUtil.CELL_SIZE) + PointCloudUtil.CELL_SIZE,
            ]
            cell_plane = map_builder.get_ground_plane(i, j)
            if (
                cell_plane is not None and cell_plane[2] > 1e-6
            ):  # check that there is a plane for the cell and that the z coefficient is not close to 0
                plane_mesh = PointCloudUtil.create_visualization_plane_mesh(  # make a mesh object for visualizing the plane
                    cell_plane, xlim=xlim, ylim=ylim
                )
                if map_builder.get_rock_flag(i, j):
                    plane_mesh.paint_uniform_color([1.0, 0.0, 0.0])
                vis.add_geometry(plane_mesh)

    vis.add_geometry(vis_pc)  # add visualization pc to visualizer
    # vis.add_geometry(pc)  # add visualization pc to visualizer

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(  # show a coordinate frame for frame of reference
        size=1.0, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)  # add coord frame to visualizer

    vis.run()
    vis.destroy_window()
