import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Robot State Publisher
    description_dir = get_package_share_directory("lac_description")
    urdf = PathJoinSubstitution([description_dir, "urdf", "robot.urdf.xacro"])
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{ "robot_description" : urdf }]
    )

    # Include RTAB-Map Launch File
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            get_package_share_directory("rtabmap_launch"), "launch", "rtabmap.launch.py"
        ])),
        launch_arguments={
            "frame_id": "base_link",
            "rtabmap_viz": "true", # TODO: REMOVE
            "rviz2": "true", # TODO: REMOVE
            "visual_odometry": "true",
            "stereo": "true",
            "left_image_topic_relay": "/front/left_image",
            "right_image_topic_relay": "/front/right_image",
            "left_camera_info_topic": "/camera_info",
            "right_camera_info_topic": "/camera_info",
            "imu_topic": "/imu"
        }.items()
    )

    # YOLO Segmentation Detector
    yolo_seg_detector = Node(
        package="perception",
        executable="yolo_seg_detector",
        name="yolo_seg_detector",
        output="screen",
    )

    # Camera Info Publisher
    camera_info_publisher = Node(
        package="perception",
        executable="camera_info_publisher",
        name="camera_info_publisher",
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        # yolo_seg_detector,
        # slam_pc_publisher,
        # slam_rgbd_publisher,
        slam_stereo_publisher,

        # =========================================================
        # RTABMAP SETUP
        # =========================================================

        # # STEREO SETUP --------------------------------------------
        Node(
            package='rtabmap_odom',
            executable='stereo_odometry',
            name='stereo_odometry',
            output='screen',
            parameters=[{
                # 'frame_id': 'base_link',
                # 'odom_frame_id': 'odom',
                # 'wait_for_transform': 0.2,
                # # 'approx_sync': True,
                # # 'approx_sync_max_interval': 0.03,
                # 'queue_size': 10,

                # 'subscribe_imu': True,
                # 'wait_imu_to_init': True,
                # 'imu_filter_angular_velocity': True,
                # # 'publish_tf': True,

                # # IMU parameters
                # 'Imu/MaxGyroBias': '0.1',        # Maximum gyroscope bias (rad/s)
                # 'Imu/MaxAccelBias': '0.5',       # Maximum accelerometer bias (m/s^2)
                # 'Imu/GravityNorm': '1.62',       # Gravity norm for acceleration

                # # IMU fusion weights (0=min trust, 100=max trust)
                # 'Imu/LinearVelocityWeight': '0.0',  # Weight for linear velocity from IMU
                # 'Imu/AngularVelocityWeight': '90.0', # High weight for rotation from IMU
                # 'Imu/AccelerationWeight': '50.0',    # Weight for acceleration from IMU


                # # SGBM Configuration
                # 'Stereo/WinSize': '11',
                # 'Stereo/DynamicSeed': 'true',
                # 'Stereo/Mode': '1',
                
                # # SGBM specific parameters
                # 'StereoSGBM/NumDisparities': '304',
                # 'StereoSGBM/BlockSize': '11',
                # 'StereoSGBM/MinDisparity': '0',
                # 'StereoSGBM/PreFilterCap': '1',
                # 'StereoSGBM/UniquenessRatio': '10',
                # 'StereoSGBM/SpeckleWindowSize': '16',
                # 'StereoSGBM/SpeckleRange': '1',
                # 'StereoSGBM/Disp12MaxDiff': '304',
                # 'StereoSGBM/P1': '72',
                # 'StereoSGBM/P2': '288',

                # # Depth filtering parameters
                # 'Stereo/MaxDepth': '10.0',
                # 'Stereo/MinDepth': '0.05',
                # 'Stereo/Eps': '0.01',
                # 'Stereo/OpticalFlow': 'true',
                # 'Stereo/Iterations': '30',
                # 'Stereo/MaxDisparity': '304',
                # 'Stereo/MaxLevel': '3',
                # 'Stereo/MinDisparity': '0',
                # 'Stereo/SSD': 'True',

                # # Visual odometry settings - increased weight on visual
                # 'Vis/FeatureType': '1',
                # 'Vis/EstimationType': '1',
                # 'Vis/MinInliers': '10',
                # 'Vis/InlierDistance': '0.3',
                # 'Vis/MaxDepth': '10.0',
                # 'Vis/MinDepth': '0.1',
                # 'Vis/MaxFeatures': '600',
                # 'Vis/MaxSpeed': '2.0',
                # 'Vis/MaxAngularSpeed': '1.0',

                # # Disable point cloud publishing in odometry
                # 'publish_cloud': 'true',
                # 'publish_scan': 'true',
                # # 'subscribe_scan_cloud': 'true',

                # # Enhanced ICP odometry configuration
                # 'Odom/Strategy': '0',           # Frame to frame
                # # 'OdomF2M/ScanMatching': 'true',
                # # 'OdomF2M/ScanMatchingICP': 'true',
                # # 'OdomF2M/ScanSubtractMeanK': 'true',
                # # 'OdomF2M/ScanSubtractRadius': '0.5',
                # # 'OdomF2M/ICPMaxRotation': '0.26',
                # # 'OdomF2M/ICPMaxTranslation': '0.5',
                # 'OdomF2M/MaxSize': '1000',
                # 'OdomF2M/MaxNewFeatures': '300',

                # # Enable both visual and ICP bundle adjustment
                # 'OdomF2M/BundleAdjustment': 'true',
                # # 'OdomF2M/ScanBundleAdjustment': 'true',

                # # ICP configuration for odometry
                # # 'Icp/VoxelSize': '0.05',
                # # 'Icp/MaxCorrespondenceDistance': '0.2',
                # # 'Icp/PointToPlane': 'true',
                # # 'Icp/CorrespondenceRatio': '0.3',
                # # 'Icp/Iterations': '50',
                # # 'Icp/Robust': 'true',

                # # Increased visual weight (0.3 ICP, 0.7 visual)
                # # 'OdomF2M/ScanMatchingICPWeight': '0.3',

                # # Add motion constraints for non-holonomic robot
                # 'Odom/GuessMotion': 'true',
                # 'Odom/FilteringStrategy': '1',    # 1=Kalman filtering
                # 'Odom/HolonomicRobot': 'false',   # Important: Set to false for non-strafing robot
                # 'Odom/VarianceLinear': '0.001',   # Reduce trust in lateral motion
                # # 'Odom/VarianceAngular': '0.001',
                # 'Odom/ParticleSize': '200',       # Increase particle count for better motion estimation







                'frame_id': 'fl_camera',
                # 'odom_frame_id': 'odom',
                'wait_for_transform': 0.2,
                'queue_size': 10,
                'subscribe_imu': True,
                'wait_imu_to_init': True,
                'imu_filter_angular_velocity': True,

                'approx_sync': True,
                'approx_sync_max_interval': 0.03,

                # Improved IMU integration
                'Imu/MaxGyroBias': '0.05',
                'Imu/MaxAccelBias': '0.2',
                'Imu/GravityNorm': '1.62',
                'Imu/LinearVelocityWeight': '0.0',
                'Imu/AngularVelocityWeight': '100.0',
                'Imu/AccelerationWeight': '80.0',

                # Enhanced visual odometry settings
                'Vis/EstimationType': '1',
                'Vis/MaxDepth': '10.0',
                'Vis/MinInliers': '20',
                'Vis/InlierDistance': '0.1',
                'Vis/MaxFeatures': '1000',
                'Vis/FeatureType': '6',  # FAST/BRIEF
                'Vis/MaxSpeed': '1.5',
                'Vis/MaxAngularSpeed': '0.8',

                # Improved ICP odometry
                'Odom/Strategy': '1',  # 1 = Frame to Map
                'Odom/GuessMotion': 'true',
                'Odom/FilteringStrategy': '1',
                'Odom/ParticleSize': '500',
                'Odom/ParticleNoiseT': '0.002',
                'Odom/ParticleNoiseR': '0.002',

                'OdomF2M/ScanMaxSize': '15000',
                'OdomF2M/ScanSubtractRadius': '0.1',
                'OdomF2M/ScanMaxRange': '20.0',
            }],
            remappings=[
                ('left/image_rect', '/stereo/left_image'),
                ('right/image_rect', '/stereo/right_image'),
                ('left/camera_info', '/stereo/left_camera_info'),
                ('right/camera_info', '/stereo/right_camera_info'),
                ('imu', '/stereo/imu'),
                # ('scan_cloud', '/pc'),
                ('odom', 'odom')
            ]
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                # 'frame_id': 'base_link',
                # 'map_frame_id': 'map',
                # 'odom_frame_id': 'odom',
                # 'subscribe_stereo': True,
                # 'subscribe_depth': False,
                # 'subscribe_scan': False,
                # 'subscribe_rgb': False,
                # 'subscribe_scan_cloud': True,
                # 'subscribe_imu': True,           # Enable IMU for SLAM
                # # 'approx_sync': True,
                # # 'approx_sync_max_interval': 0.03,
                # 'queue_size': 10,

                # # IMU integration for SLAM
                # # 'Imu/MaxGyroBias': '0.1',
                # # 'Imu/MaxAccelBias': '0.5',
                # # 'Imu/GravityNorm': '1.62',

                # # # SLAM parameters
                # # 'RGBD/AngularUpdate': '0.01',    # More frequent updates
                # # 'RGBD/LinearUpdate': '0.01',
                # # 'RGBD/OptimizeFromGraphEnd': 'true',

                # # Loop closure with IMU
                # 'RGBD/LoopClosureReextractFeatures': 'true',
                # 'RGBD/LoopClosureMaxDistance': '0.5',
                # 'Reg/Strategy': '0',              # Visual

                # # Point cloud settings
                # 'publish_cloud': 'false',
                # 'publish_cloud_map': 'true',

                # # Graph optimization with IMU constraints
                # 'Optimizer/Strategy': '1',        # g2o
                # 'Optimizer/Iterations': '100',
                # 'g2o/RobustKernelDelta': '8',    # Robust optimization
                # 'g2o/ConvergenceTolerance': '1e-6'






                'frame_id': 'base_link',
                'map_frame_id': 'map',
                'odom_frame_id': 'odom',
                'subscribe_stereo': True,
                'subscribe_depth': False,
                'subscribe_scan_cloud': True,
                'subscribe_imu': True,
                'queue_size': 10,

                'approx_sync': True,
                'approx_sync_max_interval': 0.03,

                # Improved SLAM parameters
                'Mem/IncrementalMemory': 'true',
                'Mem/InitWMWithAllNodes': 'true',
                'RGBD/NeighborLinkRefining': 'true',
                'RGBD/ProximityBySpace': 'true',
                'RGBD/AngularUpdate': '0.01',
                'RGBD/LinearUpdate': '0.01',
                'RGBD/OptimizeFromGraphEnd': 'true',
                'Grid/FromDepth': 'true',
                'Grid/RayTracing': 'true',

                'Odom/ResetCountdown': '10',

                # Loop closure refinement
                'RGBD/LoopClosureReextractFeatures': 'true',
                'Reg/Strategy': '2',  # 2 = Visual + ICP
                'Reg/Force3DoF': 'true',
                'Icp/VoxelSize': '0.05',
                'Icp/MaxCorrespondenceDistance': '0.1',
            }],
            remappings=[
                ('left/image_rect', '/stereo/left_image'),
                ('right/image_rect', '/stereo/right_image'),
                ('left/camera_info', '/stereo/left_camera_info'),
                ('right/camera_info', '/stereo/right_camera_info'),
                # ('rgb/image', '/rgb/image'),
                # ('rgb/camera_info', '/stereo/left_camera_info'),
                ('scan_cloud', '/pc'),
                ('imu', '/stereo/imu'),            # Add IMU topic remapping
                ('odom', 'odom')
            ],
            arguments=[LaunchConfiguration('rtabmap_args', default='--delete_db_on_start')]
        )

        # RGBD SETUP (WAYYY TOO BOUNCY) --------------------------------------------------------------
        # Node(
        #     package='rtabmap_odom',
        #     executable='rgbd_odometry',
        #     name='rgbd_odometry',
        #     output='screen',
        #     parameters=[{
        #         'frame_id': 'base_link',
        #         'odom_frame_id': 'odom',
        #         'subscribe_depth': True,
        #         'subscribe_rgb': True,
        #         # 'subscribe_rgbd': True,
        #         # 'approx_sync': True
        #     }],
        #     remappings=[
        #         ('rgb/image', '/rgbd_odom/rgb_image'),
        #         ('depth/image', '/rgbd_odom/depth_image'),
        #         ('rgb/camera_info', '/rgbd_odom/camera_info')
        #     ]
        # ),
        # Node(
        #     package='rtabmap_slam',
        #     executable='rtabmap',
        #     name='rtabmap',
        #     output='screen',
        #     parameters=[{
        #         'frame_id': 'base_link',
        #         'subscribe_depth': True,
        #         'subscribe_rgb': True,
        #         'subscribe_odom_info': True,
        #         # 'approx_sync': True
        #     }],
        #     remappings=[
        #         ('rgb/image', '/rgbd_odom/rgb_image'),
        #         ('depth/image', '/rgbd_odom/depth_image'),
        #         ('rgb/camera_info', '/rgbd_odom/camera_info'),
        #         ('odom', '/odom')
        #     ],
        #     arguments=['--delete_db_on_start']
        # )
    ])