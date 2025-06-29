import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration, FindExecutable
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    description_dir = get_package_share_directory("lac_description")
    urdf = PathJoinSubstitution([description_dir, "urdf", "robot.urdf.xacro"])
    
    robot_description = Command(["xacro ", urdf])

    # robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            { "robot_description" : robot_description },
        ]
    )

    # YOLO Segmentation Detector
    yolo_seg_detector = Node(
        package="perception",
        executable="yolo_seg_detector",
        name="yolo_seg_detector",
        output="screen",
    )

    # slam PC publisher for ICP odom
    slam_pc_publisher = Node(
        package="perception",
        executable="slam_pc_publisher",
        name="slam_pc_publisher",
        output="screen",
    )

    # slam publisher for RGBD odom
    slam_rgbd_publisher = Node(
        package="perception",
        executable="slam_rgbd_publisher",
        name="slam_rgbd_publisher",
        output="screen",
    )

    # slam input publisher for stereo RGBD odom
    slam_stereo_publisher = Node(
        package="perception",
        executable="slam_stereo_publisher",
        name="slam_stereo_publisher",
        output="screen",
    )
    
    ground_plane_node = Node(
        package="perception",
        executable="ground_plane_node",
        name="ground_plane_node"
    )

    return LaunchDescription([
        robot_state_publisher,
        # yolo_seg_detector,
        slam_pc_publisher,
        # slam_rgbd_publisher,
        # slam_stereo_publisher,
        # ground_plane_node,

        # =========================================================
        # RTABMAP SETUP
        # =========================================================

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             FindPackageShare('rtabmap_sync'),
        #             'launch',
        #             'rgbd_sync.launch.py'
        #         ])
        #     ]),

        # )

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('rtabmap_launch'),
                    'launch',
                    'rtabmap.launch.py'
                ])
            ]),
            launch_arguments={
                "args":"--delete_db_on_start --Odom/ResetCountdown 1",
                "stereo":"false",
                # "left_image_topic":"/stereo/left_image",
                # "right_image_topic":"/stereo/right_image",
                # "left_camera_info_topic":"/stereo/left_camera_info",
                # "right_camera_info_topic":"/stereo/right_camera_info",
                "depth_topic": "/front/depth",
                "rgb_topic": "/front/rgb",
                "camera_info_topic": "/front/camera_info",

                # "subscribe_scan_cloud": "true",
                # "scan_cloud_topic": "/pc",

                # "imu_topic":"/stereo/imu",
                "frame_id":"base_link",
                "approx_sync":"true",
                "approx_sync_max_interval":"0.01",
                "sync_queue_size":"10",
                "topic_queue_size":"10",
                # 'Odom/FillInfoDepth': 'false',
                # 'RGBD/ProximityFiltering': 'true',
                # "wait_imu_to_init":"true",
                # 'cloud_voxel_size': '0.05',

                # SGBM SETUP
                # 'Stereo/SGBM': "true",
                # 'StereoSGBM/Enabled': "true",
                # "Stereo/MaxDisparity": '288',

                # "StereoSGBM/Enabled": "true",
                # 'StereoSGBM/NumDisparities': '288',
                # 'StereoSGBM/BlockSize': '11',
                # 'StereoSGBM/MinDisparity': '0',
                # 'StereoSGBM/PreFilterCap': '1',
                # 'StereoSGBM/UniquenessRatio': '10',
                # 'StereoSGBM/SpeckleWindowSize': '16',
                # 'StereoSGBM/SpeckleRange': '1',
                # 'StereoSGBM/Disp12MaxDiff': '288',
                # 'StereoSGBM/P1': '72',
                # 'StereoSGBM/P2': '288',
                # 'cloud_decimation': '4',
                'rtabmap_viz': "false"
            }.items()
        )
    ])