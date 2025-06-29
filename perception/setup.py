from setuptools import find_packages, setup

package_name = "perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pc_generation_node = perception.pc_generation_node:main",
            "ground_plane_node = perception.ground_plane_node:main",
            "pc_aggregator_node = perception.pc_aggregator_node:main",
            "yolo_bbox_detector = perception.yolo_bbox_detector:main",
            "yolo_seg_detector = perception.yolo_seg_detector:main",
            "camera_info_publisher = perception.camera_info_publisher:main",
            "slam_pc_publisher = perception.slam_pc_publisher:main",
            "slam_rgbd_publisher = perception.slam_rgbd_publisher:main",
            "slam_stereo_publisher = perception.slam_stereo_publisher:main",
            "msg_synchronizer_node = perception.msg_synchronizer_node:main",
        ],
    },
)
