# synchronizer_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from message_filters import ApproximateTimeSynchronizer, Subscriber

class SynchronizerNode(Node):
    def __init__(self):
        super().__init__('synchronizer_node')
        
        # Create subscribers
        left_image_sub = Subscriber(self, Image, '/stereo/left_image')
        right_image_sub = Subscriber(self, Image, '/stereo/right_image')
        left_camera_info_sub = Subscriber(self, CameraInfo, '/stereo/left_camera_info')
        right_camera_info_sub = Subscriber(self, CameraInfo, '/stereo/right_camera_info')
        imu_sub = Subscriber(self, Imu, '/stereo/imu')

        # Create synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [left_image_sub, right_image_sub, left_camera_info_sub, right_camera_info_sub, imu_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # Create publishers
        self.left_image_pub = self.create_publisher(Image, '/sync/left_image', 10)
        self.right_image_pub = self.create_publisher(Image, '/sync/right_image', 10)
        self.left_camera_info_pub = self.create_publisher(CameraInfo, '/sync/left_camera_info', 10)
        self.right_camera_info_pub = self.create_publisher(CameraInfo, '/sync/right_camera_info', 10)
        self.imu_pub = self.create_publisher(Imu, '/sync/imu', 10)

    def sync_callback(self, left_image, right_image, left_camera_info, right_camera_info, imu):
        self.left_image_pub.publish(left_image)
        self.right_image_pub.publish(right_image)
        self.left_camera_info_pub.publish(left_camera_info)
        self.right_camera_info_pub.publish(right_camera_info)
        self.imu_pub.publish(imu)

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()