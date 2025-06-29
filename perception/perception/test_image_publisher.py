import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from lac_interfaces.msg import ImagePair

import cv2

from cv_bridge import CvBridge


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher_node")

        self.front_pub = self.create_publisher(ImagePair, "/front_stereo_img_pair", 10)
        # self.front_right_pub = self.create_publisher(Image, "/front_right_test_img", 10)

        self.bridge = CvBridge()

        self.images = [
            "1_155",
            "1_179",
            "1_185",
            "1_191",
            "1_247",
            "1_283",
            "2_1357",
            "2_915",
        ]

        self.i = 0

        self.timer = self.create_timer(0.5, self._timer_callback)

    def _timer_callback(self):
        # imgL = cv2.imread(
        #     f"/workspace/ORB_SLAM3/LAC-Code/perception/test_imgs/{self.images[self.i]}_0.png"
        # )
        # imgR = cv2.imread(
        #     f"/workspace/ORB_SLAM3/LAC-Code/perception/test_imgs/{self.images[self.i]}_1.png"
        # )
        imgL = cv2.imread(
            f"/workspace/team_code/perception/test_imgs/{self.images[self.i]}_0.png"
        )
        imgR = cv2.imread(
            f"/workspace/team_code/perception/test_imgs/{self.images[self.i]}_1.png"
        )

        if imgL is not None and imgR is not None:
            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            print(f"L Image Shape: {imgL.shape}")
            print(f"R Image Shape: {imgR.shape}")

            ros_left_image = self.bridge.cv2_to_imgmsg(imgL, encoding="mono8")
            ros_right_image = self.bridge.cv2_to_imgmsg(imgR, encoding="mono8")

            img_pair_msg = ImagePair()
            img_pair_msg.left = ros_left_image
            img_pair_msg.right = ros_right_image

            self.front_pub.publish(img_pair_msg)

            print("Published stereo pair")

        self.i += 1
        self.i = self.i % len(self.images)


def main():
    rclpy.init()
    test_img_pub = TestImagePublisher()

    try:
        rclpy.spin(test_img_pub)
    except KeyboardInterrupt:
        pass

    test_img_pub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
