import rclpy
from rclpy.node import Node
from mppi import MPPI_controller
from std_msgs.msg import Float32

class NavNode(Node):
    def __init__(self, init_pos):
        super().__init__("nav_node")
        self.nav = MPPI_controller(horizons=120, rollouts=100, dt=.1, temperature=0.01, noise_std=0.1, init_pos=init_pos)

        # publishes an array of two floats
        self.nav_pub = self.create_publisher(Float32, "/optimal_output", 10)
        self.nav_rock_listener = self.create_subscription(Float32, "/rock_locations", self.nav_rock_callback, 10)
        self.nav_location_update = self.create_subscription(Float32, "/location_update", self.nav_location_callback, 10)

    def nav_rock_callback(self, msg):
        self.nav.add_rock(msg.data)

    def nav_location_callback(self, msg):
        self.nav.set_state(msg.data)
        control = self.nav.run()
        self.nav_pub.publish(control)

def main():
    rclpy.init()
    test_img_pub = NavNode()

    try:
        rclpy.spin(test_img_pub)
    except KeyboardInterrupt:
        pass

    test_img_pub.destroy_node()
    rclpy.shutdown()