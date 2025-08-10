#qos_auto_tuner.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import psutil
import threading

class QoSAutoTuner(Node):
    def __init__(self):
        super().__init__('qos_auto_tuner')

        self.timer = self.create_timer(5.0, self.tune_qos)

        self.current_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

    def tune_qos(self):
        cpu_load = psutil.cpu_percent() / 100.0

        if cpu_load > 0.8:
            new_depth = 5
            new_reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            new_depth = 20
            new_reliability = ReliabilityPolicy.RELIABLE

        if (self.current_qos.depth != new_depth or
            self.current_qos.reliability != new_reliability):

            self.current_qos.depth = new_depth
            self.current_qos.reliability = new_reliability
            self.get_logger().info(f"QoS changed: depth={new_depth}, reliability={new_reliability}")

def main(args=None):
    rclpy.init(args=args)
    node = QoSAutoTuner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
