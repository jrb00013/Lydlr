import rclpy
from rclpy.node import Node
import psutil
import time

class BandwidthMonitorNode(Node):
    def __init__(self):
        super().__init__('bandwidth_monitor_node')

        self.declare_parameter('monitor_interval', 1.0)
        self.interval = self.get_parameter('monitor_interval').value

        self.prev_bytes_sent = psutil.net_io_counters().bytes_sent
        self.prev_bytes_recv = psutil.net_io_counters().bytes_recv
        self.prev_time = time.time()

        self.timer = self.create_timer(self.interval, self.monitor_callback)

    def monitor_callback(self):
        current_bytes_sent = psutil.net_io_counters().bytes_sent
        current_bytes_recv = psutil.net_io_counters().bytes_recv
        current_time = time.time()

        delta_sent = current_bytes_sent - self.prev_bytes_sent
        delta_recv = current_bytes_recv - self.prev_bytes_recv
        delta_time = current_time - self.prev_time

        bandwidth_sent = delta_sent / delta_time / 1024.0  # KB/s
        bandwidth_recv = delta_recv / delta_time / 1024.0  # KB/s

        self.get_logger().info(f"Bandwidth sent: {bandwidth_sent:.2f} KB/s, received: {bandwidth_recv:.2f} KB/s")

        self.prev_bytes_sent = current_bytes_sent
        self.prev_bytes_recv = current_bytes_recv
        self.prev_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = BandwidthMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
