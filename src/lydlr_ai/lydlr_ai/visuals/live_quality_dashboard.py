import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import threading

class LiveQualityDashboard(Node):
    def __init__(self):
        super().__init__('live_quality_dashboard')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/quality_metrics',
            self.quality_callback,
            10)

        self.lpips_vals = []
        self.psnr_vals = []
        self.ssim_vals = []

        self.fig, self.axs = plt.subplots(3,1)
        self.thread = threading.Thread(target=self._plt_loop)
        self.thread.daemon = True
        self.thread.start()

    def quality_callback(self, msg):
        data = msg.data
        self.lpips_vals.append(data[0])
        self.psnr_vals.append(data[1])
        self.ssim_vals.append(data[2])

        if len(self.lpips_vals) > 100:
            self.lpips_vals.pop(0)
            self.psnr_vals.pop(0)
            self.ssim_vals.pop(0)

    def _plt_loop(self):
        plt.ion()
        while rclpy.ok():
            self.axs[0].clear()
            self.axs[1].clear()
            self.axs[2].clear()

            self.axs[0].plot(self.lpips_vals, label='LPIPS')
            self.axs[1].plot(self.psnr_vals, label='PSNR')
            self.axs[2].plot(self.ssim_vals, label='SSIM')

            self.axs[0].legend()
            self.axs[1].legend()
            self.axs[2].legend()

            plt.pause(0.5)

def main(args=None):
    rclpy.init(args=args)
    node = LiveQualityDashboard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
