# synthetic_multimodal_publisher.py — simulates drone / IoT edge sensor streams
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header


class SyntheticMultimodalPublisher(Node):
    """
    Publishes multimodal streams tuned by LYDLR_VERTICAL:
      - drone: 10 Hz, IMU maneuvers, sparse LiDAR, cmd_vel telemetry
      - iot:   2 Hz low-res vision, steady IMU, minimal LiDAR
      - default: legacy 5 Hz mixed feed
    """

    PROFILES = {
        "drone": {
            "hz": 10.0,
            "img_size": 224,
            "lidar_points": 64,
            "audio_samples": 4096,
            "publish_cmd_vel": True,
            "uplink_kbps": 512,
        },
        "iot": {
            "hz": 2.0,
            "img_size": 128,
            "lidar_points": 32,
            "audio_samples": 1024,
            "publish_cmd_vel": False,
            "uplink_kbps": 64,
        },
        "default": {
            "hz": 5.0,
            "img_size": 224,
            "lidar_points": 100,
            "audio_samples": 16000,
            "publish_cmd_vel": False,
            "uplink_kbps": 256,
        },
    }

    def __init__(self):
        super().__init__("synthetic_multimodal_publisher")
        vertical = os.getenv("LYDLR_VERTICAL", os.getenv("LYDLR_PROFILE", "drone")).lower()
        self.profile = self.PROFILES.get(vertical, self.PROFILES["default"])
        self.vertical = vertical if vertical in self.PROFILES else "default"
        self._tick = 0

        self.pub_img = self.create_publisher(Image, "/camera/image_raw", 10)
        self.pub_imu = self.create_publisher(Float32MultiArray, "/imu/data", 10)
        self.pub_lidar = self.create_publisher(Float32MultiArray, "/lidar/data", 10)
        self.pub_audio = self.create_publisher(Float32MultiArray, "/audio/data", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        period = 1.0 / self.profile["hz"]
        self.timer = self.create_timer(period, self.publish_all_sensors)

        self.get_logger().info(
            f"📡 Synthetic publisher [{self.vertical}] @ {self.profile['hz']} Hz "
            f"(~{self.profile['uplink_kbps']} kbps budget)"
        )

    def publish_all_sensors(self):
        now = self.get_clock().now().to_msg()
        t = self._tick * 0.1
        self._tick += 1
        p = self.profile

        img_h = img_w = p["img_size"]
        if self.vertical == "drone":
            base = 128 + 40 * np.sin(t)
            img = np.clip(
                base
                + 30 * np.random.randn(img_h, img_w, 3)
                + np.linspace(0, 80, img_w)[None, :, None],
                0,
                255,
            ).astype(np.uint8)
        else:
            img = np.clip(
                90 + 15 * np.sin(t * 0.3) + 8 * np.random.randn(img_h, img_w, 3),
                0,
                255,
            ).astype(np.uint8)

        img_msg = Image()
        img_msg.header = Header(stamp=now, frame_id=f"{self.vertical}_camera")
        img_msg.height = img_h
        img_msg.width = img_w
        img_msg.encoding = "rgb8"
        img_msg.is_bigendian = 0
        img_msg.step = img_w * 3
        img_msg.data = img.flatten().tobytes()
        self.pub_img.publish(img_msg)

        if self.vertical == "drone":
            imu = np.array(
                [
                    0.2 * math.sin(t),
                    0.15 * math.cos(t * 1.3),
                    9.81,
                    0.05 * math.sin(t * 2),
                    0.04 * math.cos(t),
                    0.02 * math.sin(t * 0.7),
                ],
                dtype=np.float32,
            )
        else:
            imu = np.array([0.01, -0.02, 9.81, 0.001, 0.002, 0.001], dtype=np.float32)
            imu += np.random.normal(0, 0.005, 6).astype(np.float32)

        imu_msg = Float32MultiArray()
        imu_msg.data = imu.tolist()
        self.pub_imu.publish(imu_msg)

        n = p["lidar_points"]
        if self.vertical == "drone":
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            radius = 8 + 2 * np.sin(t + angles)
            lidar = np.stack(
                [
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    1.5 + 0.3 * np.random.randn(n),
                ],
                axis=1,
            ).astype(np.float32)
        else:
            lidar = np.random.uniform(-3, 3, (n, 3)).astype(np.float32)

        lidar_msg = Float32MultiArray()
        lidar_msg.data = lidar.flatten().tolist()
        self.pub_lidar.publish(lidar_msg)

        audio_n = p["audio_samples"]
        if self.vertical == "drone":
            audio = (0.3 * np.sin(2 * math.pi * 440 * np.arange(audio_n) / 16000)).astype(
                np.float32
            )
            audio += 0.05 * np.random.randn(audio_n).astype(np.float32)
        else:
            audio = (0.05 * np.random.randn(audio_n)).astype(np.float32)

        audio_msg = Float32MultiArray()
        audio_msg.data = audio.tolist()
        self.pub_audio.publish(audio_msg)

        if p["publish_cmd_vel"]:
            twist = Twist()
            twist.linear.x = 2.0 * math.sin(t * 0.5)
            twist.linear.y = 0.5 * math.cos(t * 0.3)
            twist.angular.z = 0.3 * math.sin(t * 0.8)
            self.pub_cmd.publish(twist)

        if self._tick % int(max(p["hz"], 1)) == 0:
            self.get_logger().info(
                f"📤 [{self.vertical}] frame {self._tick} | LiDAR {n} pts | "
                f"budget {p['uplink_kbps']} kbps"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticMultimodalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
