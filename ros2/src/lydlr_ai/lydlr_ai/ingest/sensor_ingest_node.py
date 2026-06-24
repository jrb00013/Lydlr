"""
Real sensor ingest for Lydlr — publishes to the shared sensor bus.

Modes (LYDLR_SENSOR_SOURCE):
  camera  — USB/V4L2 via OpenCV → /camera/image_raw (+ synthetic IMU/LiDAR if no replay)
  replay  — NPZ clip recorded by scripts/record_sensor_clip.py
  rosbag  — shell out to `ros2 bag play` (requires bag path in LYDLR_ROSBAG_PATH)

Synthetic multimodal publisher remains the default when this node is not used.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header

from lydlr_ai.communication.modality_codec import audio_to_mel


class SensorIngestNode(Node):
    PROFILES = {
        "drone": {"hz": 10.0, "img_size": 224, "lidar_points": 64, "audio_samples": 4096},
        "iot": {"hz": 2.0, "img_size": 128, "lidar_points": 32, "audio_samples": 1024},
        "warehouse": {"hz": 5.0, "img_size": 160, "lidar_points": 96, "audio_samples": 2048},
    }

    def __init__(self):
        super().__init__("sensor_ingest")
        self.source = os.getenv("LYDLR_SENSOR_SOURCE", "replay").lower()
        vertical = os.getenv("LYDLR_VERTICAL", "drone").lower()
        self.vertical = vertical if vertical in self.PROFILES else "drone"
        self.profile = self.PROFILES[self.vertical]
        self._audio_sr = int(os.getenv("LYDLR_AUDIO_SAMPLE_RATE", "16000"))
        self._tick = 0
        self._replay_frames = []
        self._camera = None
        self._bag_proc = None

        self.pub_img = self.create_publisher(Image, "/camera/image_raw", 10)
        self.pub_imu = self.create_publisher(Float32MultiArray, "/imu/data", 10)
        self.pub_lidar = self.create_publisher(Float32MultiArray, "/lidar/data", 10)
        self.pub_audio = self.create_publisher(Float32MultiArray, "/audio/data", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        if self.source == "camera":
            self._init_camera()
        elif self.source == "replay":
            self._load_replay_clip()
        elif self.source == "rosbag":
            self._start_rosbag_play()
        else:
            self.get_logger().error(f"Unknown LYDLR_SENSOR_SOURCE={self.source}")
            raise SystemExit(1)

        period = 1.0 / self.profile["hz"]
        self.timer = self.create_timer(period, self._publish_tick)
        self.get_logger().info(
            f"📥 Sensor ingest [{self.source}/{self.vertical}] @ {self.profile['hz']} Hz"
        )

    def _init_camera(self):
        try:
            import cv2
        except ImportError as exc:
            self.get_logger().error(f"OpenCV required for camera ingest: {exc}")
            raise SystemExit(1) from exc

        device = int(os.getenv("LYDLR_CAMERA_DEVICE", "0"))
        self._cv2 = cv2
        self._camera = cv2.VideoCapture(device)
        if not self._camera.isOpened():
            self.get_logger().warn(f"Camera {device} unavailable — falling back to synthetic frames")
            self._camera = None

    def _load_replay_clip(self):
        path = os.getenv(
            "LYDLR_REPLAY_PATH",
            str(Path(__file__).resolve().parents[4] / "data" / "demo_clips" / f"{self.vertical}_clip.npz"),
        )
        clip = Path(path)
        if not clip.exists():
            self.get_logger().warn(f"Replay clip missing: {clip} — generating inline synthetic clip")
            self._replay_frames = self._build_inline_clip()
            return

        data = np.load(clip, allow_pickle=True)
        self._replay_frames = list(data["frames"])
        if "hz" in data:
            self.profile["hz"] = float(data["hz"])
        self.get_logger().info(f"Loaded {len(self._replay_frames)} frames from {clip}")

    def _build_inline_clip(self, n: int = 120):
        p = self.profile
        h = w = p["img_size"]
        frames = []
        for i in range(n):
            t = i * 0.1
            img = np.clip(
                128 + 30 * np.sin(t) + 20 * np.random.randn(h, w, 3),
                0,
                255,
            ).astype(np.uint8)
            frames.append(
                {
                    "image": img,
                    "imu": np.array(
                        [0.1 * math.sin(t), 0.1 * math.cos(t), 9.81, 0, 0, 0],
                        dtype=np.float32,
                    ),
                    "lidar": np.random.randn(p["lidar_points"]).astype(np.float32),
                    "audio": self._audio_features(self._synth_audio_wave(i, p["audio_samples"])),
                }
            )
        return frames

    def _synth_audio_wave(self, tick: int, n_samples: int) -> np.ndarray:
        t = np.linspace(0, 1, n_samples, dtype=np.float32)
        wave = 0.02 * np.sin(2 * math.pi * (220 + 30 * math.sin(tick * 0.1)) * t)
        wave += 0.005 * np.random.randn(n_samples).astype(np.float32)
        return wave.astype(np.float32)

    def _audio_features(self, wave: np.ndarray) -> np.ndarray:
        return audio_to_mel(wave, sample_rate=self._audio_sr)

    def _start_rosbag_play(self):
        bag = os.getenv("LYDLR_ROSBAG_PATH", "")
        if not bag or not Path(bag).exists():
            self.get_logger().error("LYDLR_ROSBAG_PATH must point to an existing rosbag directory")
            raise SystemExit(1)

        cmd = ["ros2", "bag", "play", bag, "--loop"]
        self.get_logger().info(f"Starting rosbag play: {' '.join(cmd)}")
        self._bag_proc = subprocess.Popen(cmd)
        # rosbag publishes directly — this node only monitors heartbeat
        self.create_timer(5.0, self._bag_health_check)

    def _bag_health_check(self):
        if self._bag_proc and self._bag_proc.poll() is not None:
            self.get_logger().warn("rosbag play exited — restarting")
            self._start_rosbag_play()

    def _publish_tick(self):
        if self.source == "rosbag":
            return

        now = self.get_clock().now().to_msg()
        if self.source == "camera":
            frame = self._read_camera_frame()
        else:
            if not self._replay_frames:
                return
            frame = self._replay_frames[self._tick % len(self._replay_frames)]
            self._tick += 1

        self._publish_frame(frame, now)

        if self.vertical == "drone":
            cmd = Twist()
            cmd.linear.x = 0.5 * math.sin(self._tick * 0.05)
            self.pub_cmd.publish(cmd)

    def _read_camera_frame(self):
        p = self.profile
        h = w = p["img_size"]
        if self._camera is not None:
            ok, bgr = self._camera.read()
            if ok:
                rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                rgb = self._cv2.resize(rgb, (w, h))
                imu = np.array([0, 0, 9.81, 0, 0, 0], dtype=np.float32)
                return {
                    "image": rgb,
                    "imu": imu,
                    "lidar": np.random.randn(p["lidar_points"]).astype(np.float32),
                    "audio": self._audio_features(self._synth_audio_wave(self._tick, p["audio_samples"])),
                }

        t = self._tick * 0.1
        self._tick += 1
        img = np.clip(128 + 40 * np.sin(t) + 15 * np.random.randn(h, w, 3), 0, 255).astype(np.uint8)
        return {
            "image": img,
            "imu": np.array([0, 0, 9.81, 0, 0, 0], dtype=np.float32),
            "lidar": np.random.randn(p["lidar_points"]).astype(np.float32),
            "audio": self._audio_features(self._synth_audio_wave(self._tick, p["audio_samples"])),
        }

    def _audio_payload(self, frame) -> np.ndarray:
        audio = frame.get("audio")
        if audio is None:
            return self._audio_features(self._synth_audio_wave(self._tick, self.profile["audio_samples"]))
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size > 128 * 128:
            return self._audio_features(arr)
        return arr

    def _publish_frame(self, frame, stamp):
        img = frame["image"]
        h, w = img.shape[:2]
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id=f"{self.vertical}_ingest")
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img.tobytes()
        self.pub_img.publish(msg)

        for topic_data, pub in (
            (frame["imu"], self.pub_imu),
            (frame["lidar"], self.pub_lidar),
            (self._audio_payload(frame), self.pub_audio),
        ):
            arr = Float32MultiArray()
            arr.data = np.asarray(topic_data, dtype=np.float32).tolist()
            pub.publish(arr)

    def destroy_node(self):
        if self._bag_proc:
            self._bag_proc.terminate()
        if self._camera is not None:
            self._camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SensorIngestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
