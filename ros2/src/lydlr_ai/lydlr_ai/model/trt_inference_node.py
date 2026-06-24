"""
TensorRT / ONNX edge inference node for Jetson deploy bundles.

Set INFERENCE_BACKEND=onnx|trt|torch (default: onnx if bundle present else torch).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from lydlr_ai.communication.topics import LydlrTopics
from lydlr_ai.communication.edge_transport import EdgeTransportLayer, sensor_qos
from lydlr_ai.communication import wire
from lydlr_ai.utils.metrics_reporter import report_metrics


class TrtInferenceNode(Node):
    def __init__(self):
        super().__init__("trt_inference_node")
        self.node_id = os.getenv("NODE_ID", "node_0")
        self.backend = os.getenv("INFERENCE_BACKEND", "onnx").lower()
        self.bundle_dir = Path(
            os.getenv(
                "LYDLR_DEPLOY_BUNDLE",
                f"deploy_bundles/jetson_{os.getenv('MODEL_VERSION', 'vv1.0')}",
            )
        )
        self._session = None
        self._torch_model = None
        self._load_backend()

        self.transport = EdgeTransportLayer(self, self.node_id)
        sqos = sensor_qos()
        self.create_subscription(Image, LydlrTopics.CAMERA, self._image_cb, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.LIDAR, self._lidar_cb, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.IMU, self._imu_cb, sqos)
        self.create_subscription(Float32MultiArray, LydlrTopics.AUDIO, self._audio_cb, sqos)

        self._latest = {}
        self.create_timer(0.1, self._infer_tick)
        self.get_logger().info(
            f"TRT inference node [{self.node_id}] backend={self.backend} bundle={self.bundle_dir}"
        )

    def _load_backend(self):
        manifest_path = self.bundle_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self.backend = manifest.get("inference_backend", self.backend)

        if self.backend == "trt":
            engine = self.bundle_dir / "multimodal_compressor.trt"
            if not engine.exists():
                self.get_logger().warn("TRT engine missing — falling back to ONNX")
                self.backend = "onnx"

        if self.backend == "onnx":
            import onnxruntime as ort

            onnx_path = self.bundle_dir / "multimodal_compressor.onnx"
            if not onnx_path.exists():
                raise SystemExit(f"ONNX bundle missing: {onnx_path}")
            providers = ["CPUExecutionProvider"]
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(onnx_path), providers=providers)
            return

        if self.backend == "torch":
            import torch
            from lydlr_ai.model.compressor import EnhancedMultimodalCompressor

            version = os.getenv("MODEL_VERSION", "v1.0").lstrip("v")
            model_dir = Path(os.getenv("MODEL_DIR", f"models/{self.node_id}"))
            for name in (f"lydlr_compressor_v{version}.pth", f"compressor_v{version}.pth"):
                weights = model_dir / name
                if weights.exists():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = EnhancedMultimodalCompressor().to(device)
                    ckpt = torch.load(weights, map_location=device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    self._torch_model = (model, device)
                    return
            raise SystemExit("No torch weights found for TRT inference fallback")

    def _image_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._latest["image"] = img.astype(np.float32) / 255.0

    def _lidar_cb(self, msg):
        self._latest["lidar"] = np.array(msg.data, dtype=np.float32)

    def _imu_cb(self, msg):
        self._latest["imu"] = np.array(msg.data, dtype=np.float32)

    def _audio_cb(self, msg):
        self._latest["audio"] = np.array(msg.data, dtype=np.float32)

    def _infer_tick(self):
        if "image" not in self._latest:
            return
        t0 = time.perf_counter()

        img = self._latest["image"]
        feeds = {
            "image": img.transpose(2, 0, 1)[None, ...].astype(np.float32),
            "lidar": np.resize(self._latest.get("lidar", np.zeros(1024 * 3)), (1, 1024 * 3)).astype(np.float32),
            "imu": np.resize(self._latest.get("imu", np.zeros(6)), (1, 6)).astype(np.float32),
            "audio": np.resize(self._latest.get("audio", np.zeros(128 * 128)), (1, 128 * 128)).astype(np.float32),
        }

        quality = 0.8
        if self.backend == "onnx" and self._session:
            outputs = self._session.run(None, feeds)
            if outputs:
                quality = float(np.mean(outputs[-1])) if np.ndim(outputs[-1]) else 0.8
            blob = outputs[0].tobytes() if outputs else b""
        elif self.backend == "torch" and self._torch_model:
            import torch

            model, device = self._torch_model
            with torch.no_grad():
                out = model(
                    torch.tensor(feeds["image"], device=device),
                    torch.tensor(feeds["lidar"], device=device),
                    torch.tensor(feeds["imu"], device=device),
                    torch.tensor(feeds["audio"], device=device),
                    None,
                    0.8,
                    0.8,
                )
            quality = float(out[-2].item()) if out[-2] is not None else 0.8
            blob = out[0].cpu().numpy().tobytes()
        else:
            return

        latency_ms = (time.perf_counter() - t0) * 1000
        bytes_in = sum(v.nbytes for v in feeds.values())
        bytes_out = len(blob)
        ratio = bytes_in / max(bytes_out, 1)

        self.transport.publish_compressed(blob, os.getenv("MODEL_VERSION", ""), bytes_in, ratio)
        metrics = wire.MetricsPayload(
            node_id=self.node_id,
            vertical=self.transport.vertical,
            model_version=os.getenv("MODEL_VERSION", ""),
            compression_ratio=ratio,
            latency_ms=latency_ms,
            compression_level=0.8,
            quality_score=quality,
            bandwidth_estimate=0.0,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            modality_bytes_in={"camera": feeds["image"].nbytes},
            modality_bytes_out={"camera": bytes_out},
            modality_quality={"camera": quality},
        )
        self.transport.publish_metrics(metrics)
        report_metrics(
            node_id=self.node_id,
            compression_ratio=ratio,
            latency_ms=latency_ms,
            quality_score=quality,
            bandwidth_estimate=0.0,
            compression_level=0.8,
            vertical=self.transport.vertical,
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrtInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
