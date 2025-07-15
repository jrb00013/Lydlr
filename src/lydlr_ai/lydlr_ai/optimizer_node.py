import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import psutil 

from lydlr_ai.utils.voxel_utils import visualize_voxel_lidar
from lydlr_ai.model.compressor import MultimodalCompressor, CompressionPolicy, QualityAssessor
from lydlr_ai.model.transformer import PositionalEncoding

class StorageOptimizer(Node):
    def __init__(self):
        super().__init__('storage_optimizer')
        self.get_logger().info("StorageOptimizer node started...")

        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Float32, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Float32, '/lidar/data', self.lidar_callback, 10)
        self.create_subscription(Float32, '/audio/data', self.audio_callback, 10)

        self.compressor = None
        self.policy = CompressionPolicy()
        self.assessor = QualityAssessor()
        self.hidden_state = None

        # Adding a buffer
        self.input_seq = [] 
        self.seq_len = 4 # can be tuned if needed later

        # Initialize PositionalEncoder and transformer
        self.d_model = 256
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, max_len=50)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=4)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # Mel Spectograph
        self.mel = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)

        self.latest_inputs = {
            'image': None,
            'imu': None,
            'lidar': None,
            'audio': None
        }

    def camera_callback(self, msg):
        try:
            if msg.encoding == 'rgb8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img_np = img_np / 255.0
                img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                img_tensor = (img_tensor - 0.5) / 0.5
            elif msg.encoding == 'mono8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width) / 255.0
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                img_tensor = (img_tensor - 0.5) / 0.5
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return
            self.latest_inputs['image'] = img_tensor
            self.try_compress()
        except Exception as e:
            self.get_logger().error(f"Camera processing failed: {e}")

    def imu_callback(self, msg):
        imu_tensor = torch.tensor([[msg.data]*6], dtype=torch.float32)
        imu_tensor = (imu_tensor - imu_tensor.mean()) / (imu_tensor.std() + 1e-6)
        self.latest_inputs['imu'] = imu_tensor
        self.try_compress() # [ax, ay, az, gx, gy, gz]

    def lidar_callback(self, msg):
        lidar_tensor = torch.tensor([[msg.data]*1024], dtype=torch.float32)
        lidar_tensor = (lidar_tensor - lidar_tensor.mean()) / (lidar_tensor.std() + 1e-6)
        self.latest_inputs['lidar'] = lidar_tensor
        visualize_voxel_lidar(self.latest_inputs['lidar'])
        self.try_compress()

    def audio_callback(self, msg):
        waveform = torch.tensor(msg.data).view(1, -1)    # shape: (1, N)
        spec = self.mel(waveform)     # shape: (1, 64, T)
        spec = spec.log2().clamp(min=-20) / 20    # normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6) # normalize further
        self.latest_inputs['audio'] = spec.unsqueeze(0)     # (B, 1, 64, T)
        #self.latest_inputs['audio'] = torch.tensor([[msg.data]*16384], dtype=torch.float32)
        self.try_compress()
        

    def try_compress(self):
        if None in self.latest_inputs.values():
            return

        if self.compressor is None:
            img_shape = self.latest_inputs['image'].shape[1:]  # (C,H,W)
            self.compressor = MultimodalCompressor(
                image_shape=img_shape,
                lidar_dim=self.latest_inputs['lidar'].shape[1],
                imu_dim=self.latest_inputs['imu'].shape[1],
                audio_dim=self.latest_inputs['audio'].shape[1]
            )
            self.get_logger().info(f"Initialized compressor with inputs: {img_shape}")

        cpu_load = psutil.cpu_percent() / 100.0
        compression_level = self.policy.get_level()

        # --- Fused vector for current timestep ---
        fused = self.compressor.fuse_modalities(
            self.latest_inputs['image'],
            self.latest_inputs['lidar'],
            self.latest_inputs['imu'],
            self.latest_inputs['audio'],
            compression_level=compression_level
        )
        self.input_seq.append(fused.unsqueeze(1))  # (B, 1, 256)

        # --- Wait for full sequence ---
        if len(self.input_seq) < self.seq_len:
            return

        sequence = torch.cat(self.input_seq[-self.seq_len:], dim=1)  # (B, seq_len, 256)

        # --- Temporal compression via LSTM ---
        #lstm_out, self.hidden_state = self.compressor.lstm(sequence, self.hidden_state)
        
        sequence = sequence.permute(1, 0, 2)  # (seq_len, B, 256)
        sequence = self.pos_encoder(sequence)
        transformer_out = self.transformer(sequence)  # (seq_len, B, 256)
        
        # Decode & reconstruct from latest timestep output
        compressed_latent = transformer_out[-1] # Take last frame
        decoded = self.compressor.decoder(compressed_latent)

        # Reconstruct image for quality check
        img_feat_flat = self.compressor.image_decoder_fc(compressed_latent)
        batch_size = self.latest_inputs['image'].size(0)
        feat_map_H = self.latest_inputs['image'].size(2) // 4
        feat_map_W = self.latest_inputs['image'].size(3) // 4
        img_feat = img_feat_flat.view(batch_size, 32, feat_map_H, feat_map_W)
        reconstructed_img = self.compressor.image_decoder_conv(img_feat)

        # --- Quality Metrics ---
        quality = self.assessor.assess(
            self.latest_inputs['image'], reconstructed_img)
        self.get_logger().info(
            f"Quality - LPIPS: {quality['lpips']:.4f}, PSNR: {quality['psnr']:.2f}, SSIM: {quality['ssim']:.4f}")

        input_size = self.latest_inputs['image'].numel() * 4  # bytes
        compressed_size = compressed_latent.numel() * 4       # bytes
        compression_ratio = input_size / compressed_size

        self.policy.update_policy(compression_ratio, quality)

        self.get_logger().info(
            f"Compression ratio: {compression_ratio:.3f}, Compression level: {compression_level:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = StorageOptimizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
