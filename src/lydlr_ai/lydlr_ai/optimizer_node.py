# optimizer_node.py
import psutil 
import time
import os
import zlib
import threading
import queue
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from lydlr_ai.utils.voxel_utils import visualize_voxel_lidar
from lydlr_ai.model.compressor import MultimodalCompressor, CompressionPolicy, QualityAssessor
from lydlr_ai.model.transformer import PositionalEncoding
from lydlr_ai.utils.vae import VAE, vae_loss
from lydlr_ai.model.quality_predictor import QualityPredictor
from lydlr_ai.utils.lyd_format import save_lyd, load_lyd_progressive

import matplotlib.pyplot as plt
plt.ion()  

class StorageOptimizer(Node):
    def __init__(self):
        super().__init__('storage_optimizer')
        self.get_logger().info("StorageOptimizer node started...")

        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Float32, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Float32, '/lidar/data', self.lidar_callback, 10)
        self.create_subscription(Float32MultiArray, '/audio/data', self.audio_callback, 10)

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
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # Vae Initialization
        # Image transforms: resize + normalization as expected by pretrained ResNet
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Load pretrained ResNet18 backbone, remove last fc layer
        backbone = models.resnet18(pretrained=True)
        modules = list(backbone.children())[:-1]  # Remove the final FC layer
        self.feature_extractor = torch.nn.Sequential(*modules)
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # freeze backbone
        
        # Device initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)

        # VAE input feature_dim should match backbone output (512 for ResNet18)
        self.vae = VAE(feature_dim=512, latent_dim=128).to(self.device)

        # Mel Spectograph
        self.mel = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)

        # Quality Predictor Initialization
        self.quality_predictor = QualityPredictor().to(self.device)
        self.quality_predictor.to(self.device)
        
        # Tracking previous fused state
        self.previous_fused = None

        self.latest_inputs = {
            'image': None,
            'imu': None,
            'lidar': None,
            'audio': None
        }
        
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize Thread-safe queue and start the worker thread
        self.save_queue = queue.Queue()
        threading.Thread(target=self._save_worker, daemon=True).start()

    def camera_callback(self, msg):
        self.get_logger().info("Received image message")
        try:
            if msg.encoding == 'rgb8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img_np = img_np.astype(np.float32) / 255.0
                img_tensor = torch.tensor(img_np).permute(2,0,1)  # (3,H,W)
            elif msg.encoding == 'mono8':
                img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width) / 255.0
                img_tensor = torch.tensor(img_np).unsqueeze(0).repeat(3,1,1)  # (3,H,W)
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return

            # Resize to 224x224 as required by ResNet
            img_tensor = transforms.functional.resize(img_tensor, (224, 224))

            # Normalize using ImageNet stats
            img_tensor = self.img_transform(img_tensor)

            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1,3,224,224)

            with torch.no_grad():
                features = self.feature_extractor(img_tensor)  # (1, 512, 1, 1)
                features = features.view(features.size(0), -1)  # flatten to (1, 512)

            self.latest_inputs['image'] = features  # pass the 512-dim feature vector
            self.try_compress()
        except Exception as e:
            self.get_logger().error(f"Camera processing failed: {e}")
            
    def imu_callback(self, msg):
        self.get_logger().info("📈 Received IMU message")
        imu_tensor = torch.tensor([[msg.data]*6], dtype=torch.float32)
        imu_tensor = (imu_tensor - imu_tensor.mean()) / (imu_tensor.std() + 1e-6)
        self.latest_inputs['imu'] = imu_tensor.to(self.device)
        self.try_compress() # [ax, ay, az, gx, gy, gz]

    def lidar_callback(self, msg):
        self.get_logger().info("🛞 Received LiDAR message")
        lidar_tensor = torch.tensor([[msg.data]*1024], dtype=torch.float32)
        lidar_tensor = (lidar_tensor - lidar_tensor.mean()) / (lidar_tensor.std() + 1e-6)
        self.latest_inputs['lidar'] = lidar_tensor.to(self.device)
        visualize_voxel_lidar(self.latest_inputs['lidar'])
        self.try_compress()

    def audio_callback(self, msg: Float32MultiArray):
        self.get_logger().info("🎤 Received audio message")
        waveform = torch.tensor(msg.data).view(1, -1)    # shape: (1, N)
        spec = self.mel(waveform)     # shape: (1, 64, T)
        spec = spec.log2().clamp(min=-20) / 20    # normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6) # normalize further
        self.latest_inputs['audio'] = spec.unsqueeze(0).to(self.device)    # (B, 1, 64, T)
        #self.latest_inputs['audio'] = torch.tensor([[msg.data]*16384], dtype=torch.float32)
        self.try_compress()
    
    def _save_worker(self):
        while True:
            try:
                chunks, lpips, mask, timestamp = self.save_queue.get()
                save_path = os.path.join(self.data_dir, f"frame_{int(timestamp)}.lyd")
                save_lyd(save_path, chunks, lpips, mask, timestamp)
                self.get_logger().info(f"Async saved to {save_path}")
            except Exception as e:
                self.get_logger().error(f"Save worker error: {e}")

    def try_compress(self):
        if None in self.latest_inputs.values():
            return
        
        # Initialize VAE 
        if self.vae is None:
            img_shape = self.latest_inputs['image'].shape[1:]  # (C,H,W)
            _, C, H, W = (1,) + img_shape
            self.vae = VAE(feature_dim=256, latent_dim=128).to(self.device)

        # Run VAE on image only get reconstruction and loss
        self.vae.eval()
        with torch.no_grad():
            recon, mu, logvar = self.vae(self.latest_inputs['image'])
            loss = vae_loss(recon, self.latest_inputs['image'], mu, logvar)

        self.get_logger().info(f"VAE Reconstruction loss: {loss.item():.4f}")

        # Initialize multimodal compressor
        if self.compressor is None:
            img_shape = self.latest_inputs['image'].shape[1:]  # (C,H,W)
            self.compressor = MultimodalCompressor(
                image_shape=img_shape,
                lidar_dim=self.latest_inputs['lidar'].shape[1],
                imu_dim=self.latest_inputs['imu'].shape[1],
                audio_dim=self.latest_inputs['audio'].shape[1]
            ).to(self.device)
            
            self.get_logger().info(f"Initialized compressor with inputs: {img_shape}")

        cpu_load = psutil.cpu_percent() / 100.0
        compression_level = self.policy.get_level()

        # --- Fused vector for current timestep ---
        with torch.no_grad():
            fused, z, recon_fused, mu, logvar = self.compressor.fuse_modalities(
                self.latest_inputs['image'],
                self.latest_inputs['lidar'],
                self.latest_inputs['imu'],
                self.latest_inputs['audio'],
                compression_level
            )
            loss_vae = vae_loss(recon_fused, fused, mu, logvar)
        self.get_logger().info(f"Fused‑VAE loss: {loss_vae.item():.4f}")
        
          # Decode & reconstruct from latest timestep output
        compressed_latent = z

        if self.previous_fused is None:
            delta_fused = fused
        else:
            delta_fused = fused - self.previous_fused

        self.previous_fused = fused.detach()
        self.input_seq.append(delta_fused.unsqueeze(1))  # (B, 1, 256)

        # Predict X_t from X_{t-1}
        if len(self.input_seq) >= 2:
            pred = self.compressor.predictor(self.input_seq[-2].squeeze(1))  # predict current from previous
            pred_loss = F.mse_loss(pred, fused.detach())
        else:
            pred_loss = torch.tensor(0.0)

        # --- Wait for full sequence ---
        if len(self.input_seq) < self.seq_len:
            return

        sequence = torch.cat(self.input_seq[-self.seq_len:], dim=1)  # (B, seq_len, 256)

        # --- Temporal compression via LSTM ---
        #lstm_out, self.hidden_state = self.compressor.lstm(sequence, self.hidden_state)
        
        sequence = sequence.permute(1, 0, 2)  # (seq_len, B, 256)
        sequence = self.pos_encoder(sequence)
        transformer_out = self.transformer(sequence)  # (seq_len, B, 256)
        
        # Split compressed latent into 4 chunks for progressive decoding/storage
        #compressed_latent = transformer_out[-1] # Take the last frame man
        chunks = torch.chunk(compressed_latent, chunks=4, dim=1)  # split latent ino 4 parts along feature dim
        compressed_chunks = [zlib.compress(c.cpu().numpy().tobytes()) for c in chunks]
        decoded = self.compressor.decoder(compressed_latent)
    
        # Reconstruct image for quality check
        img_feat_flat = self.compressor.image_decoder_fc(compressed_latent)
        batch_size = self.latest_inputs['image'].size(0)
        feat_map_H = self.latest_inputs['image'].size(2) // 4
        feat_map_W = self.latest_inputs['image'].size(3) // 4
        img_feat = img_feat_flat.view(batch_size, 32, feat_map_H, feat_map_W)
        reconstructed_img = self.compressor.image_decoder_conv(img_feat)

        # --- LPIPS Quality Metrics ---
        quality = self.assessor.assess(
            self.latest_inputs['image'], reconstructed_img)
        self.get_logger().info(
            f"Quality - LPIPS: {quality['lpips']:.4f}, PSNR: {quality['psnr']:.2f}, SSIM: {quality['ssim']:.4f}")

        with torch.no_grad():
            predicted_quality = self.quality_predictor(compressed_latent)
            pred_lpips, pred_psnr, pred_ssim = predicted_quality[0].tolist()
        self.get_logger().info(
         f"Predicted Quality - LPIPS: {pred_lpips:.4f}, PSNR: {pred_psnr:.2f}, SSIM: {pred_ssim:.4f}")

        # Weighted sum of losses; VAE + predictive + LPIPS
        w_vae = 1.0
        w_pred = 0.5
        w_lpips = 2.0

        total_loss = (
            w_vae * loss +
            w_pred * pred_loss +
            w_lpips * quality["lpips"]
        )

        input_size = self.latest_inputs['image'].numel() * 4  # bytes
        compressed_size = compressed_latent.numel() * 4       # bytes
        compression_ratio = input_size / compressed_size

        self.policy.update_policy(compression_ratio, quality)

        self.get_logger().info(f"Predictive Loss: {pred_loss.item():.4f}")
        self.get_logger().info(f"Total Compression Loss: {total_loss.item():.4f}")
        self.get_logger().info(
            f"Compression ratio: {compression_ratio:.3f}, Compression level: {compression_level:.2f}")
        
        modality_mask = 0b1111  # all modalities used
        timestamp = time.time()
        save_path = os.path.join(self.data_dir, f"frame_{int(timestamp)}.lyd")

        # Save the Queue with the chunks, LPIPS quality, modality mask, and timestamp
        self.save_queue.put((compressed_chunks, quality['lpips'], modality_mask, timestamp))
        
        # Real-time progressive visualization every 5 seconds
        if int(timestamp) % 5 == 0:
           self.progressive_visualize(save_path, max_chunks=1)

def progressive_visualize(self, path, max_chunks=1):
    try:
        chunks, _, _, _ = load_lyd_progressive(path, max_chunks=max_chunks)

        latent_parts = []
        for chunk in chunks:
            decompressed = zlib.decompress(chunk)
            arr = np.frombuffer(decompressed, dtype=np.float32)
            latent_parts.append(torch.tensor(arr))

        latent_coarse = torch.cat(latent_parts, dim=-1).unsqueeze(0)  # (B, latent_dim)

        with torch.no_grad():
            # Use your existing image decoder
            img_feat_flat = self.compressor.image_decoder_fc(latent_coarse)
            batch_size = 1
            feat_map_H = self.latest_inputs['image'].size(2) // 4
            feat_map_W = self.latest_inputs['image'].size(3) // 4
            img_feat = img_feat_flat.view(batch_size, 32, feat_map_H, feat_map_W)
            img = self.compressor.image_decoder_conv(img_feat)

        # Convert image to NumPy and visualize
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # de-normalize

        plt.imshow(img_np)
        plt.title(f"Progressive Decoding: {max_chunks}/4 chunks")
        plt.axis('off')
        plt.pause(0.001)  # for real-time display

    except Exception as e:
        self.get_logger().error(f"Progressive visualization failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = StorageOptimizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
