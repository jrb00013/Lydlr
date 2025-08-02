# lydlr_ai/model/compressor.py
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lydlr_ai.utils.voxel_utils import lidar_to_pointcloud

# --- Multi-Modal Encoders ---
class ImageEncoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # fixed for now
        self.fc = nn.Linear(32 * 4 * 4, 128)

    def forward(self, x):
        conv_out = self.conv(x)
        self._output_shape = conv_out.shape  # store shape for decoder
        pooled = self.pool(conv_out)
        return self.fc(pooled.view(x.size(0), -1))

    def get_conv_output_shape(self):
        return self._output_shape

class LiDAREncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, 128)

    def forward(self, x):  # x: (B, 1, H, W)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- Multimodal Fusion + Temporal LSTM ---

class MultimodalCompressor(nn.Module):
    def __init__(self, image_shape=(3,480,640), lidar_dim=1024, imu_dim=6, audio_dim=128*128):
        super().__init__()
        channels, height, width = image_shape
        self.image_encoder = ImageEncoder(channels, height, width)
        self.lidar_encoder = LiDAREncoder(lidar_dim)
        self.imu_encoder = IMUEncoder(imu_dim)
        self.audio_encoder = AudioEncoder(audio_dim)

        self.fusion_fc = nn.Linear(128 + 128 + 32 + 128, 256)

        # Temporal context via LSTM on fused features over time
        #self.lstm = nn.LSTM(256, 128, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Predictor for prediction head
        self.predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Decoder for reconstruction (simple linear for demo)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128 + 128 + 32 + 128)  # reconstruct fusion features
        )

        self.image_decoder_fc = nn.Linear(128, 32 * (image_shape[1] // 4) * (image_shape[2] // 4))
        
        self.image_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # Upsample H/4 -> H/2
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_shape[0], 4, stride=2, padding=1),  # Upsample H/2 -> H
            nn.Sigmoid()
        )

        self.vae_compress = VAE(feature_dim=256, latent_dim=64) 
    
    def fuse_modalities(self, image, lidar, imu, audio, compression_level=1.0):
        img_enc = self.image_encoder(image)
        lidar_xyz = lidar_to_pointcloud(lidar)  
        lidar_enc = self.lidar_encoder(lidar_xyz.view(lidar.size(0), -1))
        imu_enc = self.imu_encoder(imu)
        audio_enc = self.audio_encoder(audio)

        fused = torch.cat([img_enc, lidar_enc, imu_enc, audio_enc], dim=1)
        fused = self.fusion_fc(fused)
        fused = F.dropout(fused, p=1.0 - compression_level, training=self.training)

        # Pass through VAE for compression
        mu, logvar = self.vae_compress.encode(fused)
        z = self.vae_compress.reparameterize(mu, logvar)
        recon_fused = self.vae_compress.decode(z)

        return fused, z, recon_fused, mu, logvar

    def forward(self, image, lidar, imu, audio, hidden_state=None, compression_level=1.0):
        img_enc = self.image_encoder(image)
        lidar_xyz = lidar_to_pointcloud(lidar)  # shape: (B, N, 3)
        lidar_enc = self.lidar_encoder(lidar_xyz.view(lidar.size(0), -1))
        imu_enc = self.imu_encoder(imu)
        audio_enc = self.audio_encoder(audio)

        fused = torch.cat([img_enc, lidar_enc, imu_enc, audio_enc], dim=1)
        fused = self.fusion_fc(fused)  # Add seq dim for LSTM (B,1,256)
        fused = F.dropout(fused, p=1.0 - compression_level, training=self.training)

        # Run LSTM -> temporal context
        lstm_out, hidden_state = self.lstm(fused, hidden_state)  # lstm_out (B,1,128)

        decoded = self.decoder(lstm_out.squeeze(1))

         # --- Image reconstruction from latent for quality assessment ---
        batch_size = image.size(0)
        feat_shape = self.image_encoder.get_conv_output_shape()  # (B, C, H', W')
        feat_H, feat_W = feat_shape[2], feat_shape[3]
        img_feat_flat = self.image_decoder_fc(lstm_out.squeeze(1))
        img_feat = img_feat_flat.view(batch_size, 32, feat_H, feat_W)
        reconstructed_img = self.image_decoder_conv(img_feat)

        return lstm_out.squeeze(1), decoded, hidden_state, reconstructed_img

# --- Reinforcement Learning Stub for Dynamic Compression Controller ---

class CompressionPolicy:
    def __init__(self):
        self.compression_level = 1.0  # 1.0=full quality, <1.0 more compression
        self.reward = 0.0

    def estimate_battery(self):
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return 1.0  # assume plugged-in or desktop system
            return battery.percent / 100.0
        except Exception as e:
            print(f"[Battery Estimation Error]: {e}")
            return 1.0  # fallback


    def update_policy(self, compression_ratio, quality_dict):
        lpips = quality_dict['lpips']
        psnr = quality_dict['psnr']
        ssim = quality_dict['ssim']

        # Combine multiple metrics
        quality_score = (1 - lpips) * 0.5 + psnr / 50.0 * 0.25 + ssim * 0.25
        reward = quality_score / (compression_ratio + 1e-6)

        cpu = psutil.cpu_percent() / 100.0
        battery = self.estimate_battery()
        priority = self.priority_map.get(topic, 0.7)

        # Policy condition final score
        score = 0.3 * reward + 0.3 * (1 - cpu) + 0.2 * battery + 0.2 * priority

        # Adjust compression level based on the policy condition score
        if score > 0.6:
            self.compression_level = max(0.1, self.compression_level - 0.05)
        else:
            self.compression_level = min(1.0, self.compression_level + 0.05)


    def get_level(self):
        return self.compression_level

# --- Real-time Quality Assessment using LPIPS ---

class QualityAssessor:
    def __init__(self, device='cpu'):
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.device = device

    def assess(self, img1, img2):
        img1_np = img1.squeeze().permute(1,2,0).cpu().numpy()
        img2_np = img2.squeeze().permute(1,2,0).cpu().numpy()

        psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
        ssim = structural_similarity(img1_np, img2_np, multichannel=True)

        lpips_score = self.loss_fn((img1 * 2 - 1), (img2 * 2 - 1)).mean().item()
        return {
        "lpips": lpips_score,
        "psnr": psnr,
        "ssim": ssim
        }

# --- Quantization & Export Utility ---

def quantize_model(model):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    # Calibrate with dummy input or dataset here...
    torch.quantization.convert(model, inplace=True)
    return model

def export_torchscript(model, path="model_scripted.pt"):
    scripted = torch.jit.script(model)
    scripted.save(path)
    return path

# --- VAE and VQ-VAE stub models (for future training) ---

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, input_height=480, input_width=640):
        super().__init__()
        # Encoder: convert layers to latent mean and logvar

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # H/8, W/8
            nn.ReLU()
        )

        conv_output_size = 128 * (input_height // 8) * (input_width // 8)
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

        # Decoder: latent to feature map to conv transpose layers
        self.decoder_fc = nn.Linear(latent_dim, conv_output_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # H/4, W/4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1), # H, W
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, x.size(1) // 128 // (x.size(0) or 1), -1)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# --- Complete VQ-VAE ---

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B,C,H,W)
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        flat_inputs = flat_inputs.view(-1, self.embedding_dim)  # (B*H*W, C)

        # Compute distances
        distances = (flat_inputs.pow(2).sum(1, keepdim=True)
                     - 2 * flat_inputs @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.size())

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            *list(models.resnet18(pretrained=True).children())[:-2],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # B,64,H/2,W/2
        #     nn.ReLU(),
        #     nn.Conv2d(64, embedding_dim, 4, stride=2, padding=1),  # B,C,H/4,W/4
        #     nn.ReLU()
        # ) # may need soon

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.vq_layer(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, perplexity

# --- Minimal Normalizing Flow stub ---

class SimpleFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Just an affine coupling layer example
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Forward flow: z = scale * x + shift
        z = self.scale * x + self.shift
        log_det_jacobian = torch.sum(torch.log(torch.abs(self.scale) + 1e-6))
        return z, log_det_jacobian

    def inverse(self, z):
        x = (z - self.shift) / (self.scale + 1e-6)
        return x
