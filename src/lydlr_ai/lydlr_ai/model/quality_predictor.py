import torch
import torch.nn as nn

class QualityPredictor(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # outputs: LPIPS, PSNR, SSIM
        )

    def forward(self, x):  # x: (B, 256)
        return self.net(x)
