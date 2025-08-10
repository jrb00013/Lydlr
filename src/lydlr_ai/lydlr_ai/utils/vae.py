# lydlr_ai/utils/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, feature_dim=512, latent_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder: fuse latent (mu/logvar)
        self.encoder_fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # outputs both mu and logvar stacked
        )

        # Decoder: latent reconstruct fused feature
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def encode(self, x):
        stats = self.encoder_fc(x)
        mu, logvar = stats[:, :self.latent_dim], stats[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_fc(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld
