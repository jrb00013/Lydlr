import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, input_height=64, input_width=64):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Encoder: Conv layers -> FC layers to get mu and logvar
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # H/8, W/8
            nn.ReLU()
        )

        # Calculate conv output size for flattening
        conv_output_h = input_height // 8
        conv_output_w = input_width // 8
        self.conv_output_size = 128 * conv_output_h * conv_output_w

        # Latent vectors mu and logvar
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        # Decoder: FC then ConvTranspose to reconstruct image
        self.decoder_fc = nn.Linear(latent_dim, self.conv_output_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # H/2, W/2
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # H, W
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        # reshape to conv feature map size
        batch_size = z.size(0)
        h = self.decoder_conv[0].weight.size(1)  # 128 channels for first decoder conv transpose input
        conv_h = int((self.conv_output_size // 128) ** 0.5)  # Assuming square spatial dims
        x = x.view(batch_size, 128, conv_h, conv_h)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld