import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640*480, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 640*480),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

class AICompressor:
    def __init__(self):
        self.model = SimpleAutoencoder()
        self.model.eval()

    def compress(self, tensor):
        with torch.no_grad():
            encoded = self.model.encoder(tensor.float())
            return encoded.numpy()
