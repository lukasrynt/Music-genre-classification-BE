from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(400 * 16 * 2, 500),
            nn.ReLU(True),
            nn.Linear(500, 300)
        )

        self.decoder = nn.Sequential(
            nn.Linear(300, 500),
            nn.ReLU(True),
            nn.Linear(500, 400 * 16 * 2),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(4, 32, 100)),
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        compressed = x
        x = self.decoder(x)
        return x, compressed
