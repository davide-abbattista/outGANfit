from utility.pytorch_msssim import SSIM
import torch
import torch.nn as nn
from utility.gdn import GDN


# Autoencoder model based on the paper https://arxiv.org/pdf/1611.01704.pdf with a few modifications to the architecture
class AutoEncoder(nn.Module):
    def __init__(self, C=512, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return out, code


class Encoder(nn.Module):
    def __init__(self, C=512, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(  # input image: 3x128x128
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x64x64
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x32x32
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x16x16
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),  # 128x8x8
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=2 * M, kernel_size=5, stride=2, padding=2, bias=False),  # 256x4x4
            GDN(2 * M),
            nn.Conv2d(in_channels=2 * M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False),  # 512X2X2
            nn.Flatten(start_dim=1),
            nn.Linear(C * 2 * 2, C)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, C=512, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(C, C * 2 * 2),
            nn.Unflatten(1, (C, 2, 2)),  # 512x2x2
            nn.ConvTranspose2d(in_channels=C, out_channels=2 * M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # 256x4x4
            GDN(2 * M, inverse=True),
            nn.ConvTranspose2d(in_channels=2 * M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # 128x8x8
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # 128x16x16
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # 128x32x32
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # 128x64x64
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),  # output image: 3x128x128
        )

    def forward(self, q):
        return torch.sigmoid(self.dec(q))
