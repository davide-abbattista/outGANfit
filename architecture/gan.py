import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, depth_in, n_filters, filter_size=4, stride=2, padding=1, batchnorm=True):
        super(Encoder, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, filter_size, stride, padding, bias=False),
                                       nn.BatchNorm2d(n_filters),
                                       nn.LeakyReLU(0.2, inplace=True))
        else:
            self.model = nn.Sequential(nn.Conv2d(depth_in, n_filters, 4, 2, 1, bias=False),
                                       nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.model(input)


class Decoder(nn.Module):
    def __init__(self, depth_in, n_filters, filter_size=4, stride=2, padding=1):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(depth_in, n_filters, filter_size, stride, padding, bias=False),
                                   nn.BatchNorm2d(n_filters),
                                   nn.ReLU(inplace=True))

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):

    def __init__(self, autoencoder=True):
        super(Generator, self).__init__()
        self.autoencoder = autoencoder

        if not autoencoder:
            # encoding blocks
            self.e1 = Encoder(3, 16, batchnorm=False)  # 16x64x64 (input image: 3x128x128)
            self.e2 = Encoder(16, 32)  # 32x32x32
            self.e3 = Encoder(32, 64)  # 64x16x16
            self.e4 = Encoder(64, 128)  # 128x8x8
            self.e5 = Encoder(128, 256)  # 256x4x4
            self.e6 = Encoder(256, 512)  # 512x2x2

            # bottleneck
            self.b = Encoder(512, 512, 2, 1, 0)  # 512x1x1

        # decoding blocks
        self.d1 = Decoder(512, 512, 2, 1, 0)  # 512x2x2
        self.d2 = Decoder(512, 256)  # 256x4x4
        self.d3 = Decoder(256, 128)  # 128x8x8
        self.d4 = Decoder(128, 64)  # 64x16x16
        self.d5 = Decoder(64, 32)  # 32x32x32
        self.d6 = Decoder(32, 16)  # 16x64x64

        # output
        self.o = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)  # output image: 3x128x128
        self.t = nn.Tanh()

    def forward(self, cond):
        if not self.autoencoder:
            return self.t(self.o(
                self.d6(self.d5(self.d4(self.d3(self.d2(self.d1(
                    self.b(
                        self.e6(self.e5(self.e4(self.e3(self.e2(self.e1(cond))))))
                    )
                ))))))
            ))
        else:
            return self.t(self.o(
                self.d6(self.d5(self.d4(self.d3(self.d2(self.d1(cond))))))
            ))


class Discriminator(nn.Module):
    def __init__(self, depth_in):
        super(Discriminator, self).__init__()
        self.depth_in = depth_in

        # encoding blocks
        self.e1 = Encoder(self.depth_in, 16, batchnorm=False)  # 16x64x64 (input image: 3x128x128 or 6x128x128)
        self.e2 = Encoder(16, 32)  # 32x32x32
        self.e3 = Encoder(32, 64)  # 64x16x16
        self.e4 = Encoder(64, 128)  # 128x8x8
        self.e5 = Encoder(128, 256)  # 256x4x4
        self.e6 = Encoder(256, 512)  # 512x2x2

        # output
        self.o = nn.Conv2d(512, 1, 2, 1, 0, bias=False)  # 1x1x1
        self.s = nn.Sigmoid()

    def forward(self, img, cond=None):
        if cond is not None:
            # concatenate along the channels dimension the real or generated image with the conditioning one
            img = torch.cat((img, cond), 1)

        output = self.s(self.o(
            self.e6(self.e5(self.e4(self.e3(self.e2(self.e1(img))))))
        )).view(-1, 1)

        return output
