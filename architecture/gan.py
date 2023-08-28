import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # self.label_conditioned_generator = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
        #                                                  nn.Linear(embedding_dim, 16))

        # self.latent = nn.Sequential(nn.Linear(self.latent_dim, 512 * 4 * 4),
        #                             nn.LeakyReLU(0.2, inplace=True))

        # self.latent = nn.Linear(self.latent_dim, 512 * 4 * 4)

        # self.model = nn.Sequential(nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, 0, bias=False),  # 512x4x4 (from 100x1x1)
        #                            nn.BatchNorm2d(64 * 8),  # initially eps=0.8 and momentum=0.1
        #                            nn.ReLU(inplace=True),
        #                            nn.ConvTranspose2d(512, 64 * 8, 4, 2, 1, bias=False),  # 512x8x8
        #                            nn.BatchNorm2d(64 * 8),
        #                            nn.ReLU(inplace=True),
        #                            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),  # 256x16x16
        #                            nn.BatchNorm2d(64 * 4),
        #                            nn.ReLU(inplace=True),
        #                            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),  # 128x32x32
        #                            nn.BatchNorm2d(64 * 2),
        #                            nn.ReLU(inplace=True),
        #                            nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),  # 64x64x64
        #                            nn.BatchNorm2d(64 * 1),
        #                            nn.ReLU(inplace=True),
        #                            # nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        #                            # nn.BatchNorm2d(32),
        #                            # nn.ReLU(inplace=True),
        #                            nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),  # 3x128x128
        #                            nn.Tanh())

        self.model = nn.Sequential(nn.ConvTranspose2d(self.latent_dim, 512, 2, 1, 0, bias=False),  # 512x2x2 (from 100x1x1)
                                   nn.BatchNorm2d(64 * 8),  # initially eps=0.8 and momentum=0.1
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512, 64 * 4, 4, 2, 1, bias=False),  # 256x4x4
                                   nn.BatchNorm2d(64 * 4),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),  # 128x8x8
                                   nn.BatchNorm2d(64 * 2),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),  # 64x16x16
                                   nn.BatchNorm2d(64 * 1),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(64 * 1, 32, 4, 2, 1, bias=False),  # 32x32x32
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # 16x64x64
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),  # 3x128x128
                                   nn.Tanh())

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.02)

        self.model.apply(init_weights)

    def forward(self, noise_vector):
        # noise_vector, contitioning_image = input
        # label_output = self.label_conditioned_generator(label)
        # label_output = label_output.view(-1, 1, 4, 4)
        # latent_output = self.latent(noise_vector)
        # latent_output = latent_output.view(-1, 512, 4, 4)
        # concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(noise_vector.view(-1, self.latent_dim, 1, 1))
        # print(image.size())
        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.label_condition_disc =
        # nn.Sequential(nn.Embedding(n_classes, embedding_dim),
        #               nn.Linear(embedding_dim, 3 * 128 * 128))

        # self.model = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 64x64x64 (from 3x128x128)
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),  # 128x22x22
        #                            nn.BatchNorm2d(64 * 2),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),  # 256x8x8
        #                            nn.BatchNorm2d(64 * 4),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),  # 512x3x3
        #                            nn.BatchNorm2d(64 * 8),
        #                            nn.LeakyReLU(0.2, inplace=True),
        #                            nn.Conv2d(64 * 8, 1, 3, 1, 0, bias=False),  # 1x1x1
        #                            # nn.Flatten(),
        #                            # nn.Dropout(0.4),
        #                            # nn.Linear(4608, 1),
        #                            nn.Sigmoid()
        #                            )

        self.model = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1, bias=False),  # 16x64x64 (from 3x128x128)
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # 32x32x32
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64x16x16
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 1, 64 * 2, 4, 2, 1, bias=False),  # 128x8x8
                                   nn.BatchNorm2d(64 * 2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),  # 256x4x4
                                   nn.BatchNorm2d(64 * 4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),  # 512x2x2
                                   nn.BatchNorm2d(64 * 8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 8, 1, 2, 1, 0, bias=False),  # 1x1x1
                                   nn.Sigmoid()
                                   )

    def forward(self, img):
        # img, label = inputs
        # label_output = self.label_condition_disc(label)
        # label_output = label_output.view(-1, 3, 128, 128)
        # concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(img).view(-1, 1)
        return output