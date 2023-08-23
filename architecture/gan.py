import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # self.label_conditioned_generator = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
        #                                                  nn.Linear(embedding_dim, 16))

        self.latent = nn.Sequential(nn.Linear(self.latent_dim, 512 * 4 * 4),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(nn.ConvTranspose2d(512, 64 * 8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())

    def forward(self, noise_vector):
        # noise_vector, contitioning_image = input
        # label_output = self.label_conditioned_generator(label)
        # label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        # concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(latent_output)
        # print(image.size())
        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.label_condition_disc =
        # nn.Sequential(nn.Embedding(n_classes, embedding_dim),
        #               nn.Linear(embedding_dim, 3 * 128 * 128))

        self.model = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(4608, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, img):
        # img, label = inputs
        # label_output = self.label_condition_disc(label)
        # label_output = label_output.view(-1, 3, 128, 128)
        # concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(img)
        return output