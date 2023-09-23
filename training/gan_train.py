import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from architecture.gan_with_embeddings import Discriminator as Discriminator_AE
from architecture.gan_with_embeddings import Generator as Generator_AE
from architecture.gan_without_embeddings import Discriminator as Discriminator_no_AE
from architecture.gan_without_embeddings import Generator as Generator_no_AE
from utility.utils import read_json, weights_init
from utility.custom_image_dataset import CustomImageDatasetGAN
from utility.fid import FID


class GenerativeAdversarialNetworkTrainer:
    def __init__(self, train_set_path, validation_set_path, test_set_path, autoencoder=True, images_dir='../images'):
        self.fid = None
        self.criterion = None
        self.g_optimizer = None
        self.d_c_optimizer = None
        self.d_rf_optimizer = None
        self.compatibility_discriminator = None
        self.real_fake_discriminator = None
        self.generator = None
        self.device = None
        self.testloader = None
        self.validationloader = None
        self.trainloader = None
        self.autoencoder = autoencoder
        self.setup(train_set_path, validation_set_path, test_set_path, images_dir)

    def setup(self, train_set_path, validation_set_path, test_set_path, images_dir):
        train_set = read_json(train_set_path)
        validation_set = read_json(validation_set_path)
        test_set = read_json(test_set_path)

        transform = transforms.Compose([transforms.Resize(128),
                                        transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
                                        # range [-1, 1]
                                        # transforms.Normalize([0, 0, 0], [255, 255, 255])  # map values in the range [0, 1]
                                        ])

        trainset = CustomImageDatasetGAN(img_dir=images_dir, data=train_set, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

        validationset = CustomImageDatasetGAN(img_dir=images_dir, data=validation_set, transform=transform)
        self.validationloader = torch.utils.data.DataLoader(validationset, batch_size=128, shuffle=False)

        testset = CustomImageDatasetGAN(img_dir=images_dir, data=test_set, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.autoencoder:
            self.generator = Generator_AE().to(self.device)
            self.real_fake_discriminator = Discriminator_AE(depth_in=3).to(self.device)
            self.compatibility_discriminator = Discriminator_AE(depth_in=6).to(self.device)
        else:
            self.generator = Generator_no_AE().to(self.device)
            self.real_fake_discriminator = Discriminator_no_AE(depth_in=3).to(self.device)
            self.compatibility_discriminator = Discriminator_no_AE(depth_in=6).to(self.device)

        self.generator.apply(weights_init)
        self.real_fake_discriminator.apply(weights_init)
        self.compatibility_discriminator.apply(weights_init)

        self.d_rf_optimizer = torch.optim.Adam(self.real_fake_discriminator.parameters(), lr=2e-4,
                                               betas=(0.5, 0.999))
        self.d_c_optimizer = torch.optim.Adam(self.compatibility_discriminator.parameters(), lr=2e-4,
                                              betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = torch.nn.BCELoss()
        self.fid = FID()

    def generator_train_step(self, cond_images, batch_size):
        self.g_optimizer.zero_grad()

        fake_images = self.generator(cond_images)

        validity = self.real_fake_discriminator(fake_images)
        g1_loss = self.criterion(validity, torch.ones(batch_size, 1).to(self.device))
        # g1_loss.backward(retain_graph=True)

        compatibility = self.compatibility_discriminator(fake_images, cond_images)
        g2_loss = self.criterion(compatibility, torch.ones(batch_size, 1).to(self.device))
        # g2_loss.backward()

        # g_loss = (g1_loss + g2_loss) * 0.5
        g_loss = g1_loss + g2_loss
        g_loss.backward()

        self.g_optimizer.step()
        return g_loss.data.item()

    def compatibility_discriminator_train_step(self, cond_images, real_images, not_compatible_images, batch_size):
        self.d_c_optimizer.zero_grad()

        # train with compatible items
        real_validity = self.compatibility_discriminator(real_images, cond_images)
        compatibility_loss = self.criterion(real_validity, torch.ones(batch_size, 1).to(self.device))
        # compatibility_loss.backward()

        # train with fake images
        fake_images = self.generator(cond_images)
        fake_validity = self.compatibility_discriminator(fake_images, cond_images)
        fake_compatibility_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # fake_compatibility_loss.backward()

        # train with not compatible items
        fake_validity = self.compatibility_discriminator(not_compatible_images, cond_images)
        not_compatibility_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # not_compatibility_loss.backward()

        d_loss = compatibility_loss + fake_compatibility_loss + not_compatibility_loss
        d_loss.backward()

        self.d_c_optimizer.step()
        return d_loss.data.item()

    def real_fake_discriminator_train_step(self, cond_images, real_images, batch_size):
        self.d_rf_optimizer.zero_grad()

        # train with real images
        real_validity = self.real_fake_discriminator(real_images)
        real_loss = self.criterion(real_validity, torch.ones(batch_size, 1).to(self.device))
        # real_loss.backward()

        # train with fake images
        fake_images = self.generator(cond_images)
        fake_validity = self.real_fake_discriminator(fake_images)
        fake_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # fake_loss.backward()

        d_loss = real_loss + fake_loss
        d_loss.backward()

        self.d_rf_optimizer.step()
        return d_loss.data.item()

    def train(self, category, num_epochs=300):
        best_fid_score = float('inf')  # Initialize with a high value
        best_epoch = 0

        train_fids = []
        validation_fids = []

        for epoch in range(1, num_epochs + 1):
            print('\nStarting epoch {}...'.format(epoch))
            train_d_rf_loss = 0.0
            train_d_c_loss = 0.0
            train_g_loss = 0.0
            train_fid = 0.0

            for images, _ in self.trainloader:
                cond_images = images[0].to(self.device)
                real_images = images[1].to(self.device)
                not_compatible_images = images[2].to(self.device)

                batch_size = real_images.size(0)

                self.generator.train()
                self.compatibility_discriminator.train()
                self.real_fake_discriminator.train()

                d_rf_loss = self.real_fake_discriminator_train_step(cond_images, real_images, batch_size)
                d_c_loss = self.compatibility_discriminator_train_step(cond_images, real_images, not_compatible_images,
                                                                       batch_size)
                g_loss = self.generator_train_step(cond_images, batch_size)

                train_d_rf_loss += d_rf_loss * batch_size
                train_d_c_loss += d_c_loss * batch_size
                train_g_loss = g_loss * batch_size

                self.generator.eval()
                train_fid += self.fid.calculate_fid(real_images, self.generator(cond_images)) * batch_size

            train_d_rf_loss = train_d_rf_loss / len(self.trainloader.dataset)
            train_d_c_loss = train_d_c_loss / len(self.trainloader.dataset)
            train_g_loss = train_g_loss / len(self.trainloader.dataset)

            train_fid = train_fid / len(self.trainloader.dataset)
            train_fids.append(train_fid)

            print(f'{category} train losses')
            print('g_loss: {}, d_rf_loss: {}, d_c_loss: {}'.format(train_g_loss, train_d_rf_loss, train_d_c_loss))
            print('FID Score: {}'.format(train_fid))

            # EVALUATION

            self.generator.eval()
            validation_fid = 0.0

            for images, _ in self.validationloader:
                cond_images = images[0].to(self.device)
                real_images = images[1].to(self.device)

                batch_size = real_images.size(0)

                validation_fid += self.fid.calculate_fid(real_images, self.generator(cond_images)) * batch_size

            validation_fid = validation_fid / len(self.validationloader.dataset)
            validation_fids.append(validation_fid)

            print(f'{category} validation loss')
            print('FID Score: {}'.format(validation_fid))
            if validation_fid < best_fid_score:
                best_epoch = epoch
                print('Validation FID decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    best_fid_score,
                    validation_fid))
                best_fid_score = validation_fid
                torch.save(self.generator.state_dict(), f'{category}_generator.pth')

            images = next(iter(self.validationloader))
            cond_images = images[0].to(self.device)
            real_images = images[1].to(self.device)
            generated_images = self.generator(cond_images)

            top = [el.detach().cpu().numpy() for el in cond_images]
            compatible = [el.detach().cpu().numpy() for el in real_images]
            generated = [el.detach().cpu().numpy() for el in generated_images]

            fig = plt.figure(figsize=(10, 10))
            for idx in np.arange(2):
                ax1 = fig.add_subplot(3, 3, 3 * idx + 1, xticks=[], yticks=[])
                img1 = (top[idx] * 127.5 + 127.5).astype(int)
                plt.imshow(np.transpose(img1, (1, 2, 0)))
                ax1.set_title(f"Input top epoch {epoch}")

                ax2 = fig.add_subplot(3, 3, 3 * idx + 2, xticks=[], yticks=[])
                img2 = (compatible[idx] * 127.5 + 127.5).astype(int)
                plt.imshow(np.transpose(img2, (1, 2, 0)))
                ax2.set_title(f"Real compatible {category} epoch {epoch}")

                ax3 = fig.add_subplot(3, 3, 3 * idx + 3, xticks=[], yticks=[])
                img3 = (generated[idx] * 127.5 + 127.5).astype(int)
                plt.imshow(np.transpose(img3, (1, 2, 0)))
                ax3.set_title(f"Generated {category} epoch {epoch}")
            plt.show()

            print('\nBest epoch: ', best_epoch)
            print('Best FID: ', best_fid_score)
