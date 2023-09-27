import csv
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from architecture.gan_with_embeddings import Discriminator as Discriminator_AE
from architecture.gan_with_embeddings import Generator as Generator_AE
from architecture.gan_without_embeddings import Discriminator as Discriminator_no_AE
from architecture.gan_without_embeddings import Generator as Generator_no_AE
from architecture.ae import AutoEncoder
from utility.utils import read_json, gan_weights_init
from utility.custom_image_dataset import CustomImageDatasetGAN
from utility.fid import FID


class GenerativeAdversarialNetworkTrainer:
    def __init__(self, train_set_path, validation_set_path, test_set_path, category, autoencoder=True,
                 images_dir='../images',
                 autoencoder_checkpoint_path='../checkpoints/trained_ae_512.pth'):
        self.validation_fids = None
        self.train_fids = None
        self.test_fid = None
        self.best_generator = None
        self.best_epoch = None
        self.best_fid_score = None

        self.fid = None
        self.criterion = None
        self.g_optimizer = None
        self.d_c_optimizer = None
        self.d_rf_optimizer = None
        self.compatibility_discriminator = None
        self.real_fake_discriminator = None
        self.generator = None
        self.ae = None
        self.device = None
        self.testloader = None
        self.validationloader = None
        self.trainloader = None

        self.autoencoder = autoencoder
        self.category = category

        self.setup(train_set_path, validation_set_path, test_set_path, images_dir, autoencoder_checkpoint_path)

    def setup(self, train_set_path, validation_set_path, test_set_path, images_dir, autoencoder_checkpoint_path):
        train_set = read_json(train_set_path)
        validation_set = read_json(validation_set_path)
        test_set = read_json(test_set_path)

        transform = transforms.Compose([transforms.Resize(128),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                        # # map values in the range [-1, 1]
                                        # transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
                                        # # map values in the range [0, 1]
                                        # transforms.Normalize([0, 0, 0], [255, 255, 255])
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
            self.ae = AutoEncoder().to(self.device)
            checkpoint = torch.load(autoencoder_checkpoint_path, map_location=torch.device('cpu'))
            self.ae.load_state_dict(checkpoint)
            self.ae.eval()
        else:
            self.generator = Generator_no_AE().to(self.device)
            self.real_fake_discriminator = Discriminator_no_AE(depth_in=3).to(self.device)
            self.compatibility_discriminator = Discriminator_no_AE(depth_in=6).to(self.device)

        self.generator.apply(gan_weights_init)
        self.real_fake_discriminator.apply(gan_weights_init)
        self.compatibility_discriminator.apply(gan_weights_init)

        self.d_rf_optimizer = torch.optim.Adam(self.real_fake_discriminator.parameters(), lr=2e-4,
                                               betas=(0.5, 0.999))
        self.d_c_optimizer = torch.optim.Adam(self.compatibility_discriminator.parameters(), lr=2e-4,
                                              betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = torch.nn.BCELoss()
        self.fid = FID()

    def train_and_test(self, num_epochs=300):
        self.best_fid_score = float('inf')  # Initialize with a high value
        self.best_epoch = 0
        self.best_generator = deepcopy(self.generator)

        self.train_fids, self.validation_fids = [], []

        for epoch in range(1, num_epochs + 1):
            print('\nStarting epoch {}...'.format(epoch))
            # keep track of training losses and train and validation fid
            train_d_rf_loss = 0.0
            train_d_c_loss = 0.0
            train_g_loss = 0.0

            train_fid = 0.0
            validation_fid = 0.0

            ###################
            # train the model #
            ###################
            for images, _ in self.trainloader:
                # move tensors to GPU if CUDA is available
                cond_images = images[0].to(self.device)
                real_images = images[1].to(self.device)
                not_compatible_images = images[2].to(self.device)

                batch_size = real_images.size(0)

                self.generator.train()
                self.compatibility_discriminator.train()
                self.real_fake_discriminator.train()

                if self.autoencoder:
                    _, cond_images_embeddings = self.ae(cond_images)
                    d_rf_loss = self.real_fake_discriminator_train_step(cond_images, real_images, batch_size,
                                                                        cond_images_embeddings)
                    d_c_loss = self.compatibility_discriminator_train_step(cond_images, real_images,
                                                                           not_compatible_images, batch_size,
                                                                           cond_images_embeddings)
                    g_loss = self.generator_train_step(cond_images, batch_size, cond_images_embeddings)
                else:
                    d_rf_loss = self.real_fake_discriminator_train_step(cond_images, real_images, batch_size)
                    d_c_loss = self.compatibility_discriminator_train_step(cond_images, real_images,
                                                                           not_compatible_images,
                                                                           batch_size)
                    g_loss = self.generator_train_step(cond_images, batch_size)

                # update training losses
                train_d_rf_loss += d_rf_loss * batch_size
                train_d_c_loss += d_c_loss * batch_size
                train_g_loss = g_loss * batch_size

                # calculate the batch fid and update the train fid
                torch.cuda.empty_cache()
                self.generator.eval()
                with torch.no_grad():
                    if self.autoencoder:
                        _, cond_images_embeddings = self.ae(cond_images)
                        train_fid += self.fid.calculate_fid(real_images,
                                                            self.generator(cond_images_embeddings)) * batch_size
                    else:
                        train_fid += self.fid.calculate_fid(real_images, self.generator(cond_images)) * batch_size

            # calculate average train losses
            train_d_rf_loss = train_d_rf_loss / len(self.trainloader.dataset)
            train_d_c_loss = train_d_c_loss / len(self.trainloader.dataset)
            train_g_loss = train_g_loss / len(self.trainloader.dataset)
            # calculate average train fid
            train_fid = train_fid / len(self.trainloader.dataset)
            self.train_fids.append(train_fid)

            # print training losses
            print('Train losses:')
            print('Generator loss: {:.6f} \tReal-Fake Discriminator loss: {:.6f} \t Compatibility Discriminator '
                  'loss: {:.6f}'.format(train_g_loss, train_d_rf_loss, train_d_c_loss))

            ######################
            # validate the model #
            ######################
            torch.cuda.empty_cache()
            self.generator.eval()
            with torch.no_grad():
                for images, _ in self.validationloader:
                    # move tensors to GPU if CUDA is available
                    cond_images = images[0].to(self.device)
                    real_images = images[1].to(self.device)
                    # calculate the batch fid and update validation fid
                    batch_size = real_images.size(0)
                    if self.autoencoder:
                        _, cond_images_embeddings = self.ae(cond_images)
                        validation_fid += self.fid.calculate_fid(real_images,
                                                                 self.generator(cond_images_embeddings)) * batch_size
                    else:
                        validation_fid += self.fid.calculate_fid(real_images, self.generator(cond_images)) * batch_size

                # calculate average validation fid
                validation_fid = validation_fid / len(self.validationloader.dataset)
                self.validation_fids.append(validation_fid)

                # print training/validation fids
                print('Train/Validation FIDs:')
                print('Training FID: {:.6f} \tValidation FID: {:.6f}'.format(
                    train_fid, validation_fid))

                # save model if validation fid has decreased
                if validation_fid <= self.best_fid_score:
                    self.best_epoch = epoch
                    print('\nValidation FID decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        self.best_fid_score,
                        validation_fid))
                    torch.save(self.generator.state_dict(), f'trained_{self.category}_generator.pth')
                    self.best_fid_score = validation_fid
                    self.best_generator = deepcopy(self.generator)
                    self.save_fids(train=True)
                    self.save_fids(train=False)

                # visually evaluate the generator
                self.visually_evaluation(self.category)

            plt.plot(self.train_fids, label='Training FID')
            plt.plot(self.validation_fids, label='Validation FID')
            plt.legend(frameon=False)

            self.save_fids(train=True)
            self.save_fids(train=False)

            print('\nBest epoch: ', self.best_epoch)
            print('Best FID: ', self.best_fid_score)

        # track test fid
        self.test_fid = 0.0
        generator = self.best_generator
        torch.cuda.empty_cache()
        generator.eval()
        with torch.no_grad():
            # iterate over test data
            for images, _ in self.testloader:
                # move tensors to GPU if CUDA is available
                cond_images = images[0].to(self.device)
                real_images = images[1].to(self.device)
                # update test fid
                batch_size = real_images.size(0)
                if self.autoencoder:
                    _, cond_images_embeddings = self.ae(cond_images)
                    self.test_fid += self.fid.calculate_fid(real_images,
                                                            self.generator(cond_images_embeddings)) * batch_size
                else:
                    self.test_fid += self.fid.calculate_fid(real_images, self.generator(cond_images)) * batch_size

            # calculate average test fid
            test_fid = self.test_fid / len(self.testloader.dataset)
            print('\n Test Loss: {:.6f}'.format(test_fid))

        return self.best_generator, self.train_fids, self.validation_fids, self.test_fid

    def generator_train_step(self, cond_images, batch_size, cond_images_embeddings=None):
        # clear the gradients of all optimized variables
        self.g_optimizer.zero_grad()

        if self.autoencoder:
            fake_images = self.generator(cond_images_embeddings)
        else:
            fake_images = self.generator(cond_images)

        # calculate the batch loss with respect to the real-fake discriminator
        validity = self.real_fake_discriminator(fake_images)
        g1_loss = self.criterion(validity, torch.ones(batch_size, 1).to(self.device))
        # g1_loss.backward(retain_graph=True)

        # calculate the batch loss with respect to the compatibility discriminator
        compatibility = self.compatibility_discriminator(fake_images, cond_images)
        g2_loss = self.criterion(compatibility, torch.ones(batch_size, 1).to(self.device))
        # g2_loss.backward()

        # g_loss = (g1_loss + g2_loss) * 0.5
        g_loss = g1_loss + g2_loss
        # backward pass: compute gradient of the loss with respect to model parameters
        g_loss.backward()

        # perform a single optimization step (parameter update)
        self.g_optimizer.step()
        return g_loss.data.item()

    def compatibility_discriminator_train_step(self, cond_images, real_images, not_compatible_images, batch_size,
                                               cond_images_embeddings=None):
        # clear the gradients of all optimized variables
        self.d_c_optimizer.zero_grad()

        # calculate the batch loss with compatible items
        real_validity = self.compatibility_discriminator(real_images, cond_images)
        compatibility_loss = self.criterion(real_validity, torch.ones(batch_size, 1).to(self.device))
        # compatibility_loss.backward()

        # calculate the batch loss with fake images
        if self.autoencoder:
            fake_images = self.generator(cond_images_embeddings)
        else:
            fake_images = self.generator(cond_images)
        fake_validity = self.compatibility_discriminator(fake_images, cond_images)
        fake_compatibility_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # fake_compatibility_loss.backward()

        # calculate the batch loss with not compatible items
        fake_validity = self.compatibility_discriminator(not_compatible_images, cond_images)
        not_compatibility_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # not_compatibility_loss.backward()

        d_loss = compatibility_loss + fake_compatibility_loss + not_compatibility_loss
        # backward pass: compute gradient of the loss with respect to model parameters
        d_loss.backward()

        # perform a single optimization step (parameter update)
        self.d_c_optimizer.step()
        return d_loss.data.item()

    def real_fake_discriminator_train_step(self, cond_images, real_images, batch_size, cond_images_embeddings=None):
        # clear the gradients of all optimized variables
        self.d_rf_optimizer.zero_grad()

        # calculate the batch loss with real images
        real_validity = self.real_fake_discriminator(real_images)
        real_loss = self.criterion(real_validity, torch.ones(batch_size, 1).to(self.device))
        # real_loss.backward()

        # calculate the batch loss with fake images
        if self.autoencoder:
            fake_images = self.generator(cond_images_embeddings)
        else:
            fake_images = self.generator(cond_images)
        fake_validity = self.real_fake_discriminator(fake_images)
        fake_loss = self.criterion(fake_validity, torch.zeros(batch_size, 1).to(self.device))
        # fake_loss.backward()

        d_loss = real_loss + fake_loss
        # backward pass: compute gradient of the loss with respect to model parameters
        d_loss.backward()

        # perform a single optimization step (parameter update)
        self.d_rf_optimizer.step()
        return d_loss.data.item()

    def visually_evaluation(self, category):
        # images = next(iter(self.validationloader))
        for images, _ in self.validationloader:
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
                ax1.set_title(f"Input top")

                ax2 = fig.add_subplot(3, 3, 3 * idx + 2, xticks=[], yticks=[])
                img2 = (compatible[idx] * 127.5 + 127.5).astype(int)
                plt.imshow(np.transpose(img2, (1, 2, 0)))
                ax2.set_title(f"Real compatible {category}")

                ax3 = fig.add_subplot(3, 3, 3 * idx + 3, xticks=[], yticks=[])
                img3 = (generated[idx] * 127.5 + 127.5).astype(int)
                plt.imshow(np.transpose(img3, (1, 2, 0)))
                ax3.set_title(f"Generated {category}")
            plt.show()
            break

    def save_fids(self, train=True):
        if train:
            csv_name = f'{self.category}_gan_train_FIDs'
            fids = self.train_fids
        else:
            csv_name = f'{self.category}_gan_validation_FIDs'
            fids = self.validation_fids
        with open(f'../checkpoints/{csv_name}', mode='w', newline='') as file_csv:
            fieldnames = ['Epoch', 'FID']
            writer = csv.DictWriter(file_csv, fieldnames=fieldnames)

            writer.writeheader()

            for epoch, fid in enumerate(fids, start=1):
                writer.writerow({'Epoch': epoch, 'FID': fid})


accessories_gan_trainer = GenerativeAdversarialNetworkTrainer(train_set_path=
                                                              '../preprocessing/json/filtered/gan_train_set_ta.json',
                                                              validation_set_path=
                                                              '../preprocessing/json/filtered/gan_validation_set_ta'
                                                              '.json',
                                                              test_set_path=
                                                              '../preprocessing/json/filtered/gan_test_set_ta.json',
                                                              autoencoder=False, category='accessory')
accessories_generator, ag_train_fids, ag_validation_fids, ag_test_fid = accessories_gan_trainer.train_and_test()
