from copy import deepcopy
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

from utility.custom_image_dataset import CustomImageDatasetAE
from architecture.ae import SSIM_Loss
from architecture.ae import AutoEncoder
from utility.utils import read_json


class AutoencoderTrainer:
    def __init__(self, train_set_path, validation_set_path, test_set_path, images_dir='../images'):
        self.optimizer = None
        self.criterion = None
        self.ae = None
        self.device = None
        self.testloader = None
        self.validationloader = None
        self.trainloader = None
        self.setup(train_set_path, validation_set_path, test_set_path, images_dir)

    def setup(self, train_set_path, validation_set_path, test_set_path, images_dir):
        train_set = read_json(train_set_path)
        validation_set = read_json(validation_set_path)
        test_set = read_json(test_set_path)

        train_transform = transforms.Compose([
            transforms.Normalize([0, 0, 0], [255, 255, 255]),
            transforms.RandomCrop(size=128, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Normalize([0, 0, 0], [255, 255, 255]),
            transforms.Resize(128),
            # transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(128),
            # transforms.ToTensor()
        ])

        trainset = CustomImageDatasetAE(img_dir=images_dir, data=train_set, transform=train_transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

        validationset = CustomImageDatasetAE(img_dir=images_dir, data=validation_set, transform=val_transform)
        self.validationloader = torch.utils.data.DataLoader(validationset, batch_size=128, shuffle=False)

        testset = CustomImageDatasetAE(img_dir=images_dir, data=test_set, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ae = AutoEncoder().to(self.device)
        self.criterion = SSIM_Loss()
        self.optimizer = Adam(self.ae.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.optimizer = Adam(ae.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-4)

    def train(self, n_epochs=30):
        valid_loss_min = np.Inf  # track change in validation loss
        best_model = deepcopy(self.ae)

        train_losses, validation_losses = [], []

        for epoch in range(1, n_epochs + 1):
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            self.ae.train()
            for data, _ in self.trainloader:
                # move tensors to GPU if CUDA is available
                data = data.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.ae(data)
                # calculate the batch loss
                loss = self.criterion(data, output)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            ######################
            # validate the model #
            ######################
            self.ae.eval()
            for data, _ in self.validationloader:
                # move tensors to GPU if CUDA is available
                data = data.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.ae(data)
                # calculate the batch loss
                loss = self.criterion(data, output)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(self.trainloader.dataset)  # train_loss/len(trainloader.sampler)
            valid_loss = valid_loss / len(self.validationloader.dataset)

            train_losses.append(train_loss)
            validation_losses.append(valid_loss)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(self.ae.state_dict(), 'trained_ae_128.pth')
                valid_loss_min = valid_loss
                best_model = deepcopy(self.ae)

        plt.plot(train_losses, label='Training loss')
        plt.plot(validation_losses, label='Validation loss')
        plt.legend(frameon=False)

        # track test loss
        test_loss = 0.0

        ae = best_model
        ae.eval()
        # iterate over test data
        for data, _ in self.testloader:
            # move tensors to GPU if CUDA is available
            data = data.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = ae(data)
            # calculate the batch loss
            loss = self.criterion(data, output)
            # update test loss
            test_loss += loss.item() * data.size(0)

        # average test loss
        test_loss = test_loss / len(self.testloader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
