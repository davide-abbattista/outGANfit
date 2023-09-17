import json
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
from torchvision import transforms

from utility.ae_custom_image_dataset import CustomImageDataset
from architecture.ae import SSIM_Loss
from architecture.ae import AutoEncoder

import matplotlib.pyplot as plt

with open('..\preprocessing\json\\filtered\\ae_train_set.json', 'r') as train_data:
    train_set = json.load(train_data)

with open('..\preprocessing\json\\filtered\\ae_validation_set.json', 'r') as validation_data:
    validation_set = json.load(validation_data)

with open('..\preprocessing\json\\filtered\\ae_test_set.json', 'r') as test_data:
    test_set = json.load(test_data)

train_transform = transforms.Compose([
    transforms.RandomCrop(size=128, pad_if_needed=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize(128),
    # transforms.ToTensor()
])

trainset = CustomImageDataset(img_dir='..\images', data=train_set, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

validationset = CustomImageDataset(img_dir='..\images', data=validation_set, transform=val_test_transform)
validationloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)

testset = CustomImageDataset(img_dir='..\images', data=test_set, transform=val_test_transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae = AutoEncoder().to(device)
criterion = SSIM_Loss()
optimizer = Adam(ae.parameters(), lr=1e-4, weight_decay=1e-5)


# optimizer = Adam(ae.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-4)


def train(ae, optimizer, criterion, trainloader, validationloader, device, n_epochs=30):
    valid_loss_min = np.Inf  # track change in validation loss
    best_model = deepcopy(ae)

    train_losses, validation_losses = [], []

    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        ae.train()
        for data, _ in trainloader:
            # move tensors to GPU if CUDA is available
            data = data.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = ae(data)
            # calculate the batch loss
            loss = criterion(data, output)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        ae.eval()
        for data, _ in validationloader:
            # move tensors to GPU if CUDA is available
            data = data.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = ae(data)
            # calculate the batch loss
            loss = criterion(data, output)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(trainloader.dataset)  # train_loss/len(trainloader.sampler)
        valid_loss = valid_loss / len(validationloader.dataset)

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
            torch.save(ae.state_dict(), 'trained_ae_128.pth')
            valid_loss_min = valid_loss
            best_model = deepcopy(ae)

    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend(frameon=False)

    # track test loss
    test_loss = 0.0

    ae = best_model
    ae.eval()
    # iterate over test data
    for data, _ in testloader:
        # move tensors to GPU if CUDA is available
        data = data.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = ae(data)
        # calculate the batch loss
        loss = criterion(data, output)
        # update test loss
        test_loss += loss.item() * data.size(0)

    # average test loss
    test_loss = test_loss / len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))


train(ae, optimizer, criterion, trainloader, validationloader, device)
