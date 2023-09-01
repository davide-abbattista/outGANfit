import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from utility.custom_image_dataset import CustomImageDataset

with open('.\preprocessing\json\\filtered\\train_set.json', 'r') as train_data:
    train_set = json.load(train_data)

with open('.\preprocessing\json\\filtered\\validation_set.json', 'r') as validation_data:
    validation_set = json.load(validation_data)

with open('.\preprocessing\json\\filtered\\test_set.json', 'r') as test_data:
    test_set = json.load(test_data)

transform = transforms.Compose([transforms.Resize(128),
                                transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),  # range [-1, 1]
                                # transforms.Normalize([0, 0, 0], [255, 255, 255])  # map values in the range [0, 1]
                                ])

trainset = CustomImageDataset(img_dir='.\images', data=train_set, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

validationset = CustomImageDataset(img_dir='.\images', data=validation_set, transform=transform)
validationloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = CustomImageDataset(img_dir='.\images', data=test_set, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

# plot the images in the first outfit sample of the batch, along with the corresponding labels
images = [el[0].numpy() for el in images]
labels = [el[0] for el in labels]
fig = plt.figure(figsize=(6, 2))
for idx in np.arange(4):
    ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
    img = (images[idx] * 127.5 + 127.5).astype(int)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_title(labels[idx])
plt.show()

#######################################################################################################################
from architecture.new_gan2 import Discriminator
from architecture.new_gan2 import Generator
from utility.architecture import weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
real_fake_discriminator = Discriminator(depth_in=3).to(device)
compatibility_discriminator = Discriminator(depth_in=6).to(device)
generator.apply(weights_init)
real_fake_discriminator.apply(weights_init)
compatibility_discriminator.apply(weights_init)

criterion = torch.nn.BCELoss()
d_rf_optimizer = torch.optim.Adam(real_fake_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_c_optimizer = torch.optim.Adam(compatibility_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))


# def generator_train_step(cond_images, batch_size, real_fake_discriminator, compatibility_discriminator, generator,
#                          g_optimizer, criterion):
#     g_optimizer.zero_grad()
#
#     fake_images = generator(cond_images)
#
#     validity = real_fake_discriminator(fake_images)
#     g1_loss = criterion(validity, torch.ones(batch_size, 1).to(device))
#     g1_loss.backward(retain_graph=True)
#
#     # fake_images = generator(cond_images)
#     compatibility = compatibility_discriminator(fake_images, cond_images)
#     g2_loss = criterion(compatibility, torch.ones(batch_size, 1).to(device))
#     g2_loss.backward()
#
#     g_loss = g1_loss + g2_loss
#     g_optimizer.step()
#     return g_loss.data.item()

def generator_train_step(cond_images, batch_size, real_fake_discriminator, compatibility_discriminator, generator,
                         g_optimizer, criterion):
    g_optimizer.zero_grad()

    fake_images = generator(cond_images)

    validity = real_fake_discriminator(fake_images)
    g1_loss = criterion(validity, torch.ones(batch_size, 1).to(device))

    compatibility = compatibility_discriminator(fake_images, cond_images)
    g2_loss = criterion(compatibility, torch.ones(batch_size, 1).to(device))

    g_loss = (g1_loss + g2_loss) * 0.5
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

def compatibility_discriminator_train_step(cond_images, batch_size, compatibility_discriminator, generator,
                                           d_c_optimizer, criterion, real_images):
    d_c_optimizer.zero_grad()

    # train with real images
    real_validity = compatibility_discriminator(real_images, cond_images)
    compatibility_loss = criterion(real_validity, torch.ones(batch_size, 1).to(device))
    compatibility_loss.backward()

    # train with fake images
    fake_images = generator(cond_images)
    fake_validity = compatibility_discriminator(fake_images, cond_images)
    not_compatibility_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    not_compatibility_loss.backward()

    d_loss = compatibility_loss + not_compatibility_loss
    d_c_optimizer.step()
    return d_loss.data.item()


def real_fake_discriminator_train_step(batch_size, real_fake_discriminator, generator, d_rf_optimizer, criterion,
                                       real_images):
    d_rf_optimizer.zero_grad()

    # train with real images
    real_validity = real_fake_discriminator(real_images)
    real_loss = criterion(real_validity, torch.ones(batch_size, 1).to(device))
    real_loss.backward()

    # train with fake images
    fake_images = generator(cond_images)
    fake_validity = real_fake_discriminator(fake_images)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    fake_loss.backward()

    d_loss = real_loss + fake_loss
    d_rf_optimizer.step()
    return d_loss.data.item()


num_epochs = 300
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(trainloader):
        real_images = images[1].to(device)
        cond_images = images[3].to(device)
        batch_size = real_images.size(0)

        generator.train()
        compatibility_discriminator.train()
        real_fake_discriminator.train()

        d_rf_loss = real_fake_discriminator_train_step(batch_size, real_fake_discriminator, generator, d_rf_optimizer,
                                                       criterion, real_images)
        d_c_loss = compatibility_discriminator_train_step(cond_images, batch_size, compatibility_discriminator,
                                                          generator, d_c_optimizer, criterion, real_images)
        g_loss = generator_train_step(cond_images, batch_size, real_fake_discriminator, compatibility_discriminator,
                                      generator, g_optimizer, criterion)

    generator.eval()
    print('g_loss: {}, d_rf_loss: {}, d_c_loss: {}'.format(g_loss, d_rf_loss, d_c_loss))

    images = next(iter(trainloader))
    top_images = images[0][3].to(device)
    bottom_images = images[0][1].to(device)
    bottom_generated_images = generator(top_images)
    top = [el.detach().cpu().numpy() for el in top_images]
    bottom = [el.detach().cpu().numpy() for el in bottom_images]
    generated = [el.detach().cpu().numpy() for el in bottom_generated_images]
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(3):
        ax1 = fig.add_subplot(3, 3, 3 * idx + 1, xticks=[], yticks=[])
        img1 = (top[idx] * 127.5 + 127.5).astype(int)
        plt.imshow(np.transpose(img1, (1, 2, 0)))
        ax1.set_title(f"Input top epoch {epoch}")

        ax2 = fig.add_subplot(3, 3, 3 * idx + 2, xticks=[], yticks=[])
        img2 = (bottom[idx] * 127.5 + 127.5).astype(int)
        plt.imshow(np.transpose(img2, (1, 2, 0)))
        ax2.set_title(f"Real bottom epoch {epoch}")

        ax3 = fig.add_subplot(3, 3, 3 * idx + 3, xticks=[], yticks=[])
        img3 = (generated[idx] * 127.5 + 127.5).astype(int)
        plt.imshow(np.transpose(img3, (1, 2, 0)))
        ax3.set_title(f"Generated bottom epoch {epoch}")
    plt.show()

torch.save(generator.state_dict(), 'generator.pth')
