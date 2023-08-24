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
    img = images[idx]
    plt.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_title(labels[idx])
plt.show()

#######################################################################################################################
from architecture.gan import Discriminator
from architecture.gan import Generator
from torch.autograd import Variable
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = torch.nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-2, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2, betas=(0.5, 0.999))

#
# def generator_loss(label, fake_output):
#     gen_loss = binary_cross_entropy(label, fake_output)
#     # print(gen_loss)
#     return gen_loss
#
#
# def discriminator_loss(label, output):
#     disc_loss = binary_cross_entropy(label, output)
#     # print(total_loss)
#     return disc_loss
#
#
# num_epochs = 200
# for epoch in range(1, num_epochs + 1):
#
#     D_loss_list, G_loss_list = [], []
#
#     for index, (real_images, labels) in enumerate(trainloader):
#         D_optimizer.zero_grad()
#         real_images = real_images[3].to(device)  # select tops from the images
#
#         real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
#         fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))
#
#         D_real_loss = discriminator_loss(discriminator(real_images), real_target)
#         # print(discriminator(real_images))
#         # D_real_loss.backward()
#
#         noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
#         noise_vector = noise_vector.to(device)
#
#         generated_image = generator(noise_vector)
#         output = discriminator(generated_image.detach())
#         D_fake_loss = discriminator_loss(output, fake_target)
#
#         # train with fake
#         # D_fake_loss.backward()
#
#         D_total_loss = (D_real_loss + D_fake_loss) / 2
#         D_loss_list.append(D_total_loss)
#
#         D_total_loss.backward()
#         D_optimizer.step()
#
#         # Train generator with real labels
#         G_optimizer.zero_grad()
#         G_loss = generator_loss(discriminator(generated_image), real_target)
#         G_loss_list.append(G_loss)
#
#         G_loss.backward()
#         G_optimizer.step()

def generator_train_step(z, batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    # fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    noise = Variable(torch.randn(batch_size, 100)).to(device)
    fake_images = generator(noise)
    validity = discriminator(fake_images)
    g_loss = criterion(validity, Variable(torch.ones(batch_size, 1)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


def discriminator_train_step(z, batch_size, discriminator, generator, d_optimizer, criterion, real_images):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size, 1)).to(device))

    # train with fake images
    # fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    noise = Variable(torch.randn(batch_size, 100)).to(device)
    fake_images = generator(noise)
    fake_validity = discriminator(fake_images)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size, 1)).to(device))

    d_loss = 0.5 * (real_loss + fake_loss)
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()


num_epochs = 30
batch_size = 64
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(trainloader):
        real_images = Variable(images[3]).to(device)
        batch_size = real_images.size(0)
        z = Variable(torch.randn(batch_size, 100)).to(device)
        # labels = Variable(labels[3]).cuda()
        generator.train()
        d_loss = discriminator_train_step(z, len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images)

        g_loss = generator_train_step(z, batch_size, discriminator, generator, g_optimizer, criterion)

    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    z = Variable(torch.randn(9, 100)).to(device)
    # labels = Variable(torch.LongTensor(np.arange(9))).cuda()

    # sample_images = generator(z).unsqueeze(1).data.cpu()
    # grid = torchvision.utils.make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
    # plt.imshow(grid)
    # plt.show()
    sample_images = generator(z)
    images = [el.detach().numpy() for el in sample_images]
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(9):
        ax = fig.add_subplot(3, 3, idx + 1, xticks=[], yticks=[])
        img = (images[idx] * 127.5 + 127.5).astype(int)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(idx)
    plt.show()
