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
from architecture.gan import Discriminator
from architecture.gan import Generator
from utility.architecture import weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = torch.nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()

    # fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    noise = torch.randn(batch_size, 100).to(device)
    fake_images = generator(noise)
    validity = discriminator(fake_images)

    g_loss = criterion(validity, torch.ones(batch_size, 1).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


# def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images):
#     d_optimizer.zero_grad()
#
#     # train with real images
#     real_validity = discriminator(real_images)
#     real_loss = criterion(real_validity, Variable(torch.ones(batch_size, 1)).to(device))
#
#     # train with fake images
#     # fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
#     noise = Variable(torch.randn(batch_size, 100)).to(device)
#     fake_images = generator(noise)
#     fake_validity = discriminator(fake_images)
#     fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size, 1)).to(device))
#
#     d_loss = real_loss + fake_loss
#     d_loss.backward()
#     d_optimizer.step()
#     return d_loss.data.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images)
    real_loss = criterion(real_validity, torch.ones(batch_size, 1).to(device))
    real_loss.backward()

    # train with fake images
    # fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    noise = torch.randn(batch_size, 100).to(device)
    fake_images = generator(noise)
    fake_validity = discriminator(fake_images)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    fake_loss.backward()

    d_loss = real_loss + fake_loss
    d_optimizer.step()
    return d_loss.data.item()


num_epochs = 800
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(trainloader):
        real_images = images[3].to(device)
        batch_size = real_images.size(0)
        # labels = Variable(labels[3]).cuda()
        generator.train()
        discriminator.train()
        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images)

        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

    # labels = Variable(torch.LongTensor(np.arange(9))).cuda()

    # sample_images = generator(z).unsqueeze(1).data.cpu()
    # grid = torchvision.utils.make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
    # plt.imshow(grid)
    # plt.show()
    if np.mod(epoch, 10) == 0:
        z = torch.randn(9, 100).to(device)
        sample_images = generator(z)
        images = [el.detach().numpy() for el in sample_images]
        fig = plt.figure(figsize=(10, 10))
        for idx in np.arange(9):
            ax = fig.add_subplot(3, 3, idx + 1, xticks=[], yticks=[])
            img = (images[idx] * 127.5 + 127.5).astype(int)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(f"Epoch {epoch}")
        plt.show()

torch.save(generator.state_dict(), 'generator.pth')

generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Generate new images using the generator
num_samples = 9  # Number of images to generate
z = torch.randn(num_samples, latent_dim, device=device)  # Generate random noise vectors
with torch.no_grad():
    generated_images = generator(z)

# Plot the generated images
generated_images = [image.detach().cpu().numpy() for image in generated_images]
fig = plt.figure(figsize=(10, 10))
for idx in range(num_samples):
    ax = fig.add_subplot(3, 3, idx + 1, xticks=[], yticks=[])
    img = ((generated_images[idx] + 1) / 2).transpose(1, 2, 0)  # Undo normalization
    plt.imshow(img)
plt.show()
