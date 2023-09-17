import json

import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from architecture.gan import Discriminator
from architecture.gan import Generator
from utility.weights_init import weights_init
from utility.gan_custom_image_dataset import CustomImageDataset
from utility.fid import calculate_fid

with open('.\preprocessing\json\\filtered\\gan_train_set.json', 'r') as train_data:
    train_set = json.load(train_data)

with open('.\preprocessing\json\\filtered\\gan_validation_set.json', 'r') as validation_data:
    validation_set = json.load(validation_data)

with open('.\preprocessing\json\\filtered\\gan_test_set.json', 'r') as test_data:
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accessories_generator = Generator().to(device)
bottoms_generator = Generator().to(device)
shoes_generator = Generator().to(device)

accessories_real_fake_discriminator = Discriminator(depth_in=3).to(device)
accessories_compatibility_discriminator = Discriminator(depth_in=6).to(device)
bottoms_real_fake_discriminator = Discriminator(depth_in=3).to(device)
bottoms_compatibility_discriminator = Discriminator(depth_in=6).to(device)
shoes_real_fake_discriminator = Discriminator(depth_in=3).to(device)
shoes_compatibility_discriminator = Discriminator(depth_in=6).to(device)

accessories_generator.apply(weights_init)
accessories_real_fake_discriminator.apply(weights_init)
accessories_compatibility_discriminator.apply(weights_init)
bottoms_generator.apply(weights_init)
bottoms_real_fake_discriminator.apply(weights_init)
bottoms_compatibility_discriminator.apply(weights_init)
shoes_generator.apply(weights_init)
shoes_real_fake_discriminator.apply(weights_init)
shoes_compatibility_discriminator.apply(weights_init)

accessories_d_rf_optimizer = torch.optim.Adam(accessories_real_fake_discriminator.parameters(), lr=2e-4,
                                              betas=(0.5, 0.999))
accessories_d_c_optimizer = torch.optim.Adam(accessories_compatibility_discriminator.parameters(), lr=2e-4,
                                             betas=(0.5, 0.999))
accessories_g_optimizer = torch.optim.Adam(accessories_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
bottoms_d_rf_optimizer = torch.optim.Adam(bottoms_real_fake_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
bottoms_d_c_optimizer = torch.optim.Adam(bottoms_compatibility_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
bottoms_g_optimizer = torch.optim.Adam(bottoms_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
shoes_d_rf_optimizer = torch.optim.Adam(shoes_real_fake_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
shoes_d_c_optimizer = torch.optim.Adam(shoes_compatibility_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
shoes_g_optimizer = torch.optim.Adam(shoes_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion = torch.nn.BCELoss()

# inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
inception_model = inception_model.eval().to(device)


def generator_train_step(cond_images, batch_size, real_fake_discriminator, compatibility_discriminator, generator,
                         g_optimizer, criterion):
    g_optimizer.zero_grad()

    fake_images = generator(cond_images)

    validity = real_fake_discriminator(fake_images)
    g1_loss = criterion(validity, torch.ones(batch_size, 1).to(device))
    # g1_loss.backward(retain_graph=True)

    compatibility = compatibility_discriminator(fake_images, cond_images)
    g2_loss = criterion(compatibility, torch.ones(batch_size, 1).to(device))
    # g2_loss.backward()

    # g_loss = (g1_loss + g2_loss) * 0.5
    g_loss = g1_loss + g2_loss
    g_loss.backward()

    g_optimizer.step()
    return g_loss.data.item()


def compatibility_discriminator_train_step(cond_images, batch_size, compatibility_discriminator, generator,
                                           d_c_optimizer, criterion, real_images, not_compatible_images):
    d_c_optimizer.zero_grad()

    # train with compatible items
    real_validity = compatibility_discriminator(real_images, cond_images)
    compatibility_loss = criterion(real_validity, torch.ones(batch_size, 1).to(device))
    # compatibility_loss.backward()

    # train with fake images
    fake_images = generator(cond_images)
    fake_validity = compatibility_discriminator(fake_images, cond_images)
    fake_compatibility_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    # fake_compatibility_loss.backward()

    # train with not compatible items
    fake_validity = compatibility_discriminator(not_compatible_images, cond_images)
    not_compatibility_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    # not_compatibility_loss.backward()

    d_loss = compatibility_loss + fake_compatibility_loss + not_compatibility_loss
    d_loss.backward()

    d_c_optimizer.step()
    return d_loss.data.item()


def real_fake_discriminator_train_step(cond_images, batch_size, real_fake_discriminator, generator, d_rf_optimizer,
                                       criterion,
                                       real_images):
    d_rf_optimizer.zero_grad()

    # train with real images
    real_validity = real_fake_discriminator(real_images)
    real_loss = criterion(real_validity, torch.ones(batch_size, 1).to(device))
    # real_loss.backward()

    # train with fake images
    fake_images = generator(cond_images)
    fake_validity = real_fake_discriminator(fake_images)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, 1).to(device))
    # fake_loss.backward()

    d_loss = real_loss + fake_loss
    d_loss.backward()

    d_rf_optimizer.step()
    return d_loss.data.item()


def train(category, generator, real_fake_discriminator, compatibility_discriminator, trainloader, validationloader,
          criterion, d_c_optimizer, d_rf_optimizer, g_optimizer, device, num_epochs=300):
    best_fid_score = float('inf')  # Initialize with a high value
    best_epoch = 0

    train_fids = []
    validation_fids = []

    for epoch in range(num_epochs):
        print('\nStarting epoch {}...'.format(epoch))
        train_d_rf_loss = 0.0
        train_d_c_loss = 0.0
        train_g_loss = 0.0
        train_fid = 0.0

        for images, _ in trainloader:
            match category:
                case 'accessories':
                    real_images = images[0].to(device)
                    not_compatible_images = images[4].to(device)
                case 'bottoms':
                    real_images = images[1].to(device)
                    not_compatible_images = images[5].to(device)
                case 'shoes':
                    real_images = images[2].to(device)
                    not_compatible_images = images[6].to(device)
            cond_images = images[3].to(device)

            batch_size = real_images.size(0)

            generator.train()
            compatibility_discriminator.train()
            real_fake_discriminator.train()

            d_rf_loss = real_fake_discriminator_train_step(cond_images, batch_size, real_fake_discriminator,
                                                           generator, d_rf_optimizer,
                                                           criterion, real_images)
            d_c_loss = compatibility_discriminator_train_step(cond_images, batch_size,
                                                              compatibility_discriminator,
                                                              generator, d_c_optimizer,
                                                              criterion, real_images,
                                                              not_compatible_images)
            g_loss = generator_train_step(cond_images, batch_size, real_fake_discriminator,
                                          compatibility_discriminator,
                                          generator, g_optimizer, criterion)

            train_d_rf_loss += d_rf_loss * batch_size
            train_d_c_loss += d_c_loss * batch_size
            train_g_loss = g_loss * batch_size

            generator.eval()
            train_fid += calculate_fid(real_images, generator(cond_images), inception_model, device) * batch_size

        train_d_rf_loss = train_d_rf_loss / len(trainloader.dataset)
        train_d_c_loss = train_d_c_loss / len(trainloader.dataset)
        train_g_loss = train_g_loss / len(trainloader.dataset)

        train_fid = train_fid / len(trainloader.dataset)
        train_fids.append(train_fid)

        print(f'{category} train losses')
        print('g_loss: {}, d_rf_loss: {}, d_c_loss: {}'.format(train_g_loss, train_d_rf_loss, train_d_c_loss))
        print('FID Score: {}'.format(train_fid))

        # EVALUATION

        generator.eval()
        validation_fid = 0.0

        for images, _ in validationloader:
            match category:
                case 'accessories':
                    real_images = images[0].to(device)
                case 'bottoms':
                    real_images = images[1].to(device)
                case 'shoes':
                    real_images = images[2].to(device)
            cond_images = images[3].to(device)

            validation_fid += calculate_fid(real_images, generator(cond_images), inception_model, device) * batch_size

        validation_fid = validation_fid / len(validationloader.dataset)
        validation_fids.append(validation_fid)

        print(f'{category} validation loss')
        print('FID Score: {}'.format(validation_fid))
        if validation_fid < best_fid_score:
            best_epoch = epoch
            print('Validation FID decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                best_fid_score,
                validation_fid))
            best_fid_score = validation_fid
            torch.save(generator.state_dict(), f'{category}_generator.pth')
        #
        # images = next(iter(validationloader))
        # top_images = images[0][3].to(device)
        # # bottom_images = images[0][1].to(device)
        # accessories_images = images[0][0].to(device)
        # # shoes_images = images[0][2].to(device)
        #
        # accessories_generated_images = accessories_generator(top_images)
        # # bottom_generated_images = bottom_generator(top_images)
        # # shoes_generated_images = shoes_generator(top_images)
        #
        # top = [el.detach().cpu().numpy() for el in top_images]
        # # bottom = [el.detach().cpu().numpy() for el in bottom_images]
        # accessories = [el.detach().cpu().numpy() for el in accessories_images]
        # # shoes = [el.detach().cpu().numpy() for el in shoes_images]
        # accessories_generated = [el.detach().cpu().numpy() for el in accessories_generated_images]
        # # bottom_generated = [el.detach().cpu().numpy() for el in bottom_generated_images]
        # # shoes_generated = [el.detach().cpu().numpy() for el in shoes_generated_images]
        #
        # fig = plt.figure(figsize=(10, 10))
        # for idx in np.arange(2):
        #     ax1 = fig.add_subplot(3, 3, 3 * idx + 1, xticks=[], yticks=[])
        #     img1 = (top[idx] * 127.5 + 127.5).astype(int)
        #     plt.imshow(np.transpose(img1, (1, 2, 0)))
        #     ax1.set_title(f"Input top epoch {epoch}")
        #
        #     # #bottom
        #     # ax2 = fig.add_subplot(2, 7, 7 * idx + 2, xticks=[], yticks=[])
        #     # img2 = (bottom[idx] * 127.5 + 127.5).astype(int)
        #     # plt.imshow(np.transpose(img2, (1, 2, 0)))
        #     # ax2.set_title(f"Real bottom epoch {epoch}")
        #
        #     # ax3 = fig.add_subplot(2, 7, 7 * idx + 3, xticks=[], yticks=[])
        #     # img3 = (bottom_generated[idx] * 127.5 + 127.5).astype(int)
        #     # plt.imshow(np.transpose(img3, (1, 2, 0)))
        #     # ax3.set_title(f"Generated bottom epoch {epoch}")
        #
        #     # accessories
        #     ax2 = fig.add_subplot(3, 3, 3 * idx + 2, xticks=[], yticks=[])
        #     img2 = (accessories[idx] * 127.5 + 127.5).astype(int)
        #     plt.imshow(np.transpose(img2, (1, 2, 0)))
        #     ax2.set_title(f"Real accessory epoch {epoch}")
        #
        #     ax3 = fig.add_subplot(3, 3, 3 * idx + 3, xticks=[], yticks=[])
        #     img3 = (accessories_generated[idx] * 127.5 + 127.5).astype(int)
        #     plt.imshow(np.transpose(img3, (1, 2, 0)))
        #     ax3.set_title(f"Generated accessory epoch {epoch}")
        #
        #     # #shoes
        #     # ax6 = fig.add_subplot(2, 7, 7 * idx + 6, xticks=[], yticks=[])
        #     # img6 = (shoes[idx] * 127.5 + 127.5).astype(int)
        #     # plt.imshow(np.transpose(img6, (1, 2, 0)))
        #     # ax6.set_title(f"Real shoes epoch {epoch}")
        #
        #     # ax7 = fig.add_subplot(2, 7, 7 * idx + 7, xticks=[], yticks=[])
        #     # img7 = (shoes_generated[idx] * 127.5 + 127.5).astype(int)
        #     # plt.imshow(np.transpose(img7, (1, 2, 0)))
        #     # ax7.set_title(f"Generated shoes epoch {epoch}")
        # plt.show()
        #
        print('\nBest epoch: ', best_epoch)
        print('Best FID: ', best_fid_score)


train('accessories', accessories_generator, accessories_real_fake_discriminator,
      accessories_compatibility_discriminator, trainloader, validationloader,
      criterion, accessories_d_c_optimizer, accessories_d_rf_optimizer, accessories_g_optimizer, device)
train('bottoms', bottoms_generator, bottoms_real_fake_discriminator, bottoms_compatibility_discriminator, trainloader,
      validationloader,
      criterion, bottoms_d_c_optimizer, bottoms_d_rf_optimizer, bottoms_g_optimizer, device)
train('shoes', shoes_generator, shoes_real_fake_discriminator, shoes_compatibility_discriminator, trainloader,
      validationloader,
      criterion, shoes_d_c_optimizer, shoes_d_rf_optimizer, shoes_g_optimizer, device)
