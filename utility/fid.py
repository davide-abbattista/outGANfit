import numpy as np
from scipy.linalg import sqrtm
from torch import nn
from torchvision.models import inception_v3

from torchvision import transforms
import torch


class FID:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.inception_model = None
        self.device = device
        self.load_model()

    def load_model(self):
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()
        self.inception_model = self.inception_model.eval().to(self.device)

    def calculate_activation(self, images):
        transform_for_FID = transforms.Compose([transforms.Resize(299),
                                                transforms.CenterCrop(299),
                                                ])
        images = transform_for_FID(images)
        images = images.to(self.device)

        activations = self.inception_model(images).detach().cpu().numpy()
        return activations

    def calculate_fid(self, real_images, generated_images):
        real_activations = self.calculate_activation(real_images)
        generated_activations = self.calculate_activation(generated_images)

        mu_real, sigma_real = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu_generated, sigma_generated = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)
        diff = mu_real - mu_generated

        cov_mean, _ = sqrtm(np.dot(sigma_real, sigma_generated), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        fid_score = np.dot(diff, diff) + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
        return fid_score
