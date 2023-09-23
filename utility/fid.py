import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
import torch


class FID:
    def __init__(self, range=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.inception_model = None
        self.device = device
        self.range = range
        self.load_model()

    def load_model(self):
        self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.inception_model = self.inception_model.eval().to(self.device)

    def calculate_activation(self, images):
        transform_for_FID = transforms.Compose([transforms.Resize(299),
                                                transforms.CenterCrop(299),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                ])
        if self.range == 255:
            images = images / 255.0
        else:
            images = (images + 1.0) / 2.0  # Rescale images from [-1, 1] to [0, 1]
        images = transform_for_FID(images)

        images = images.to(self.device)
        # images = images.clone().detach().to(self.device)
        # images = torch.tensor(images).to(self.device)  # Convert images to PyTorch tensors and move to device

        # activations = self.inception_model(images)
        if len(images) == 3:
            images = torch.unsqueeze(images, 0)

        activations = self.inception_model(images).detach().cpu().numpy()
        return activations

    def calculate_fid(self, real_images, generated_images):
        real_activations = self.calculate_activation(real_images)
        generated_activations = self.calculate_activation(generated_images)

        mu_real, sigma_real = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu_generated, sigma_generated = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)
        diff = mu_real - mu_generated

        if len(sigma_real.shape) != 2:
            sigma_real = np.expand_dims(np.expand_dims(sigma_real, 0), 0)
            sigma_generated = np.expand_dims(np.expand_dims(sigma_generated, 0), 0)

        cov_mean, _ = sqrtm(np.dot(sigma_real, sigma_generated), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        fid_score = np.dot(diff, diff) + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
        return fid_score
