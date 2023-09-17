import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
import torch

transform_for_FID = transforms.Compose([transforms.Resize(299),
                                        transforms.CenterCrop(299),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

def calculate_activation(images, model, device):
    images = (images + 1.0) / 2.0  # Rescale images from [-1, 1] to [0, 1]
    images = transform_for_FID(images)

    images = images.to(device)
    # images = images.clone().detach().to(device)
    # images = torch.tensor(images).to(device)  # Convert images to PyTorch tensors and move to device

    # activations = model(images)
    activations = model(images).detach().cpu().numpy()
    return activations

def calculate_fid(real_images, generated_images, model, device):
    real_activations = calculate_activation(real_images, model, device)
    generated_activations = calculate_activation(generated_images, model, device)

    mu_real, sigma_real = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu_generated, sigma_generated = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)
    diff = mu_real - mu_generated
    cov_mean, _ = sqrtm(np.dot(sigma_real, sigma_generated), disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid_score = np.dot(diff, diff) + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
    return fid_score