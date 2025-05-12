import torch
import numpy as np

def psnr(img1, img2):
    """Calculate PSNR between two images."""
    # Convert from [-1, 1] to [0, 1] range
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_val.item()

def ssim(img1, img2):
    """Calculate SSIM between two images."""
    # Convert from [-1, 1] to [0, 1] range
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()
    
    mu1 = np.mean(img1, axis=(1, 2), keepdims=True)
    mu2 = np.mean(img2, axis=(1, 2), keepdims=True)
    
    sigma1 = np.std(img1, axis=(1, 2), keepdims=True)
    sigma2 = np.std(img2, axis=(1, 2), keepdims=True)
    
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(1, 2), keepdims=True)
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
    return np.mean(ssim_map) 


