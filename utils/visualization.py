import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
import os
from utils.metrics import psnr, ssim

def save_sample_images(generator, data_loader, epoch, device, num_samples=4):
    """
    Save sample enhanced images during training.
    
    Args:
        generator: Generator model
        data_loader: DataLoader containing validation images
        epoch (int): Current epoch number
        device: Device to run the model on
        num_samples (int): Number of samples to save
    """
    generator.eval()
    
    # Create directory for samples if it doesn't exist
    os.makedirs('samples', exist_ok=True)
    
    with torch.no_grad():
        # Get a batch of images
        batch = next(iter(data_loader))
        low_imgs = batch['low'].to(device)
        normal_imgs = batch['normal'].to(device)
        
        # Generate enhanced images
        enhanced_imgs = generator(low_imgs)
        
        # Create a grid of images
        comparison = torch.cat([low_imgs[:num_samples], 
                              enhanced_imgs[:num_samples], 
                              normal_imgs[:num_samples]], dim=0)
        
        # Save the grid
        vutils.save_image(comparison,
                         f'samples/comparison_epoch_{epoch}.png',
                         normalize=True,
                         nrow=num_samples)

def plot_training_history(history):
    """
    Plot training history metrics and save each plot separately.
    
    Args:
        history (dict): Dictionary containing training metrics
    """
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot generator loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['g_loss'], label='Generator Loss', color='red')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/generator_loss.png')
    plt.close()
    
    # Plot discriminator loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['d_loss'], label='Discriminator Loss', color='blue')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/discriminator_loss.png')
    plt.close()
    
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_psnr'], label='Validation PSNR', color='green')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/psnr.png')
    plt.close()
    
    # Plot SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_ssim'], label='Validation SSIM', color='purple')
    plt.title('Structural Similarity Index (SSIM)')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/ssim.png')
    plt.close()
    

def enhance_image(generator, image_path, output_path=None, device=None):
    """
    Enhance a single low-light image using the trained generator and calculate metrics.
    
    Args:
        generator: Generator model
        image_path (str): Path to the input low-light image
        output_path (str, optional): Path to save the enhanced image
        device: Device to run the model on
    
    Returns:
        tuple: (enhanced_image, metrics_dict) where metrics_dict contains PSNR and SSIM values
    """
    generator.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate enhanced image
    with torch.no_grad():
        enhanced_tensor = generator(input_tensor)
    
    # Convert to PIL image
    enhanced_tensor = (enhanced_tensor + 1) / 2  # [-1, 1] to [0, 1]
    enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())
    
    # Calculate metrics
    metrics = {}
    
    # If we have a ground truth image (assuming it's in the same directory with '_gt' suffix)
    gt_path = image_path.replace('.', '_gt.')
    if os.path.exists(gt_path):
        gt_image = Image.open(gt_path).convert('RGB')
        gt_tensor = transform(gt_image).unsqueeze(0).to(device)
        
        # Calculate PSNR using our implementation
        metrics['psnr'] = psnr(enhanced_tensor, gt_tensor)
        
        # Calculate SSIM using our implementation
        metrics['ssim'] = ssim(enhanced_tensor, gt_tensor)
    
    # Save enhanced image if output path is provided
    if output_path:
        enhanced_image.save(output_path)
    
    # Display original and enhanced images with metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Low-light Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image)
    if metrics:
        plt.title(f'Enhanced Image\nPSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.4f}')
    else:
        plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return enhanced_image, metrics 