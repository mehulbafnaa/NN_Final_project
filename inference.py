import torch
from models.generator import UNetGenerator
from utils.visualization import enhance_image
from utils.metrics import psnr, ssim
from PIL import Image
from torchvision import transforms
import os

def calculate_metrics(enhanced_tensor, gt_tensor):
    """
    Calculate PSNR and SSIM metrics between enhanced and ground truth images.
    
    Args:
        enhanced_tensor (torch.Tensor): Enhanced image tensor
        gt_tensor (torch.Tensor): Ground truth image tensor
    
    Returns:
        dict: Dictionary containing PSNR and SSIM values
    """
    metrics = {
        'psnr': psnr(enhanced_tensor, gt_tensor),
        'ssim': ssim(enhanced_tensor, gt_tensor)
    }
    return metrics

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained generator
    generator = UNetGenerator().to(device)
    checkpoint = torch.load("checkpoints/final_model.pth", map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Example usage
    test_image_path = "/Users/seanhuvaya/Downloads/Project/Data/Dataset_Part1/360/1.JPG"  # Replace with actual path
    output_path = "enhanced_output.jpg"
    
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load low-light image
    low_image = Image.open(test_image_path).convert('RGB')
    low_tensor = transform(low_image).unsqueeze(0).to(device)
    
    # Generate enhanced image
    with torch.no_grad():
        enhanced_tensor = generator(low_tensor)
    
    # Check for ground truth image
    gt_path = "/Users/seanhuvaya/Downloads/Project/Data/Dataset_Part1/Label/1.JPG"
    metrics = None
    if os.path.exists(gt_path):
        # Load ground truth image
        gt_image = Image.open(gt_path).convert('RGB')
        gt_tensor = transform(gt_image).unsqueeze(0).to(device)
        
        # Calculate metrics
        metrics = calculate_metrics(enhanced_tensor, gt_tensor)
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Enhance and save the image
    enhanced_image = enhance_image(
        generator=generator,
        image_path=test_image_path,
        output_path=output_path,
        device=device
    )
    
    print(f"Enhanced image saved to {output_path}")

if __name__ == "__main__":
    main()