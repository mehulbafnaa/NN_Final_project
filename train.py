import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import VGGPerceptualLoss, adversarial_loss, l1_loss
from data.dataset import LOLDataset, get_transforms
from utils.metrics import psnr, ssim
from utils.visualization import save_sample_images, plot_training_history

def train_model(generator, discriminator, train_loader, val_loader, num_epochs=50, device=None):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Learning rate schedulers with warmup
    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    
    warmup_iters = min(1000, len(train_loader) - 1)
    warmup_scheduler_G = warmup_lr_scheduler(optimizer_G, warmup_iters, 0.001)
    warmup_scheduler_D = warmup_lr_scheduler(optimizer_D, warmup_iters, 0.001)
    
    # Main schedulers with more aggressive decay
    scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_G, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6
    )
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_D, 
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss weights
    lambda_adv = 0.05
    lambda_pixel = 300.0
    lambda_perceptual = 30.0
    lambda_feature = 10.0
    
    # Initialize the perceptual loss
    perceptual_loss = VGGPerceptualLoss(device)
    
    # Training history
    history = {
        'g_loss': [], 'd_loss': [], 'val_psnr': [], 'val_ssim': []
    }
    
    # Gradient clipping value
    clip_value = 0.5
    
    for epoch in range(num_epochs):
        # Training
        generator.train()
        discriminator.train()
        
        train_g_loss = 0.0
        train_d_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Get data
            low_imgs = batch['low'].to(device)
            normal_imgs = batch['normal'].to(device)
            batch_size = low_imgs.size(0)
            
            # Real and fake labels with label smoothing
            real_label = torch.ones((batch_size, 1, 30, 30), device=device) * 0.9
            fake_label = torch.zeros((batch_size, 1, 30, 30), device=device) + 0.1
            
            #------------------------
            # Train Discriminator
            #------------------------
            optimizer_D.zero_grad()
            
            # Generate enhanced images
            with torch.no_grad():
                enhanced_imgs = generator(low_imgs)
            
            # Real loss
            pred_real, real_features = discriminator(low_imgs, normal_imgs)
            d_real_loss = adversarial_loss(pred_real, real_label)
            
            # Fake loss
            pred_fake, _ = discriminator(low_imgs, enhanced_imgs.detach())
            d_fake_loss = adversarial_loss(pred_fake, fake_label)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            
            # Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
            
            optimizer_D.step()
            
            #------------------------
            # Train Generator
            #------------------------
            optimizer_G.zero_grad()
            
            # Generate enhanced images again
            enhanced_imgs = generator(low_imgs)
            
            # Adversarial loss
            pred_fake, fake_features = discriminator(low_imgs, enhanced_imgs)
            g_adv_loss = adversarial_loss(pred_fake, real_label)
            
            # Pixel-wise loss
            g_pixel_loss = l1_loss(enhanced_imgs, normal_imgs)
            
            # Perceptual loss
            g_percep_loss = perceptual_loss(enhanced_imgs, normal_imgs)
            
            # Feature matching loss
            g_feature_loss = 0.0
            for fake_feat, real_feat in zip(fake_features, real_features):
                g_feature_loss += torch.nn.functional.l1_loss(fake_feat, real_feat.detach())
            
            # Total generator loss
            g_loss = (lambda_adv * g_adv_loss + 
                     lambda_pixel * g_pixel_loss + 
                     lambda_perceptual * g_percep_loss +
                     lambda_feature * g_feature_loss)
            
            g_loss.backward()
            
            # Gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            
            optimizer_G.step()
            
            # Update learning rates with warmup
            if batch_idx < warmup_iters:
                warmup_scheduler_G.step()
                warmup_scheduler_D.step()
            
            # Save losses
            train_g_loss += g_loss.item()
            train_d_loss += d_loss.item()
        
        # Step the main schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Calculate average losses
        train_g_loss /= len(train_loader)
        train_d_loss /= len(train_loader)
        
        history['g_loss'].append(train_g_loss)
        history['d_loss'].append(train_d_loss)
        
        # Validation
        val_psnr, val_ssim = evaluate_model(generator, val_loader, device)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        
        print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
        
        # Save sample enhanced images
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, val_loader, epoch+1, device)
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'history': history
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
    
    return history

def evaluate_model(generator, data_loader, device):
    """Evaluate model with PSNR and SSIM metrics."""
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            low_imgs = batch['low'].to(device)
            normal_imgs = batch['normal'].to(device)
            
            # Generate enhanced images
            enhanced_imgs = generator(low_imgs)
            
            # Calculate metrics for each image in batch
            for i in range(low_imgs.size(0)):
                total_psnr += psnr(enhanced_imgs[i:i+1], normal_imgs[i:i+1])
                total_ssim += ssim(enhanced_imgs[i:i+1], normal_imgs[i:i+1])
    
    # Calculate average metrics
    avg_psnr = total_psnr / len(data_loader.dataset)
    avg_ssim = total_ssim / len(data_loader.dataset)
    
    return avg_psnr, avg_ssim

def main():
    # Set device
    if torch.backends.mps.is_available():
    # Apple Silicon Mac with Metal support
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for Apple Silicon Mac")
    elif torch.cuda.is_available():
        # Intel Mac with external GPU
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        print("Using CPU - training will be slower")
        
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create datasets and data loaders
    transform = get_transforms()
    
    train_dataset = LOLDataset(
        root_dir=os.path.join("data", "train"),
        transform=transform
    )
    
    val_dataset = LOLDataset(
        root_dir=os.path.join("data", "val"),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Train the model
    history = train_model(generator, discriminator, train_loader, val_loader, num_epochs=50, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, "checkpoints/final_model.pth")
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main() 