# Low-Light Image Enhancement with Deep Learning
# Sean Huvaya

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torchvision.utils as vutils
from torch.nn import functional as F

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set up device for Mac
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

# 1. Dataset Preparation
class LOLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with LOL dataset structure
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Assuming LOL dataset structure with 'low' and 'normal' subdirectories
        self.low_light_dir = os.path.join(root_dir, 'low')
        self.normal_light_dir = os.path.join(root_dir, 'normal')
        
        # Get image file names (assuming same names in both directories)
        self.image_names = sorted(os.listdir(self.low_light_dir))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load low-light and normal-light images
        low_light_path = os.path.join(self.low_light_dir, img_name)
        normal_light_path = os.path.join(self.normal_light_dir, img_name)
        
        low_light_image = Image.open(low_light_path).convert('RGB')
        normal_light_image = Image.open(normal_light_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            low_light_image = self.transform(low_light_image)
            normal_light_image = self.transform(normal_light_image)
        
        return {'low': low_light_image, 'normal': normal_light_image}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Images normalized to [-1, 1] range for GAN training
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Model Architecture
# 2.1 U-Net Generator
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False, relu=True):
        super(UNetBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5) if dropout else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = UNetBlock(in_channels, features, down=True, bn=False, relu=False)  # 128x128
        self.down2 = UNetBlock(features, features*2, down=True)  # 64x64
        self.down3 = UNetBlock(features*2, features*4, down=True)  # 32x32
        self.down4 = UNetBlock(features*4, features*8, down=True)  # 16x16
        self.down5 = UNetBlock(features*8, features*8, down=True)  # 8x8
        self.down6 = UNetBlock(features*8, features*8, down=True)  # 4x4
        self.down7 = UNetBlock(features*8, features*8, down=True, bn=False)  # 2x2
        
        # Decoder (upsampling with skip connections)
        self.up1 = UNetBlock(features*8, features*8, down=False, dropout=True)  # 4x4
        self.up2 = UNetBlock(features*8*2, features*8, down=False, dropout=True)  # 8x8
        self.up3 = UNetBlock(features*8*2, features*8, down=False, dropout=True)  # 16x16
        self.up4 = UNetBlock(features*8*2, features*4, down=False)  # 32x32
        self.up5 = UNetBlock(features*4*2, features*2, down=False)  # 64x64
        self.up6 = UNetBlock(features*2*2, features, down=False)  # 128x128
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Decoder with skip connections
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        
        return self.final(torch.cat([u6, d1], 1))

# 2.2 PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Input: concatenated low-light and enhanced/normal image (6 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(features*4, features*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, True)
        )
        
        # Output: 1-channel prediction map
        self.final = nn.Conv2d(features*8, 1, 4, 1, 1)
    
    def forward(self, x, y):
        # x: low-light image, y: enhanced/normal image
        input_tensor = torch.cat([x, y], dim=1)
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        return x

# 3. Loss Functions
# 3.1 Adversarial Loss
adversarial_loss = nn.BCEWithLogitsLoss()

# 3.2 Perceptual Loss (using VGG16 features)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Updated for newer PyTorch versions and Mac compatibility
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        vgg = vgg.to(device).eval()
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # Normalize to match VGG input
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Convert from [-1, 1] to [0, 1] range
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Normalize
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Get feature maps
        x_features = [self.slice1(x), self.slice2(self.slice1(x)), 
                      self.slice3(self.slice2(self.slice1(x))), 
                      self.slice4(self.slice3(self.slice2(self.slice1(x))))]
        y_features = [self.slice1(y), self.slice2(self.slice1(y)), 
                      self.slice3(self.slice2(self.slice1(y))), 
                      self.slice4(self.slice3(self.slice2(self.slice1(y))))]
        
        # Calculate L1 loss on feature maps
        loss = 0
        weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
        for i in range(len(x_features)):
            loss += weights[i] * F.l1_loss(x_features[i], y_features[i])
        
        return loss

# 3.3 L1 Loss
l1_loss = nn.L1Loss()

# 4. Training Pipeline
def train_model(generator, discriminator, train_loader, val_loader, num_epochs=50):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)
    
    # Loss weights
    lambda_adv = 1.0
    lambda_pixel = 100.0
    lambda_perceptual = 10.0
    
    # Initialize the perceptual loss
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    # Training history
    history = {
        'g_loss': [], 'd_loss': [], 'val_psnr': [], 'val_ssim': []
    }
    
    for epoch in range(num_epochs):
        # Training
        generator.train()
        discriminator.train()
        
        train_g_loss = 0.0
        train_d_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get data
            low_imgs = batch['low'].to(device)
            normal_imgs = batch['normal'].to(device)
            batch_size = low_imgs.size(0)
            
            # Real and fake labels
            real_label = torch.ones((batch_size, 1, 30, 30), device=device)
            fake_label = torch.zeros((batch_size, 1, 30, 30), device=device)
            
            #------------------------
            # Train Discriminator
            #------------------------
            optimizer_D.zero_grad()
            
            # Generate enhanced images
            with torch.no_grad():
                enhanced_imgs = generator(low_imgs)
            
            # Real loss
            pred_real = discriminator(low_imgs, normal_imgs)
            d_real_loss = adversarial_loss(pred_real, real_label)
            
            # Fake loss
            pred_fake = discriminator(low_imgs, enhanced_imgs.detach())
            d_fake_loss = adversarial_loss(pred_fake, fake_label)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            #------------------------
            # Train Generator
            #------------------------
            optimizer_G.zero_grad()
            
            # Generate enhanced images again (needed because we detached earlier)
            enhanced_imgs = generator(low_imgs)
            
            # Adversarial loss
            pred_fake = discriminator(low_imgs, enhanced_imgs)
            g_adv_loss = adversarial_loss(pred_fake, real_label)
            
            # Pixel-wise loss
            g_pixel_loss = l1_loss(enhanced_imgs, normal_imgs)
            
            # Perceptual loss
            g_percep_loss = perceptual_loss(enhanced_imgs, normal_imgs)
            
            # Total generator loss
            g_loss = lambda_adv * g_adv_loss + lambda_pixel * g_pixel_loss + lambda_perceptual * g_percep_loss
            g_loss.backward()
            optimizer_G.step()
            
            # Save losses
            train_g_loss += g_loss.item()
            train_d_loss += d_loss.item()
        
        # Step the schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Calculate average losses
        train_g_loss /= len(train_loader)
        train_d_loss /= len(train_loader)
        
        history['g_loss'].append(train_g_loss)
        history['d_loss'].append(train_d_loss)
        
        # Validation
        val_psnr, val_ssim = evaluate_model(generator, val_loader)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        
        print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
        
        # Save sample enhanced images
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, val_loader, epoch+1)
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'history': history
            }, f"checkpoint_epoch_{epoch+1}.pth")
    
    return history

# 5. Evaluation Metrics
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

def evaluate_model(generator, data_loader):
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

# 6. Visualization
def save_sample_images(generator, data_loader, epoch):
    """Save sample enhanced images."""
    generator.eval()
    
    with torch.no_grad():
        batch = next(iter(data_loader))
        low_imgs = batch['low'].to(device)
        normal_imgs = batch['normal'].to(device)
        
        # Generate enhanced images
        enhanced_imgs = generator(low_imgs)
        
        # Select a few samples to display
        idx = min(5, low_imgs.size(0))
        low_imgs = low_imgs[:idx]
        normal_imgs = normal_imgs[:idx]
        enhanced_imgs = enhanced_imgs[:idx]
        
        # Create grid of images
        img_grid = torch.cat([low_imgs, enhanced_imgs, normal_imgs], dim=0)
        img_grid = vutils.make_grid(img_grid, nrow=idx, normalize=True)
        
        # Save grid
        vutils.save_image(img_grid, f"samples_epoch_{epoch}.png")
        
        # Display grid
        plt.figure(figsize=(15, 5))
        plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(f'Epoch {epoch} - Top: Low-light, Middle: Enhanced, Bottom: Normal-light')
        plt.savefig(f"grid_epoch_{epoch}.png")
        plt.close()

# 7. Inference on New Images
def enhance_image(generator, image_path, output_path=None):
    """Enhance a single low-light image using the trained generator."""
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
    
    # Save or display enhanced image
    if output_path:
        enhanced_image.save(output_path)
    
    # Display original and enhanced images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Low-light Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image)
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return enhanced_image

# 8. Main execution
def main():
    # Mac-friendly path handling
    import os.path
    
    # Paths to your dataset - using os.path.expanduser for Mac home directory support
    data_root = os.path.expanduser("~/Downloads/LOL_dataset")  # Adjust as needed
    
    # Check if dataset paths exist
    if not os.path.exists(data_root):
        print(f"Dataset path {data_root} doesn't exist. Please update the path.")
        print("If you haven't downloaded the LOL dataset yet, you can find it at:")
        print("https://daooshee.github.io/BMVC2018website/")
        return
    
    # Create datasets and data loaders
    train_dataset = LOLDataset(
        root_dir=os.path.join(data_root, 'train'),
        transform=transform
    )
    
    val_dataset = LOLDataset(
        root_dir=os.path.join(data_root, 'val'),
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
    
    # Mac optimization: Set num_workers properly for DataLoader
    # For Mac, using too many workers can cause issues
    num_workers = 0 if device.type == 'mps' else 2
    
    # Update DataLoader configurations
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Train the model
    history = train_model(generator, discriminator, train_loader, val_loader, num_epochs=50)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_psnr'], label='PSNR')
    plt.plot(history['val_ssim'], label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Save final model
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, "final_model.pth")
    
    print("Training completed and model saved.")
    
    # Test on a new image (if available)
    test_image_path = "path/to/test_image.jpg"  # Replace with actual path
    if os.path.exists(test_image_path):
        enhance_image(generator, test_image_path, "enhanced_output.jpg")

if __name__ == "__main__":
    main()
