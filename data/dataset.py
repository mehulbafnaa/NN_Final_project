import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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

# Default image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Images normalized to [-1, 1] range for GAN training
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 