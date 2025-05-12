import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
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
        
        self.device = device
    
    def forward(self, x, y):
        # Normalize to match VGG input
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
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

# Adversarial loss
adversarial_loss = nn.BCEWithLogitsLoss()

# L1 Loss
l1_loss = nn.L1Loss() 