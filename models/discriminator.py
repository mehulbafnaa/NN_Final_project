import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Input: concatenated low-light and enhanced/normal image (6 channels)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, features, 4, 2, 1)),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(features, features*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features*2),
            nn.GELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(features*4),
            nn.GELU()
        )
        
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(features*4, features*8, 4, 1, 1, bias=False)),
            nn.BatchNorm2d(features*8),
            nn.GELU()
        )
        
        # Output: 1-channel prediction map
        self.final = nn.utils.spectral_norm(nn.Conv2d(features*8, 1, 4, 1, 1))
    
    def forward(self, x, y):
        # x: low-light image, y: enhanced/normal image
        input_tensor = torch.cat([x, y], dim=1)
        x1 = self.conv1(input_tensor)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.final(x4)
        return x, [x1, x2, x3, x4] 