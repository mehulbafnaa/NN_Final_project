import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False, relu=True):
        super(UNetBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels) if bn else nn.Identity(),
            nn.GELU(),
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