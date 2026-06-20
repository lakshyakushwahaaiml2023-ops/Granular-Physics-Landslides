import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Standard block containing (Conv2d -> BatchNorm2d -> ReLU) x 2.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    A 2D U-Net architecture designed for predicting grid-based states.
    Input: (B, 4, H, W) - [density, velocity_x, velocity_y, terrain_mask]
    Output: (B, 3, H, W) - [predicted_density, predicted_vx, predicted_vy]
    """
    def __init__(self, in_channels: int = 5, out_channels: int = 4):
        super().__init__()
        
        # 1. Down-sampling path (Encoder)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(128, 256)
        )
        
        # 2. Up-sampling path (Decoder)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)  # concat: 128 + 128 = 256
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)   # concat: 64 + 64 = 128
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)    # concat: 32 + 32 = 64
        
        # Output projection
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # Simple verification shape check
    model = UNet2D()
    test_input = torch.randn(2, 5, 64, 128)
    test_output = model(test_input)
    print(f"U-Net test shape check:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    assert test_output.shape == (2, 4, 64, 128), "Dimension mismatch!"
    print("Shape check passed.")
