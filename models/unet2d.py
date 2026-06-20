import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EncBlock(nn.Module):
    """
    Encoder (double-conv) block using Conv2d, GroupNorm, and GELU.
    Kernel size 7 and padding 3 preserves spatial height and width.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    """
    Decoder block: upsamples, fuses skip connections from encoder, and performs double-conv.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = EncBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)  # Upsample 2x in height and width
        
        # Interpolate if shape mismatches by 1 pixel
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net 2D
# ---------------------------------------------------------------------------

class UNet2D(nn.Module):
    """
    2D U-Net surrogate model for granular landslide state transition prediction.
    Input shape: (B, 5, H, W) - [density, vx, vy, KE, terrain_occupancy]
    Output shape: (B, 4, H, W) - [predicted_density, predicted_vx, predicted_vy, predicted_KE]
    
    Residual formulation:
        output = input[:, :4, :, :] + raw_network_output
    """
    def __init__(self) -> None:
        super().__init__()
        
        # Encoder (skip connections saved BEFORE pooling)
        self.enc1 = EncBlock(5, 32)
        self.enc2 = EncBlock(32, 64)
        self.enc3 = EncBlock(64, 128)
        self.enc4 = EncBlock(128, 256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = EncBlock(256, 512)
        
        # Decoder
        self.dec4 = DecBlock(512, 256, 256)
        self.dec3 = DecBlock(256, 128, 128)
        self.dec2 = DecBlock(128, 64, 64)
        self.dec1 = DecBlock(64, 32, 32)
        
        # Output Head
        self.head = nn.Conv2d(32, 4, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes Conv2d/ConvTranspose2d weights with Kaiming normal.
        Initializes the output head weight and bias to zero for stable early training.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # Zero-init output head so model outputs near-identity initially (delta ≈ 0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_in = x[:, :4, :, :]  # Keep initial density, vx, vy, KE for residual connection
        
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(s4))
        
        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        
        delta = self.head(d1)
        return state_in + delta


# ---------------------------------------------------------------------------
# Physics-aware loss
# ---------------------------------------------------------------------------

class GranularPhysicsLoss(nn.Module):
    """
    Physics-aware loss function balancing standard grid fidelity (L1 loss)
    with physical constraints: mass conservation, momentum conservation, and spatial smoothness.
    """
    def __init__(self, dx=4.0/128, dy=3.0/64,
                 lambda_mass=2.0, lambda_momentum=1.0,
                 lambda_l1=1.0, lambda_smooth=0.5):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.lambda_mass = lambda_mass
        self.lambda_momentum = lambda_momentum
        self.lambda_l1 = lambda_l1
        self.lambda_smooth = lambda_smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, input_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # 1. Fidelity L1 Loss
        l1 = F.l1_loss(pred, target)
        
        # 2. Mass Conservation Loss (Sum of density channel)
        pred_mass = pred[:, 0, :, :].sum(dim=(-1, -2))
        target_mass = target[:, 0, :, :].sum(dim=(-1, -2))
        mass = F.mse_loss(pred_mass, target_mass)
        
        # 3. Momentum Conservation Loss (x-momentum component conservation)
        pred_px = (pred[:, 0] * pred[:, 1]).sum(dim=(-1, -2))
        target_px = (target[:, 0] * target[:, 1]).sum(dim=(-1, -2))
        momentum = F.mse_loss(pred_px, target_px)
        
        # 4. Smoothness Loss (Total variation regularization on predicted density)
        dx_grad = pred[:, 0, :, 1:] - pred[:, 0, :, :-1]
        dy_grad = pred[:, 0, 1:, :] - pred[:, 0, :-1, :]
        smooth = dx_grad.abs().mean() + dy_grad.abs().mean()
        
        # 5. Combined loss
        total = (self.lambda_l1 * l1 + 
                 self.lambda_mass * mass + 
                 self.lambda_momentum * momentum + 
                 self.lambda_smooth * smooth)
        
        return {
            'total': total,
            'l1': l1,
            'mass': mass,
            'momentum': momentum,
            'smooth': smooth
        }


# ---------------------------------------------------------------------------
# Runout Distance Predictor
# ---------------------------------------------------------------------------

class RunoutPredictor(nn.Module):
    """
    CNN regressor to predict final scalar runout distance directly from initial state.
    Provides fast inference (<1ms) compared to multi-minute physical DEM simulations.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Input size (5, 64, 128) -> output (32, 32, 64)
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 2: (32, 32, 64) -> (64, 16, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 3: (64, 16, 32) -> (128, 8, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 4: (128, 8, 16) -> (256, 1, 1)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Test Verification
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = UNet2D()
    x = torch.randn(2, 5, 64, 128)
    out = model(x)
    print(f"UNet2D output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    rp = RunoutPredictor()
    runout = rp(x)
    print(f"RunoutPredictor output: {runout.shape}")
    print(f"Parameters: {sum(p.numel() for p in rp.parameters()):,}")
