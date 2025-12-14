import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FiLM(nn.Module):
    """Produces (gamma, beta) for FiLM modulation."""
    def __init__(self, age_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(age_dim, channels * 2)

    def forward(self, age_emb):
        gb = self.to_gb(age_emb)                  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)          # (B, C), (B, C)
        # stabilize: start near identity transform
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)
        return gamma, beta

class ResBlockFiLM(nn.Module):
    def __init__(self, channels: int, age_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)
        self.film1 = FiLM(age_dim, channels)
        self.film2 = FiLM(age_dim, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, age_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        if age_emb is not None:
            g, b = self.film1(age_emb)
            h = h * g.view(-1, h.size(1), 1, 1) + b.view(-1, h.size(1), 1, 1)

        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)

        if age_emb is not None:
            g, b = self.film2(age_emb)
            h = h * g.view(-1, h.size(1), 1, 1) + b.view(-1, h.size(1), 1, 1)

        return self.act(x + h)

class AgeEmbed(nn.Module):
    """Age scalar -> learned embedding"""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, emb_dim), nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, age_norm):
        return self.net(age_norm)

class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 backbone returning multi-scale feature maps
    for U-Net-style decoding.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.resnet18(weights="DEFAULT" if pretrained else None)

        # Input conv (keep stride=2)
        self.conv1 = m.conv1     # 64, stride 2
        self.bn1   = m.bn1
        self.relu  = m.relu
        self.maxpool = m.maxpool # stride 2

        # ResNet stages
        self.layer1 = m.layer1   # 64   (32×32)
        self.layer2 = m.layer2   # 128  (16×16)
        self.layer3 = m.layer3   # 256  (8×8)
        self.layer4 = m.layer4   # 512  (4×4)

    def forward(self, x):
        feats = {}

        x = self.conv1(x)   # 64, 64×64
        x = self.bn1(x)
        x = self.relu(x)
        feats["c1"] = x

        x = self.maxpool(x) # 64, 32×32
        x = self.layer1(x)  # 64, 32×32
        feats["c2"] = x

        x = self.layer2(x)  # 128, 16×16
        feats["c3"] = x

        x = self.layer3(x)  # 256, 8×8
        feats["c4"] = x

        x = self.layer4(x)  # 512, 4×4
        feats["c5"] = x

        return feats

class SAMResNetFiLMGenerator(nn.Module):
    """
    ResNet18 encoder + FiLM bottleneck + UNet-style decoder.
    Landmark head is attached to mid-level (32x32) feature to supervise geometry.
    Returns:
      img: (B,3,128,128) in [-1,1]
      lm : (B,68,2) in [-1,1]
    """
    def __init__(self, age_emb_dim=128):
        super().__init__()

        # --- age embedding ---
        self.age_emb = AgeEmbed(age_emb_dim)

        # --- encoder ---
        self.enc = ResNet18Encoder(pretrained=True)

        # --- bottleneck FiLM blocks ---
        self.b1 = ResBlockFiLM(512, age_emb_dim)
        self.b2 = ResBlockFiLM(512, age_emb_dim)

        # --- learnable skip scales ---
        self.skip_w_c4 = nn.Parameter(torch.tensor(1.0))  # for c4 (8x8)
        self.skip_w_c3 = nn.Parameter(torch.tensor(1.0))  # for c3 (16x16)
        self.skip_w_c2 = nn.Parameter(torch.tensor(1.0))  # for c2 (32x32)
        self.skip_w_c1 = nn.Parameter(torch.tensor(1.0))  # for c1 (64x64)

        # --- decoder ---
        self.up4 = UpBlock(512, 256)            # 4 -> 8
        self.r8  = ResBlockFiLM(256 + 256, age_emb_dim)

        self.up3 = UpBlock(256 + 256, 128)      # 8 -> 16
        self.r16 = ResBlockFiLM(128 + 128, age_emb_dim)

        self.up2 = UpBlock(128 + 128, 64)       # 16 -> 32
        self.r32 = ResBlockFiLM(64 + 64, age_emb_dim)

        self.up1 = UpBlock(64 + 64, 64)         # 32 -> 64
        self.r64 = ResBlockFiLM(64 + 64, age_emb_dim)

        self.up0 = UpBlock(64 + 64, 64)         # 64 -> 128

        # --- image head ---
        self.out_img = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

        # --- landmark head ---
        self.lm_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lm_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 136),
            nn.Tanh()   # landmarks in [-1,1]
        )

    def forward(self, x, age_src, age_tgt, age_min=0.0, age_max=80.0):
        # normalize age to [0,1]
        age = (age_tgt - age_min) / (age_max - age_min + 1e-8)  # (B,1)
        age_emb = self.age_emb(age)

        # --- encode ---
        feats = self.enc(x)
        h = feats["c5"]                    # (B,512,4,4)

        # --- bottleneck ---
        h = self.b1(h, age_emb)
        h = self.b2(h, age_emb)

        # --- decode ---
        h = self.up4(h)                    # -> (B,256,8,8)
        h = torch.cat([h, self.skip_w_c4 * feats["c4"]], dim=1)
        h = self.r8(h, age_emb)            # -> (B,512,8,8)

        h = self.up3(h)                    # -> (B,128,16,16)
        h = torch.cat([h, self.skip_w_c3 * feats["c3"]], dim=1)
        h = self.r16(h, age_emb)           # -> (B,256,16,16)

        h = self.up2(h)                    # -> (B,64,32,32)
        h = torch.cat([h, self.skip_w_c2 * feats["c2"]], dim=1)
        h = self.r32(h, age_emb)           # -> (B,128,32,32)

        # ===== landmark prediction from MID-LEVEL structural feature =====
        lm_feat = self.lm_conv(h)          # (B,64,32,32)
        lm = self.lm_fc(lm_feat).view(-1, 68, 2)  # (B,68,2) in [-1,1]

        h = self.up1(h)                    # -> (B,64,64,64)
        h = torch.cat([h, self.skip_w_c1 * feats["c1"]], dim=1)
        h = self.r64(h, age_emb)           # -> (B,128,64,64)

        h = self.up0(h)                    # -> (B,64,128,128)
        img = self.out_img(h)              # -> (B,3,128,128)

        return img, lm