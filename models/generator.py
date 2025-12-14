import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

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

class SAMLiteGeneratorFiLM(nn.Module):
    """
    U-Net + FiLM with AGE-GATED skip connections.
    Allows real geometric age change.
    """
    def __init__(self, base=64, age_emb_dim=128, age_min=0.0, age_max=80.0):
        super().__init__()
        self.age_min = age_min
        self.age_max = age_max

        self.age_emb = AgeEmbed(age_emb_dim)

        # encoder (age-agnostic)
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.InstanceNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.d1 = DownBlock(base, base*2)     # 128 -> 64
        self.d2 = DownBlock(base*2, base*4)   # 64  -> 32
        self.d3 = DownBlock(base*4, base*4)   # 32  -> 16

        # bottleneck (FiLM)
        self.b1 = ResBlockFiLM(base*4, age_emb_dim)
        self.b2 = ResBlockFiLM(base*4, age_emb_dim)

        # decoder
        self.u3 = UpBlock(base*4, base*4)
        self.r32 = ResBlockFiLM(base*4 + base*4, age_emb_dim)

        self.u2 = UpBlock(base*4 + base*4, base*2)
        self.r64 = ResBlockFiLM(base*2 + base*2, age_emb_dim)

        self.u1 = UpBlock(base*2 + base*2, base)
        self.r128 = ResBlockFiLM(base + base, age_emb_dim)

        self.out = nn.Sequential(
            nn.Conv2d(base + base, 3, 1),
            nn.Tanh()
        )

    def _norm_age(self, age):
        return (age - self.age_min) / (self.age_max - self.age_min + 1e-8)

    def _skip_gate(self, age_src, age_tgt):
        # age_src, age_tgt: (B,1)
        gap = (age_tgt - age_src).abs()      # years
        # 0y -> 1.0, 60y+ -> 0.2
        gate = 1.0 - (gap / 60.0).clamp(0, 1) * 0.8
        return gate.view(-1, 1, 1, 1)

    def forward(self, x, age_src, age_tgt):
        same_age = torch.allclose(age_src, age_tgt)
        # normalize target age
        age_tgt_n = self._norm_age(age_tgt)
        if same_age:
          age_emb = None
        else:
          age_emb = self.age_emb(age_tgt_n)


        # encode
        x0 = self.in_conv(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)

        # bottleneck
        h = self.b1(x3, age_emb)
        h = self.b2(h, age_emb)

        # skip gate
        if same_age:
          g = 1.0
        else:
          g = self._skip_gate(age_src, age_tgt)

        # decode with GATED skips
        h = self.u3(h)
        h = torch.cat([h, g * x2], dim=1)
        h = self.r32(h, age_emb)

        h = self.u2(h)
        h = torch.cat([h, g * x1], dim=1)
        h = self.r64(h, age_emb)

        h = self.u1(h)
        h = torch.cat([h, g * x0], dim=1)
        h = self.r128(h, age_emb)

        return self.out(h)