import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64, num_bins=8):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.c1 = block(in_ch, base, norm=False)     # 128 -> 64
        self.c2 = block(base, base*2)                # 64 -> 32
        self.c3 = block(base*2, base*4)              # 32 -> 16
        self.c4 = block(base*4, base*4, stride=1)    # 16 -> 16

        # patch realism output
        self.out_patch = nn.Conv2d(base*4, 1, 3, padding=1)

        # age classification head (AC-GAN)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_age = nn.Linear(base*4, num_bins)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)

        patch_logits = self.out_patch(h)          # (B,1,H,W)
        feat = self.gap(h).flatten(1)             # (B,C)
        age_logits = self.out_age(feat)           # (B,num_bins)
        return patch_logits, age_logits