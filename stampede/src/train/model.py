"""Stage-1 model: 2.5D series classifier with slice-position awareness.

Input: [B, K, 3, H, W] slice-triplets + series metadata + per-slice normalized
positions. Positions get Fourier features projected and added to each slice
embedding BEFORE attention pooling, so the model can localize along the stack
(medial vs lateral for sagittal series, etc.).
"""
import math

import timm
import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x):             # x: [B, K, D]
        a = self.w(x).softmax(dim=1)  # [B, K, 1]
        return (x * a).sum(1), a


class PosEnc(nn.Module):
    """Fourier features of normalized slice position -> D-dim additive code."""
    def __init__(self, dim, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2 ** torch.arange(n_freq) * math.pi)
        self.proj = nn.Linear(2 * n_freq + 1, dim)

    def forward(self, pos):           # pos: [B, K] in [0, 1]
        f = pos.unsqueeze(-1) * self.freqs        # [B, K, F]
        feats = torch.cat([pos.unsqueeze(-1), f.sin(), f.cos()], dim=-1)
        return self.proj(feats)       # [B, K, D]


class SeriesModel(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_s.in21k_ft_in1k", n_labels=12,
                 pretrained=True, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, drop_rate=drop)
        d = self.backbone.num_features
        self.pos_enc = PosEnc(d)
        self.pool = AttnPool(d)
        self.meta_emb = nn.ModuleList([nn.Embedding(3, 16), nn.Embedding(2, 8), nn.Embedding(2, 8)])
        self.head = nn.Sequential(
            nn.Linear(d + 32, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, n_labels))

    def forward(self, x, meta, pos):
        # x: [B, K, 3, H, W]; meta: [B, 3] ints; pos: [B, K] floats
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, K, -1)
        f = f + self.pos_enc(pos)
        pooled, attn = self.pool(f)
        m = torch.cat([emb(meta[:, i]) for i, emb in enumerate(self.meta_emb)], dim=1)
        return self.head(torch.cat([pooled, m], dim=1)), attn
