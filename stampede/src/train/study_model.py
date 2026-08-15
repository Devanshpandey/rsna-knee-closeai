"""Study-level MIL model — devansh's architecture (team_repo lonestar/scripts/
training/model.py, verbatim): shared 2D backbone per slice -> slice attention
pool per series -> +series metadata -> series attention pool -> 12-label head.
Ported to Stampede3 to train under our 384px recipe (mirror+label-swap, EMA)."""
import timm
import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None):
        # x: (B, N, D); mask: (B, N) 1=valid
        a = self.net(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(mask < 0.5, -1e4)
        w = a.softmax(-1).unsqueeze(-1)
        return (x * w).sum(1)


class KneeStudyModel(nn.Module):
    def __init__(self, backbone="convnext_small.fb_in22k_ft_in1k", n_labels=12,
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.slice_pool = AttnPool(d)
        self.meta_proj = nn.Linear(5, 64)
        self.series_proj = nn.Sequential(nn.Linear(d + 64, d), nn.GELU())
        self.series_pool = AttnPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, n_labels))

    def forward(self, x, meta, mask):
        # x: (B, S, K, 3, H, W), meta: (B, S, 5), mask: (B, S)
        B, S, K = x.shape[:3]
        feats = self.backbone(x.flatten(0, 2))            # (B*S*K, D)
        feats = feats.view(B * S, K, -1)
        series_emb = self.slice_pool(feats)               # (B*S, D)
        series_emb = series_emb.view(B, S, -1)
        m = self.meta_proj(meta)
        series_emb = self.series_proj(torch.cat([series_emb, m], -1))
        study = self.series_pool(series_emb, mask)        # (B, D)
        return self.head(study)
