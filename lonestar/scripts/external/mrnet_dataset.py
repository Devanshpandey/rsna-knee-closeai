"""MRNet (Stanford) adapter producing the same tensors as KneeStudyDataset.

MRNet layout: <root>/{train,valid}/{axial,coronal,sagittal}/XXXX.npy  (s,256,256) uint8
Labels: <root>/{train,valid}-{abnormal,acl,meniscus}.csv  (id,label) no header.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MRNET_TASKS = ["abnormal", "acl", "meniscus"]
# (plane_onehot_index, fluid_sensitive, fat_sat) per MRNet plane — approximate protocol mapping
PLANE_META = {"sagittal": (0, 1, 1), "coronal": (1, 0, 0), "axial": (2, 1, 1)}


class MRNetDataset(Dataset):
    def __init__(self, root, split="train", k_slices=20, crop=224, train_aug=True):
        self.root = Path(root)
        self.split = split
        labels = {}
        for t in MRNET_TASKS:
            df = pd.read_csv(self.root / f"{split}-{t}.csv", header=None, names=["id", t],
                             dtype={"id": str})
            labels[t] = df.set_index("id")[t]
        self.labels = pd.DataFrame(labels)
        self.ids = sorted(self.labels.index)
        self.k = k_slices
        self.crop = crop
        self.aug = train_aug and split == "train"

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        vol = np.load(path)  # (s, 256, 256) uint8
        n = vol.shape[0]
        pos = np.linspace(0, n - 1, self.k)
        if self.aug:
            pos = pos + np.random.uniform(-0.5, 0.5, self.k) * max((n - 1) / max(self.k - 1, 1), 1)
        pos = np.clip(np.round(pos).astype(int), 0, n - 1)
        idx3 = np.stack([np.clip(pos - 1, 0, n - 1), pos, np.clip(pos + 1, 0, n - 1)])
        sl = vol[idx3.reshape(-1)].reshape(3, self.k, 256, 256)
        sl = np.transpose(sl, (1, 0, 2, 3)).astype(np.float32) / 255.0
        c = self.crop
        if self.aug:
            y0 = random.randint(0, 256 - c); x0 = random.randint(0, 256 - c)
        else:
            y0 = x0 = (256 - c) // 2
        sl = sl[:, :, y0:y0 + c, x0:x0 + c]
        if self.aug and random.random() < 0.5:
            sl = np.ascontiguousarray(sl[:, :, ::-1, :])   # vertical flip
        return torch.from_numpy((sl - 0.45) / 0.225)

    def __getitem__(self, i):
        eid = self.ids[i]
        mirror = self.aug and random.random() < 0.5      # MRNet labels are side-agnostic -> safe
        slices, metas = [], []
        for plane, (pidx, fluid, fat) in PLANE_META.items():
            sl = self._load(self.root / self.split / plane / f"{eid}.npy")
            if mirror:
                sl = torch.flip(sl, dims=[-1])
            slices.append(sl)
            m = np.zeros(5, dtype=np.float32)
            m[pidx] = 1.0; m[3] = fluid; m[4] = fat
            metas.append(m)
        y = self.labels.loc[eid].values.astype(np.float32)
        return (torch.stack(slices), torch.from_numpy(np.stack(metas)),
                torch.tensor(y), eid)
