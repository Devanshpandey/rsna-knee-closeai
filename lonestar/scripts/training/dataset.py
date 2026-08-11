"""Study-level dataset over preprocessed uint8 series volumes.

Each item: (slices[S, K, 3, H, W], meta[S, 5], mask[S], targets[12])
  S = series (padded to max in batch by collate), K = slices per series.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
# under a horizontal mirror, medial<->lateral structures swap appearance
SWAP_IDX = list(range(len(LABELS)))
for a, b in [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]:
    ia, ib = LABELS.index(a), LABELS.index(b)
    SWAP_IDX[ia], SWAP_IDX[ib] = ib, ia


class KneeStudyDataset(Dataset):
    def __init__(self, folds_csv, series_csv, proc_dir, fold, split="train",
                 k_slices=16, max_series=6, crop=224, train_aug=True,
                 target_cols=None):
        self.proc = Path(proc_dir)
        folds = pd.read_csv(folds_csv)
        series = pd.read_csv(series_csv)

        if split == "train":
            keep = folds[(folds.fold != fold) & (folds.fold >= 0)]
        elif split == "val":
            keep = folds[folds.fold == fold]
        elif split == "gold":
            keep = folds[folds.is_gold == 1]
        else:
            raise ValueError(split)
        self.meta = keep.reset_index(drop=True)
        self.targets_cols = target_cols or [f"t_{l}" for l in LABELS]

        srs = series.groupby("StudyInstanceUID")
        self.series_map = {}
        missing = 0
        for uid in self.meta.StudyInstanceUID:
            rows = []
            for r in srs.get_group(uid).itertuples():
                p = self.proc / uid / f"{r.SeriesInstanceUID}.npy"
                if p.exists():
                    rows.append((str(p), PLANES.get(r.Anatomical_Plane, 0),
                                 int(r.Fluid_Sensitive), int(r.Fat_Suppression)))
                else:
                    missing += 1
            self.series_map[uid] = rows
        # drop studies with no preprocessed series yet
        have = self.meta.StudyInstanceUID.map(lambda u: len(self.series_map[u]) > 0)
        self.meta = self.meta[have.values].reset_index(drop=True)
        self.k = k_slices
        self.max_series = max_series
        self.crop = crop
        self.aug = train_aug and split == "train"
        self.split = split
        if missing:
            print(f"[dataset {split}] {missing} series npy missing; {len(self.meta)} studies usable")

    def __len__(self):
        return len(self.meta)

    def _load_series(self, path):
        vol = np.load(path, mmap_mode="r")  # (n, H, W) uint8
        n = vol.shape[0]
        # K positions spread over depth (jitter in train)
        pos = np.linspace(0, n - 1, self.k)
        if self.aug:
            pos = pos + np.random.uniform(-0.5, 0.5, self.k) * max((n - 1) / max(self.k - 1, 1), 1)
        pos = np.clip(np.round(pos).astype(int), 0, n - 1)
        idx3 = np.stack([np.clip(pos - 1, 0, n - 1), pos, np.clip(pos + 1, 0, n - 1)])  # (3, K)
        sl = np.ascontiguousarray(vol[idx3.reshape(-1)]).reshape(3, self.k, *vol.shape[1:])
        sl = np.transpose(sl, (1, 0, 2, 3)).astype(np.float32) / 255.0  # (K, 3, H, W)

        c, H, W = self.crop, sl.shape[2], sl.shape[3]
        if self.aug:
            y0 = random.randint(0, max(H - c, 0)); x0 = random.randint(0, max(W - c, 0))
        else:
            y0 = (H - c) // 2; x0 = (W - c) // 2
        sl = sl[:, :, y0:y0 + c, x0:x0 + c]
        if self.aug and random.random() < 0.5:
            sl = np.ascontiguousarray(sl[:, :, ::-1, :])          # vertical flip only (no L/R flip:
        return torch.from_numpy((sl - 0.45) / 0.225)              # would swap medial/lateral)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        srs = list(self.series_map[row.StudyInstanceUID])
        if len(srs) > self.max_series:
            srs = random.sample(srs, self.max_series) if self.aug else srs[:self.max_series]
        mirror = self.aug and random.random() < 0.5   # study-level mirror + label swap
        slices, metas = [], []
        for path, plane, fluid, fat in srs:
            sl = self._load_series(path)
            if mirror:
                sl = torch.flip(sl, dims=[-1])
            slices.append(sl)
            m = np.zeros(5, dtype=np.float32)
            m[plane] = 1.0; m[3] = fluid; m[4] = fat
            metas.append(m)
        t = row[self.targets_cols].values.astype(np.float32)
        if mirror:
            t = t[SWAP_IDX]
        return (torch.stack(slices), torch.from_numpy(np.stack(metas)),
                torch.tensor(t), row.StudyInstanceUID)


def collate(batch):
    smax = max(b[0].shape[0] for b in batch)
    B = len(batch)
    K, C, H, W = batch[0][0].shape[1:]
    x = torch.zeros(B, smax, K, C, H, W)
    meta = torch.zeros(B, smax, 5)
    mask = torch.zeros(B, smax)
    y = torch.stack([b[2] for b in batch])
    uids = [b[3] for b in batch]
    for i, b in enumerate(batch):
        s = b[0].shape[0]
        x[i, :s] = b[0]; meta[i, :s] = b[1]; mask[i, :s] = 1.0
    return x, meta, mask, y, uids
