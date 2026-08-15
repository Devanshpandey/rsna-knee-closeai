"""Study-level dataset over our per-study npz bundles (canonical LEFT space).

One item = one STUDY: all its series (subsampled to max_series), K slice-
triplets each, per-series metadata [plane-onehot(3), fluid, fatsat]. Uses OUR
proven conventions from dataset.py: [0,1] floats, slice jitter, brightness/
contrast aug, and the plane-aware patient-x mirror WITH medial<->lateral label
swap (sagittal mirrors along the slice axis, coronal/axial along image width).
collate() pads series count with a mask for the MIL attention pool.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from dataset import PLANES, SIDE_SWAP_PAIRS, slice_indices


class StudyDataset(Dataset):
    def __init__(self, series_df: pd.DataFrame, labels_df: pd.DataFrame,
                 label_cols: list, npz_root: str, k_slices: int = 16,
                 max_series: int = 6, train: bool = True,
                 weight_cols: list | None = None, mirror_prob: float = 0.5):
        m = series_df.merge(labels_df, on="StudyInstanceUID", how="inner")
        assert not m[label_cols].isna().any().any(), "NaN labels reached the dataset"
        self.series_map = {}
        for r in m.itertuples():
            self.series_map.setdefault(r.StudyInstanceUID, []).append(
                (r.SeriesInstanceUID, PLANES.get(r.Anatomical_Plane, 0),
                 int(r.Fluid_Sensitive), int(r.Fat_Suppression)))
        self.df = (m[["StudyInstanceUID"] + label_cols +
                     (weight_cols or [])].drop_duplicates("StudyInstanceUID")
                   .reset_index(drop=True))
        self.label_cols = label_cols
        self.weight_cols = weight_cols
        self.root = Path(npz_root)
        self.k = k_slices
        self.max_series = max_series
        self.train = train
        self.mirror_prob = mirror_prob

    def __len__(self):
        return len(self.df)

    def _slices(self, vol):
        n = vol.shape[0]
        centers = slice_indices(n, self.k, jitter=self.train)
        lo = np.clip(centers - 1, 0, n - 1)
        hi = np.clip(centers + 1, 0, n - 1)
        return np.stack([vol[lo], vol[centers], vol[hi]], axis=1)  # [K,3,H,W] uint8

    def __getitem__(self, i):
        r = self.df.iloc[i]
        uid = r.StudyInstanceUID
        srs = sorted(self.series_map[uid])
        if len(srs) > self.max_series:
            if self.train:
                idx = np.random.choice(len(srs), self.max_series, replace=False)
                srs = [srs[j] for j in sorted(idx)]
            else:
                srs = srs[: self.max_series]
        y = r[self.label_cols].values.astype(np.float32).copy()
        w = (r[self.weight_cols].values.astype(np.float32).copy()
             if self.weight_cols else np.ones(len(self.label_cols), dtype=np.float32))
        mirror = self.train and np.random.rand() < self.mirror_prob

        xs, metas = [], []
        with np.load(self.root / f"{uid}.npz") as z:
            for sid, plane, fluid, fat in srs:
                vol = z[f"vol_{sid}"]
                if mirror:
                    vol = vol[::-1] if plane == 0 else vol[:, :, ::-1]
                x = self._slices(np.ascontiguousarray(vol)).astype(np.float32) / 255.0
                if self.train and np.random.rand() < 0.3:
                    x = np.clip(x * np.random.uniform(0.85, 1.15)
                                + np.random.uniform(-0.05, 0.05), 0, 1)
                xs.append(torch.from_numpy(x))
                mv = np.zeros(5, dtype=np.float32)
                mv[plane] = 1.0; mv[3] = fluid; mv[4] = fat
                metas.append(mv)
        if mirror:
            for a, b in SIDE_SWAP_PAIRS:
                y[a], y[b] = y[b], y[a]
                w[a], w[b] = w[b], w[a]
        return (torch.stack(xs), torch.from_numpy(np.stack(metas)),
                torch.from_numpy(y), torch.from_numpy(w), uid)


def collate(batch):
    smax = max(b[0].shape[0] for b in batch)
    B = len(batch)
    K, C, H, W = batch[0][0].shape[1:]
    x = torch.zeros(B, smax, K, C, H, W)
    meta = torch.zeros(B, smax, 5)
    mask = torch.zeros(B, smax)
    y = torch.stack([b[2] for b in batch])
    w = torch.stack([b[3] for b in batch])
    uids = [b[4] for b in batch]
    for i, b in enumerate(batch):
        s = b[0].shape[0]
        x[i, :s] = b[0]; meta[i, :s] = b[1]; mask[i, :s] = 1.0
    return x, meta, mask, y, w, uids
