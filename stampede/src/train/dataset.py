"""Datasets over preprocessed npz series volumes (canonical LEFT-knee space).

Stage 1 trains a per-series 2.5D classifier: each series inherits its study's
12 soft labels (MIL-style weak supervision) with a per-label weight vector.

Review fixes applied: NaN label guard; the old slice-order "flip" (a no-op for
attention pooling, and anatomically wrong for sagittal) is replaced by a
proper mirror flip along the patient-x axis WITH swapping of the medial/
lateral label pairs; per-slice normalized positions are returned so the model
can localize along the stack; sample weight folds in 1/n_series.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
# indices into LABELS for medial<->lateral swap pairs (Medial/Lateral Meniscus, Medial/Lateral OA)
SIDE_SWAP_PAIRS = [(2, 3), (4, 5)]


def slice_indices(n: int, k: int, jitter: bool) -> np.ndarray:
    if n <= k:
        idx = np.arange(n)
        idx = np.pad(idx, (0, k - n), mode="edge")
        return idx
    pos = np.linspace(0, n - 1, k)
    if jitter:
        step = (n - 1) / max(k - 1, 1)
        pos = pos + np.random.uniform(-step / 2, step / 2, size=k)
    return np.clip(pos.round().astype(int), 0, n - 1)


class SeriesDataset(Dataset):
    def __init__(self, series_df: pd.DataFrame, labels_df: pd.DataFrame,
                 label_cols: list, npz_root: str, k_slices: int = 16,
                 train: bool = True, weight_cols: list | None = None,
                 mirror_prob: float = 0.5):
        m = series_df.merge(labels_df, on="StudyInstanceUID", how="inner")
        assert not m[label_cols].isna().any().any(), "NaN labels reached the dataset"
        if "sample_weight" not in m.columns:
            m["sample_weight"] = 1.0
        # study gradient weight independent of series count
        m["sample_weight"] = m["sample_weight"] / m.groupby(
            "StudyInstanceUID")["SeriesInstanceUID"].transform("count")
        self.df = m.reset_index(drop=True)
        self.label_cols = label_cols
        self.weight_cols = weight_cols  # optional per-label weight columns
        self.root = Path(npz_root)
        self.k = k_slices
        self.train = train
        self.mirror_prob = mirror_prob

    def __len__(self):
        return len(self.df)

    def _load_vol(self, study, series):
        per_series = self.root / study / f"{series}.npz"
        if per_series.exists():  # local layout: npz_root/<study>/<series>.npz
            with np.load(per_series) as z:
                return z["vol"]
        with np.load(self.root / f"{study}.npz") as z:  # study-bundle layout
            return z[f"vol_{series}"]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = self._load_vol(r.StudyInstanceUID, r.SeriesInstanceUID)  # [n,H,W] uint8, canonical left
        y = r[self.label_cols].values.astype(np.float32).copy()
        w = (r[self.weight_cols].values.astype(np.float32).copy()
             if self.weight_cols else np.ones(len(self.label_cols), dtype=np.float32))
        w *= np.float32(r.get("sample_weight", 1.0))
        plane = PLANES.get(r.Anatomical_Plane, 0)

        # mirror augmentation along patient-x with medial<->lateral label swap.
        # canonical volumes: sagittal x-axis = slice axis; coronal/axial x-axis = image width
        if self.train and np.random.rand() < self.mirror_prob:
            if plane == 0:
                vol = vol[::-1]
            else:
                vol = vol[:, :, ::-1]
            for a, b in SIDE_SWAP_PAIRS:
                y[a], y[b] = y[b], y[a]
                w[a], w[b] = w[b], w[a]

        n = vol.shape[0]
        centers = slice_indices(n, self.k, jitter=self.train)
        lo = np.clip(centers - 1, 0, n - 1)
        hi = np.clip(centers + 1, 0, n - 1)
        x = np.stack([vol[lo], vol[centers], vol[hi]], axis=1).astype(np.float32) / 255.0
        pos = (centers / max(n - 1, 1)).astype(np.float32)

        if self.train and np.random.rand() < 0.3:
            x = np.clip(x * np.random.uniform(0.85, 1.15)
                        + np.random.uniform(-0.05, 0.05), 0, 1)

        meta = np.array([plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression)],
                        dtype=np.int64)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(meta),
                torch.from_numpy(pos),
                torch.from_numpy(y),
                torch.from_numpy(w))
