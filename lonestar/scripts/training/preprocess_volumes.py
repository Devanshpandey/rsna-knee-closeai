#!/usr/bin/env python3
"""Phase 2 preprocessing: DICOM series -> normalized uint8 volumes (.npy) + metadata table.

Per series: decode slices, sort by position, percentile-clip intensities, resize in-plane,
save (n_slices, H, W) uint8. Parallel over series with multiprocessing.

Usage (TACC):
  python preprocess_volumes.py --dicom-root data/raw/train_series \
      --series-csv data/csv/train_series.csv --out-dir data/proc/train --size 256 --workers 16
"""
import argparse
import json
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError as e:
    raise SystemExit("pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg") from e


def load_series(series_dir: Path) -> np.ndarray:
    """Decode and sort one series into (n, H, W) float32."""
    slices = []
    for f in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(str(f))
        pos = getattr(ds, "ImagePositionPatient", None)
        normal_dot = None
        if pos is not None and hasattr(ds, "ImageOrientationPatient"):
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            normal_dot = float(np.dot(normal, np.array(pos, dtype=float)))
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        arr = ds.pixel_array.astype(np.float32)
        spacing = getattr(ds, "PixelSpacing", None)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        slices.append((normal_dot, inst, arr * slope + intercept, spacing))
    if not slices:
        raise ValueError("empty series")
    # sort by projection along slice normal when available, else InstanceNumber
    if all(s[0] is not None for s in slices):
        slices.sort(key=lambda s: s[0])
    else:
        slices.sort(key=lambda s: s[1])
    shapes = {s[2].shape for s in slices}
    if len(shapes) > 1:  # rare mixed-shape series: resize everything to the modal shape
        from collections import Counter
        target = Counter(s[2].shape for s in slices).most_common(1)[0][0]
        import cv2
        slices = [(a, b, cv2.resize(img, (target[1], target[0])), sp) for a, b, img, sp in slices]
    vol = np.stack([s[2] for s in slices])
    spacing = next((s[3] for s in slices if s[3] is not None), None)
    return vol, spacing


def normalize_resize(vol: np.ndarray, size: int, spacing=None, fov_mm: float = 0) -> np.ndarray:
    """Optional physical center-crop to fov_mm, percentile clip -> uint8, resize to (size,size)."""
    import cv2
    if fov_mm and spacing is not None:
        try:
            py, px = float(spacing[0]), float(spacing[1])
            ch = min(vol.shape[1], max(32, int(round(fov_mm / py))))
            cw = min(vol.shape[2], max(32, int(round(fov_mm / px))))
            y0 = (vol.shape[1] - ch) // 2
            x0 = (vol.shape[2] - cw) // 2
            vol = vol[:, y0:y0 + ch, x0:x0 + cw]
        except (TypeError, ValueError, IndexError):
            pass
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / max(hi - lo, 1e-6) * 255.0
    out = np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in vol])
    return out.astype(np.uint8)


def process_one(task):
    study_uid, series_uid, dicom_root, out_dir, size, fov_mm, series_path = task
    out_path = Path(out_dir) / study_uid / f"{series_uid}.npy"
    try:
        if out_path.exists():
            v = np.load(out_path, mmap_mode="r")
            return dict(StudyInstanceUID=study_uid, SeriesInstanceUID=series_uid,
                        n_slices=v.shape[0], h=v.shape[1], w=v.shape[2], status="cached")
        src = Path(series_path) if series_path else Path(dicom_root) / study_uid / series_uid
        vol, spacing = load_series(src)
        n_raw_h, n_raw_w = vol.shape[1], vol.shape[2]
        vol = normalize_resize(vol, size, spacing=spacing, fov_mm=fov_mm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vol)
        return dict(StudyInstanceUID=study_uid, SeriesInstanceUID=series_uid,
                    n_slices=vol.shape[0], h=vol.shape[1], w=vol.shape[2],
                    raw_h=n_raw_h, raw_w=n_raw_w, status="ok")
    except Exception:
        return dict(StudyInstanceUID=study_uid, SeriesInstanceUID=series_uid,
                    status="error", error=traceback.format_exc(limit=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fov-mm", type=float, default=0, help="physical center crop, e.g. 140")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    series = pd.read_csv(args.series_csv)
    if args.limit:
        series = series.head(args.limit)
    has_path = "series_path" in series.columns
    tasks = [(r.StudyInstanceUID, r.SeriesInstanceUID, args.dicom_root, args.out_dir, args.size, args.fov_mm,
              getattr(r, "series_path", None) if has_path else None)
             for r in series.itertuples()]
    print(f"{len(tasks)} series -> {args.out_dir}", flush=True)

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, tasks, chunksize=8)):
            results.append(res)
            if (i + 1) % 500 == 0:
                n_err = sum(1 for r in results if r["status"] == "error")
                print(f"{i+1}/{len(tasks)} done, {n_err} errors", flush=True)

    meta = pd.DataFrame(results)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "preprocess_meta.csv", index=False)
    n_err = int((meta["status"] == "error").sum())
    print(f"done: {len(meta)} series, {n_err} errors", flush=True)
    if n_err:
        errs = meta[meta.status == "error"]
        errs.to_csv(out / "preprocess_errors.csv", index=False)
        print(errs[["StudyInstanceUID", "SeriesInstanceUID"]].head(10).to_string())


if __name__ == "__main__":
    main()
