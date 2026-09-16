#!/usr/bin/env python3
"""Offline inference: DICOM test_series/ -> submission.csv (Kaggle notebook compatible).

Self-contained: decodes DICOMs in memory (no intermediate files), runs every
checkpoint found under --ckpt-dirs, averages sigmoid outputs.

Local test (TACC):
  python predict.py --test-csv data/csv/test.csv --series-csv data/csv/test_series.csv \
     --dicom-root data/raw/test_series --ckpt-dirs results/img_v1 results/img_v2 --out submission.csv
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def load_series_volume(series_dir: Path, size: int = 256, fov_mm: float = 0) -> np.ndarray:
    """DICOM series -> (n, size, size) uint8, sorted, optional physical crop, windowed."""
    import cv2
    import pydicom
    slices = []
    spacing = None
    for f in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(str(f))
        pos = getattr(ds, "ImagePositionPatient", None)
        key = None
        if pos is not None and hasattr(ds, "ImageOrientationPatient"):
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            key = float(np.dot(np.cross(iop[:3], iop[3:]), np.array(pos, dtype=float)))
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        if spacing is None:
            spacing = getattr(ds, "PixelSpacing", None)
        slices.append((key, inst, arr * slope + inter))
    if not slices:
        raise ValueError(f"empty series {series_dir}")
    if all(s[0] is not None for s in slices):
        slices.sort(key=lambda s: s[0])
    else:
        slices.sort(key=lambda s: s[1])
    shapes = {s[2].shape for s in slices}
    if len(shapes) > 1:
        from collections import Counter
        tgt = Counter(s[2].shape for s in slices).most_common(1)[0][0]
        slices = [(a, b, cv2.resize(im, (tgt[1], tgt[0]))) for a, b, im in slices]
    vol = np.stack([s[2] for s in slices])
    if fov_mm and spacing is not None:
        try:
            py, px = float(spacing[0]), float(spacing[1])
            ch = min(vol.shape[1], max(32, int(round(fov_mm / py))))
            cw = min(vol.shape[2], max(32, int(round(fov_mm / px))))
            y0 = (vol.shape[1] - ch) // 2; x0 = (vol.shape[2] - cw) // 2
            vol = vol[:, y0:y0 + ch, x0:x0 + cw]
        except (TypeError, ValueError, IndexError):
            pass
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / max(hi - lo, 1e-6) * 255.0
    out = np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in vol])
    return out.astype(np.uint8)


def sample_slices(vol: np.ndarray, k: int, crop: int) -> torch.Tensor:
    n = vol.shape[0]
    pos = np.clip(np.round(np.linspace(0, n - 1, k)).astype(int), 0, n - 1)
    idx3 = np.stack([np.clip(pos - 1, 0, n - 1), pos, np.clip(pos + 1, 0, n - 1)])
    sl = np.ascontiguousarray(vol[idx3.reshape(-1)]).reshape(3, k, *vol.shape[1:])
    sl = np.transpose(sl, (1, 0, 2, 3)).astype(np.float32) / 255.0
    H, W = sl.shape[2], sl.shape[3]
    y0, x0 = (H - crop) // 2, (W - crop) // 2
    sl = sl[:, :, y0:y0 + crop, x0:x0 + crop]
    return torch.from_numpy((sl - 0.45) / 0.225)


DECODE_SIZE = 256
DECODE_FOV = 0.0


def _decode_safe(series_dir):
    try:
        return load_series_volume(series_dir, size=DECODE_SIZE, fov_mm=DECODE_FOV)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--ckpt-crops", nargs="+", type=int, default=None,
                    help="per-dir crop sizes (parallel to --ckpt-dirs); default = --crop for all")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--k-slices", type=int, default=16)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument("--model-src", default=None,
                    help="dir containing model.py (defaults to this file's dir)")
    ap.add_argument("--decode-workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=256, help="decode resolution (must match training volumes)")
    ap.add_argument("--fov-mm", type=float, default=0)
    ap.add_argument("--tta", action="store_true", help="average over vertical flip")
    args = ap.parse_args()

    global DECODE_SIZE, DECODE_FOV
    DECODE_SIZE, DECODE_FOV = args.size, args.fov_mm
    # fork decode workers BEFORE any CUDA init (fork inherits the globals above)
    pool = Pool(args.decode_workers) if args.decode_workers > 1 else None

    src = args.model_src or str(Path(__file__).resolve().parent.parent / "training")
    sys.path.insert(0, src)
    from model import KneeStudyModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    amp_dtype = torch.bfloat16 if cap >= 8 else torch.float16
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if device == "cuda" else ["cpu"]
    print(f"device {device} x{len(devs)} capability {cap} -> autocast {amp_dtype}", flush=True)
    crops = args.ckpt_crops or [args.crop] * len(args.ckpt_dirs)
    assert len(crops) == len(args.ckpt_dirs)
    ckpts = []
    for d, cr in zip(args.ckpt_dirs, crops):
        for p in sorted(Path(d).glob("fold*/best.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = KneeStudyModel(ck["backbone"], pretrained=False)
            m.load_state_dict(ck["model"])
            dv = devs[len(ckpts) % len(devs)]
            m.eval().to(dv)
            ckpts.append((m, cr, dv))
            print(f"loaded {p} ({ck['backbone']}, crop {cr}, dev {dv}, val_auc {ck.get('val_auc', 0):.3f})", flush=True)
    assert ckpts, "no checkpoints found"
    crop_sizes = sorted({cr for _, cr, _ in ckpts})
    ckpts_by_dev = {dv: [c for c in ckpts if c[2] == dv] for dv in devs if any(c[2] == dv for c in ckpts)}
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(len(ckpts_by_dev)) if len(ckpts_by_dev) > 1 else None

    test = pd.read_csv(args.test_csv)
    series = pd.read_csv(args.series_csv)
    root = Path(args.dicom_root)
    rows = []
    for si, uid in enumerate(test.StudyInstanceUID):
        srs = list(series[series.StudyInstanceUID == uid].head(args.max_series).itertuples())
        dirs = [root / uid / r.SeriesInstanceUID for r in srs]
        if pool is not None:
            vols = pool.map(_decode_safe, dirs)
        else:
            vols = [_decode_safe(d) for d in dirs]
        xs, metas = [], []
        for r, vol in zip(srs, vols):
            if vol is None:
                print(f"  skip series {r.SeriesInstanceUID}", flush=True)
                continue
            xs.append({cr: sample_slices(vol, args.k_slices, cr) for cr in crop_sizes})
            m = np.zeros(5, dtype=np.float32)
            m[PLANES.get(r.Anatomical_Plane, 0)] = 1.0
            m[3] = getattr(r, "Fluid_Sensitive", 0); m[4] = getattr(r, "Fat_Suppression", 0)
            metas.append(m)
        if not xs:  # no decodable series: predict global prior
            rows.append([uid] + [0.5] * len(LABELS))
            continue
        xcat_cpu = {cr: torch.stack([d[cr] for d in xs])[None] for cr in crop_sizes}
        meta_cpu = torch.from_numpy(np.stack(metas))[None]

        def run_dev(dv):
            xcat = {cr: x.to(dv) for cr, x in xcat_cpu.items()}
            meta = meta_cpu.to(dv)
            mask = torch.ones(1, len(xs), device=dv)
            out = []
            with torch.no_grad(), torch.autocast("cuda" if dv != "cpu" else "cpu",
                                                 dtype=amp_dtype, enabled=dv != "cpu"):
                for m_, cr, _ in ckpts_by_dev[dv]:
                    views = [xcat[cr]] + ([torch.flip(xcat[cr], dims=[-2])] if args.tta else [])
                    out += [m_(v, meta, mask).float().sigmoid().cpu().numpy()[0] for v in views]
            return out

        if executor is not None:
            probs = [p for chunk in executor.map(run_dev, list(ckpts_by_dev)) for p in chunk]
        else:
            probs = run_dev(next(iter(ckpts_by_dev)))
        rows.append([uid] + list(np.mean(probs, axis=0)))
        if (si + 1) % 50 == 0:
            print(f"{si+1}/{len(test)} studies", flush=True)

    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {sub.shape}", flush=True)


if __name__ == "__main__":
    main()
