"""Kaggle submission kernel (offline): hidden test DICOMs -> submission.csv.

Reuses the exact training preprocessing (canonical left knee, physical crop,
window, resize) from the code dataset. Defensive by construction: any failing
series is skipped; a study with no usable series gets prior-rate predictions;
a fallback all-0.5 submission is written first so a late crash still submits.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def find_input(name: str) -> Path:
    for pat in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
        c = sorted(Path("/kaggle/input").glob(pat))
        if c:
            return c[0]
    raise FileNotFoundError(name)


CODE = find_input("dicom_to_npz.py").parent
COMP = find_input("test_series.csv").parent
WEIGHTS = find_input("best_f0.pt").parent
sys.path.insert(0, str(CODE))

import subprocess

# The Kaggle image's torch dropped sm_60 (P100). A self-contained cu118 torch
# wheel (bundled CUDA libs, sm_60..sm_90) is installed from a dataset BEFORE
# torch is imported, so inference works on whatever GPU the kernel is assigned.
def find_input_glob(pattern: str):
    for depth in ("*/", "*/*/", "*/*/*/"):
        c = sorted(Path("/kaggle/input").glob(depth + pattern))
        if c:
            return c[0]
    return None

torch_whl = find_input_glob("torch-*cu118*.whl")
if torch_whl is not None:
    import re
    import shutil
    # Kaggle ingestion strips the '+' from wheel filenames; restore it so pip
    # accepts the local version segment (2.5.1cu118 -> 2.5.1+cu118). Installs
    # torch+torchvision (cu118, sm_60..sm_90) and their bundled nvidia libs.
    whls = []
    for p in sorted(torch_whl.parent.glob("*.whl")):
        fixed_name = re.sub(r"(\d)cu118", r"\1+cu118", p.name)
        if fixed_name != p.name:
            q = Path("/tmp") / fixed_name
            shutil.copy(p, q)
            p = q
        whls.append(str(p))
    print("installing wheels:", [Path(w).name for w in whls], flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--force-reinstall", *whls], check=True)

whl = sorted(CODE.glob("wheels/timm-*.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl[0])],
               check=True)
import importlib.metadata as _md
print("torch", _md.version("torch"), "| torchvision", _md.version("torchvision"),
      "| timm", _md.version("timm"), flush=True)

import pydicom  # noqa: E402
import torch  # noqa: E402
from dicom_to_npz import (canonicalize_axes, physical_crop, resize_vol,  # noqa: E402
                          sort_and_geometry, volume_center_x, window_uint8,
                          LATERALITY_MIN_MM)
from dataset import PLANES, slice_indices  # noqa: E402
from model import SeriesModel  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
# fallback per-label prior = extractor positive rate on train (safe mid-range values)
PRIOR = [0.114, 0.063, 0.383, 0.141, 0.272, 0.156, 0.364, 0.304, 0.113, 0.133, 0.212, 0.055]
MAX_SLICES = 48
DEFAULT_BACKBONE, DEFAULT_K, DEFAULT_SIZE = "tf_efficientnetv2_s.in21k_ft_in1k", 16, 320


def preprocess_series(series_dir: Path, sizes) -> dict:
    """DICOM series -> {img_size: canonical uint8 volume} (one resize per size)."""
    dsets = [pydicom.dcmread(f) for f in sorted(series_dir.glob("*.dcm"))]
    if not dsets:
        raise ValueError("empty")
    vol, dsets, row_dir, col_dir, normal, have_geom = sort_and_geometry(dsets)
    ds_mid = dsets[len(dsets) // 2]
    lat = str(getattr(ds_mid, "ImageLaterality", "") or
              getattr(ds_mid, "Laterality", "") or "").strip().upper()
    if have_geom and lat not in ("L", "R"):
        try:
            cx = volume_center_x(ds_mid, row_dir, col_dir)
            lat = "L" if cx > LATERALITY_MIN_MM else ("R" if cx < -LATERALITY_MIN_MM else "?")
        except Exception:
            lat = "?"
    vol, _ = canonicalize_axes(vol, row_dir, col_dir, lat)
    vol, _ = physical_crop(vol, ds_mid)
    vol = window_uint8(vol)
    return {s: resize_vol(vol, s, MAX_SLICES) for s in sizes}


def series_tensor(vol: np.ndarray, k: int):
    n = vol.shape[0]
    centers = slice_indices(n, k, jitter=False)
    lo = np.clip(centers - 1, 0, n - 1)
    hi = np.clip(centers + 1, 0, n - 1)
    x = np.stack([vol[lo], vol[centers], vol[hi]], axis=1).astype(np.float32) / 255.0
    pos = (centers / max(n - 1, 1)).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(pos)


# mirror TTA: flip along patient-x (slice axis for sagittal, width otherwise),
# then swap the medial<->lateral output columns back
SWAP = list(range(12))
SWAP[2], SWAP[3], SWAP[4], SWAP[5] = 3, 2, 5, 4


def mirrored(vol: np.ndarray, plane: int) -> np.ndarray:
    return np.ascontiguousarray(vol[::-1] if plane == 0 else vol[:, :, ::-1])


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test = pd.read_csv(COMP / "test.csv")
    tser = pd.read_csv(COMP / "test_series.csv")
    sub_path = Path("/kaggle/working/submission.csv")

    # fallback submission first — a crash after this point still submits
    fb = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID})
    for lab, p in zip(LABELS, PRIOR):
        fb[lab] = p
    fb.to_csv(sub_path, index=False)

    ckpts = sorted(WEIGHTS.glob("best_f*.pt"))
    models = []  # (model, k_slices, img_size) — read from each checkpoint's args
    for c in ckpts:
        sd = torch.load(c, map_location="cpu", weights_only=False)
        a = sd.get("args") or {}
        bb = a.get("backbone", DEFAULT_BACKBONE)
        k = int(a.get("k_slices", DEFAULT_K))
        size = 384 if "384" in str(a.get("npz_root", "")) else DEFAULT_SIZE
        m = SeriesModel(bb, pretrained=False)
        m.load_state_dict(sd["model"])
        m.to(dev).eval()
        models.append((m, k, size))
        print(f"{c.name}: {bb} K={k} size={size} weak-val "
              f"{sd.get('auc', float('nan')):.4f}", flush=True)
    uniq_sk = sorted({(s, k) for _, k, s in models})
    sizes = sorted({s for _, _, s in models})
    print(f"{len(models)} model(s), sizes {sizes}, device {dev}", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def prep_study(grp):
        """Preprocess all series of a study (runs in prefetch thread pool)."""
        out = []
        for r in grp.itertuples():
            try:
                sd = COMP / "test_series" / r.StudyInstanceUID / r.SeriesInstanceUID
                vols = preprocess_series(sd, sizes)
                plane = PLANES.get(r.Anatomical_Plane, 0)
                out.append((vols, plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression)))
            except Exception:
                out.append(None)
                traceback.print_exc(limit=1)
        return out

    studies = list(tser.groupby("StudyInstanceUID"))
    pool = ThreadPoolExecutor(4)
    futs = {}
    LOOKAHEAD = 3
    rows, n_fail = [], 0
    for si, (uid, grp) in enumerate(studies):
        for j in range(si, min(si + 1 + LOOKAHEAD, len(studies))):
            if j not in futs:
                futs[j] = pool.submit(prep_study, studies[j][1])
        prepped = futs.pop(si).result()
        per_series = []
        for item in prepped:
            if item is None:
                n_fail += 1
                continue
            vols, plane, fluid, fatsat = item
            meta = torch.tensor([[plane, fluid, fatsat]] * 2)
            tens = {}
            for s, k in uniq_sk:                # orig+mirror stacked: [2, K, 3, H, W]
                x0, pos = series_tensor(vols[s], k)
                x1, _ = series_tensor(mirrored(vols[s], plane), k)
                tens[(s, k)] = (torch.stack([x0, x1]).to(dev),
                                torch.stack([pos, pos]).to(dev))
            with torch.no_grad(), torch.amp.autocast(dev, enabled=(dev == "cuda")):
                ps = [torch.sigmoid(m(tens[(s, k)][0], meta.to(dev), tens[(s, k)][1])[0])
                      .float().cpu().numpy() for m, k, s in models]
            p = np.mean(ps, axis=0)             # [2, 12]
            per_series.append(0.5 * (p[0] + p[1][SWAP]))
        if per_series:
            P = np.stack(per_series)
            score = 0.5 * (P.mean(0) + P.max(0))
        else:
            score = np.array(PRIOR)
        rows.append([uid] + score.tolist())
        if (si + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{si+1} studies, {el/(si+1):.1f}s/study, "
                  f"eta {(len(studies)-si-1)*el/(si+1)/3600:.1f}h, fails {n_fail}", flush=True)
    pool.shutdown(wait=False)

    out = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    out = fb[["StudyInstanceUID"]].merge(out, on="StudyInstanceUID", how="left")
    for lab, p in zip(LABELS, PRIOR):
        out[lab] = out[lab].fillna(p)
    out.to_csv(sub_path, index=False)
    print(f"submission written: {len(out)} rows, {n_fail} series failures, "
          f"{(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
