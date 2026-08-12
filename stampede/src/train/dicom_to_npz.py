#!/usr/bin/env python
"""Convert DICOM series to compact, anatomically-canonicalized uint8 volumes.

v3 — incorporates adversarial-review fixes and geometry upgrades:
  * atomic npz writes (tmp + os.replace) so interrupted runs never leave
    truncated files that exists()-skip would hide forever
  * pebble pool: per-task timeout, worker-death isolation (segfaulting
    decoders lose one series, not the run), incremental errors.csv
  * physical-scale normalization: fixed CROP_MM center crop via PixelSpacing
    before resize, so 1 px means the same mm everywhere (FOV spans 70-320mm)
  * orientation canonicalization: each volume axis most aligned with a patient
    axis is re-signed to point toward +x (left), +y (posterior), +z (superior);
    RIGHT knees are then mirrored along the x-aligned axis => every volume is
    a canonical LEFT knee, making medial/lateral consistent for the model
  * per-series sidecar metadata (laterality, geometry, TR/TE, descriptions)

Input modes: --series-root DIR (Kaggle mount) or --zip-path ZIP (local, avoids
exploding 819k files onto NFS). The same canonicalize/window/crop path is
reused by the inference notebook.
"""
import argparse
import io
import multiprocessing as mp
import os
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

CROP_MM = 140.0          # physical center-crop; covers >99% of knee FOVs
LATERALITY_MIN_MM = 20.0  # |patient-x| of volume center below this => unknown

_zip = None  # per-worker zip handle


def _init_zip(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def _read_dicoms(task):
    if task["members"] is not None:
        global _zip
        if _zip is None:  # pebble may not run initializer on respawned workers
            _init_zip(task["zip_path"])
        for m in task["members"]:
            with _zip.open(m) as fh:
                yield pydicom.dcmread(io.BytesIO(fh.read()))
    else:
        for f in sorted(Path(task["series_dir"]).glob("*.dcm")):
            yield pydicom.dcmread(f)


def sort_and_geometry(dsets):
    """Sort slices along a sign-canonicalized normal; return volume + geometry.

    The slice normal is flipped so its dominant patient-axis component is
    positive; slices are sorted ascending along it. Same canonicalization for
    in-plane axes happens later in canonicalize_axes().
    """
    ds0 = dsets[0]
    try:
        ori = np.array(ds0.ImageOrientationPatient, float)
        row_dir, col_dir = ori[:3], ori[3:]          # row_dir: along columns; col_dir: along rows
        normal = np.cross(row_dir, col_dir)
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        dsets.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, float), normal)))
        have_geom = True
    except Exception:
        dsets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
        row_dir = col_dir = normal = None
        have_geom = False

    arrs = []
    for ds in dsets:
        a = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arrs.append(a * slope + inter)
    shapes = {a.shape for a in arrs}
    if len(shapes) > 1:
        H = max(a.shape[0] for a in arrs); W = max(a.shape[1] for a in arrs)
        arrs = [np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1]))) for a in arrs]
    vol = np.stack(arrs)
    return vol, dsets, row_dir, col_dir, normal, have_geom


def volume_center_x(ds_mid, row_dir, col_dir):
    """Patient-space x coordinate of the middle slice's center pixel."""
    ipp = np.array(ds_mid.ImagePositionPatient, float)
    ps = [float(v) for v in ds_mid.PixelSpacing]  # [row_spacing, col_spacing]
    rows, cols = int(ds_mid.Rows), int(ds_mid.Columns)
    center = ipp + row_dir * ps[1] * (cols - 1) / 2 + col_dir * ps[0] * (rows - 1) / 2
    return float(center[0])


def canonicalize_axes(vol, row_dir, col_dir, laterality):
    """Re-sign in-plane axes toward +patient axes; mirror right knees along
    the x-aligned axis. Returns vol and the axis flip flags applied."""
    flips = {"slice": False, "row": False, "col": False, "mirrored": False}
    if row_dir is None:
        return vol, flips
    # axis 1 of vol = image rows (direction col_dir), axis 2 = image cols (direction row_dir)
    if col_dir[np.argmax(np.abs(col_dir))] < 0:
        vol = vol[:, ::-1, :]; flips["row"] = True
    if row_dir[np.argmax(np.abs(row_dir))] < 0:
        vol = vol[:, :, ::-1]; flips["col"] = True
    # after re-signing, determine which volume axis carries patient-x
    normal = np.cross(row_dir, col_dir)
    normal = normal if normal[np.argmax(np.abs(normal))] >= 0 else -normal
    axis_dirs = {0: np.abs(normal[0]), 1: np.abs(col_dir[0]), 2: np.abs(row_dir[0])}
    x_axis = max(axis_dirs, key=axis_dirs.get)
    if laterality == "R":  # mirror to canonical LEFT knee
        sl = [slice(None)] * 3
        sl[x_axis] = slice(None, None, -1)
        vol = vol[tuple(sl)]
        flips["mirrored"] = True
    return np.ascontiguousarray(vol), flips


def physical_crop(vol, ds_mid, crop_mm=CROP_MM):
    """Center-crop (or zero-pad) each slice to crop_mm x crop_mm."""
    try:
        ps = [float(v) for v in ds_mid.PixelSpacing]
    except Exception:
        return vol, False
    n, H, W = vol.shape
    ph, pw = int(round(crop_mm / ps[0])), int(round(crop_mm / ps[1]))
    def crop_pad(size, target):
        if size >= target:
            s = (size - target) // 2
            return slice(s, s + target), (0, 0)
        pad = target - size
        return slice(0, size), (pad // 2, pad - pad // 2)
    sh, padh = crop_pad(H, ph)
    sw, padw = crop_pad(W, pw)
    out = vol[:, sh, sw]
    if padh != (0, 0) or padw != (0, 0):
        out = np.pad(out, ((0, 0), padh, padw))
    return out, True


def window_uint8(vol, lo_p=0.5, hi_p=99.5):
    lo, hi = np.percentile(vol, [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255.0).astype(np.uint8)


def resize_vol(vol, size, max_slices):
    import cv2
    n = vol.shape[0]
    if n > max_slices:
        idx = np.linspace(0, n - 1, max_slices).round().astype(int)
        vol = vol[idx]
    return np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in vol])


META_TAGS = ["SeriesDescription", "RepetitionTime", "EchoTime", "InversionTime",
             "FlipAngle", "EchoTrainLength", "ScanningSequence", "SequenceVariant",
             "ScanOptions", "SliceThickness", "SpacingBetweenSlices",
             "MagneticFieldStrength", "Rows", "Columns", "ImageLaterality",
             "Laterality", "BodyPartExamined"]


def process_one(task):
    try:
        out = Path(task["out_root"]) / task["study"] / f"{task['series']}.npz"
        if out.exists():
            return ("skip", task["study"], task["series"], "", None)
        dsets = list(_read_dicoms(task))
        if not dsets:
            raise ValueError("empty series")
        vol, dsets, row_dir, col_dir, normal, have_geom = sort_and_geometry(dsets)
        n_raw = vol.shape[0]
        ds_mid = dsets[len(dsets) // 2]

        laterality = str(getattr(ds_mid, "ImageLaterality", "") or
                         getattr(ds_mid, "Laterality", "") or "").strip().upper()
        cx = None
        if have_geom:
            try:
                cx = volume_center_x(ds_mid, row_dir, col_dir)
                if laterality not in ("L", "R"):
                    if cx > LATERALITY_MIN_MM:
                        laterality = "L"
                    elif cx < -LATERALITY_MIN_MM:
                        laterality = "R"
                    else:
                        laterality = "?"
            except Exception:
                laterality = laterality or "?"
        else:
            laterality = laterality or "?"

        vol, flips = canonicalize_axes(vol, row_dir, col_dir, laterality)
        vol, phys_ok = physical_crop(vol, ds_mid)
        vol = window_uint8(vol)
        vol = resize_vol(vol, task["size"], task["max_slices"])

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, vol=vol, n_raw=n_raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        meta = {"StudyInstanceUID": task["study"], "SeriesInstanceUID": task["series"],
                "n_raw": n_raw, "laterality": laterality, "center_x_mm": cx,
                "phys_crop": phys_ok, "mirrored": flips["mirrored"],
                "have_geom": have_geom}
        for t in META_TAGS:
            v = getattr(ds_mid, t, None)
            meta[t] = str(v) if v is not None else ""
        try:
            meta["PixelSpacing"] = "x".join(str(float(v)) for v in ds_mid.PixelSpacing)
        except Exception:
            meta["PixelSpacing"] = ""
        return ("ok", task["study"], task["series"], str(vol.shape), meta)
    except Exception as e:
        return ("err", task["study"], task["series"],
                f"{type(e).__name__}: {e} | {traceback.format_exc()[-300:]}", None)


def main():
    from pebble import ProcessPool, ProcessExpired

    ap = argparse.ArgumentParser()
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--series-root", default="")
    ap.add_argument("--zip-path", default="")
    ap.add_argument("--zip-prefix", default="train_series")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--task-timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert bool(args.series_root) != bool(args.zip_path), "exactly one of --series-root/--zip-path"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for t in out_root.rglob("*.tmp"):  # leftovers from killed runs
        t.unlink()

    sdf = pd.read_csv(args.series_csv)
    members_by_series = None
    if args.zip_path:
        with zipfile.ZipFile(args.zip_path) as zf:  # also validates the zip early
            members_by_series = {}
            pfx = args.zip_prefix.strip("/") + "/"
            for n in zf.namelist():
                if n.startswith(pfx) and n.endswith(".dcm"):
                    parts = n.split("/")
                    members_by_series.setdefault((parts[1], parts[2]), []).append(n)
        print(f"zip index: {sum(len(v) for v in members_by_series.values())} dcm "
              f"in {len(members_by_series)} series", flush=True)

    tasks = []
    for r in sdf.itertuples():
        key = (r.StudyInstanceUID, r.SeriesInstanceUID)
        tasks.append(dict(
            study=r.StudyInstanceUID, series=r.SeriesInstanceUID,
            members=members_by_series.get(key) if members_by_series else None,
            zip_path=args.zip_path or None,
            series_dir=str(Path(args.series_root) / r.StudyInstanceUID / r.SeriesInstanceUID)
            if args.series_root else None,
            out_root=str(out_root), size=args.size, max_slices=args.max_slices))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} series, {args.workers} workers", flush=True)

    err_f = open(out_root / "errors.csv", "a", buffering=1)
    if err_f.tell() == 0:
        err_f.write("StudyInstanceUID,SeriesInstanceUID,error\n")
    metas, n_ok, n_skip, n_err = [], 0, 0, 0
    init, initargs = (_init_zip, (args.zip_path,)) if args.zip_path else (None, ())

    with ProcessPool(max_workers=args.workers, initializer=init, initargs=initargs) as pool:
        futures = [(t, pool.schedule(process_one, args=(t,), timeout=args.task_timeout))
                   for t in tasks]
        for i, (t, fut) in enumerate(futures):
            try:
                st, study, series, info, meta = fut.result()
                if st == "ok":
                    n_ok += 1; metas.append(meta)
                elif st == "skip":
                    n_skip += 1
                else:
                    n_err += 1
                    err_f.write(f'{study},{series},"{info[:300]}"\n'.replace("\n", " ") + "\n")
            except (ProcessExpired, TimeoutError) as e:
                n_err += 1
                err_f.write(f'{t["study"]},{t["series"]},"worker_died_or_timeout: {e}"\n')
            except Exception as e:
                n_err += 1
                err_f.write(f'{t["study"]},{t["series"]},"future_error: {str(e)[:200]}"\n')
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(tasks)} ok={n_ok} skip={n_skip} err={n_err}", flush=True)
                if metas:
                    _append_meta(out_root, metas); metas = []
    if metas:
        _append_meta(out_root, metas)
    err_f.close()
    print(f"done ok={n_ok} skip={n_skip} err={n_err}", flush=True)


def _append_meta(out_root: Path, metas: list):
    p = out_root / "series_meta.parquet"
    df = pd.DataFrame(metas)
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df = df.drop_duplicates(["StudyInstanceUID", "SeriesInstanceUID"], keep="last")
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


if __name__ == "__main__":
    main()
