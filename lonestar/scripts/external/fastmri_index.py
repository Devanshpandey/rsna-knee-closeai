#!/usr/bin/env python3
"""Walk extracted fastMRI knee DICOMs -> series CSV compatible with preprocess_volumes.py.

Infers plane + fluid-sensitivity/fat-sat from DICOM headers (SeriesDescription /
ScanningSequence / ImageOrientationPatient) so downstream metadata matches the
competition schema.

Usage: python fastmri_index.py --root /scratch/.../external/fastmri --out fastmri_series.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def plane_from_iop(iop):
    try:
        iop = np.array(iop, dtype=float)
        normal = np.abs(np.cross(iop[:3], iop[3:]))
        return ["Sagittal", "Coronal", "Axial"][int(np.argmax(normal))] if False else \
               {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(np.argmax(normal))]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, seen = [], set()
    root = Path(args.root)
    for dcm in root.rglob("*.dcm"):
        series_dir = dcm.parent
        if series_dir in seen:
            continue
        seen.add(series_dir)
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        except Exception:
            continue
        desc = (getattr(ds, "SeriesDescription", "") or "").upper()
        plane = plane_from_iop(getattr(ds, "ImageOrientationPatient", None))
        if plane is None:
            for kw, p in [("SAG", "Sagittal"), ("COR", "Coronal"), ("AX", "Axial"), ("TRA", "Axial")]:
                if kw in desc:
                    plane = p; break
        if plane is None:
            continue
        fat = int(bool(re.search(r"FS|FAT ?SAT|SPAIR|STIR", desc)))
        fluid = int(bool(re.search(r"T2|PD|STIR|PROTON", desc)))
        n_files = len(list(series_dir.glob("*.dcm")))
        if n_files < 8:
            continue
        rows.append(dict(
            StudyInstanceUID=str(getattr(ds, "StudyInstanceUID", series_dir.parent.name)),
            SeriesInstanceUID=str(getattr(ds, "SeriesInstanceUID", series_dir.name)),
            Anatomical_Plane=plane, Fluid_Sensitive=fluid, Fat_Suppression=fat,
            series_path=str(series_dir), n_files=n_files, desc=desc))
        if len(rows) % 2000 == 0:
            print(len(rows), "series indexed", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"indexed {len(df)} series / {df.StudyInstanceUID.nunique()} studies -> {args.out}")
    print(df.groupby(["Anatomical_Plane", "Fluid_Sensitive"]).size().to_string())


if __name__ == "__main__":
    main()
