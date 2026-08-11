#!/usr/bin/env python3
"""Time-boxed recipe sweep (autoresearch-style) for the knee study model.

Each trial: fold 0, 3 epochs (~20 min proxy), scored by val-vs-teacher AUC
(n=1088; the gold-58 is recorded but NEVER used for selection).
Runs one worker per GPU; results append to results/sweep/trials.csv.

Usage (on a GPU node):  python sweep.py --gpus 0 1 2 --hours 2.5
"""
import argparse
import csv
import itertools
import random
import re
import subprocess
import threading
import time
from pathlib import Path

random.seed()  # want different draws across restarts

W = Path("/work/07880/devansh/lonestar/rsna_knee")
CS = ["--folds-csv", "/scratch/07880/devansh/rsna_knee/data/csv/train_folds_r2.csv",
      "--series-csv", "/scratch/07880/devansh/rsna_knee/data/csv/train_series.csv"]

SPACE = {
    "lr": [1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4],
    "k_slices": [12, 16, 20, 24],
    "batch_accum": [(2, 4), (3, 3), (4, 2)],
}
BASELINE = {"lr": 2e-4, "k_slices": 16, "batch_accum": (3, 3)}


def draw(trial_idx):
    if trial_idx < 2:                      # noise-floor replicates of the baseline
        return dict(BASELINE)
    return {k: random.choice(v) for k, v in SPACE.items()}


def run_trial(gpu, cfg, tid, out_csv, lock):
    b, a = cfg["batch_accum"]
    out = W / f"results/sweep/trial{tid:03d}"
    log = W / f"logs/sweep_trial{tid:03d}.log"
    cmd = ["python", "train.py", "--fold", "0", "--backbone", "tf_efficientnetv2_s",
           "--epochs", "3", "--lr", str(cfg["lr"]), "--k-slices", str(cfg["k_slices"]),
           "--batch", str(b), "--accum", str(a), "--out-dir", str(out), *CS]
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=W / "scripts/training", stdout=f, stderr=subprocess.STDOUT,
                       env={"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HOME": "/work/07880/devansh/hf_cache",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                            "PATH": "/work/07880/devansh/anaconda3/envs/rsna_env/bin:/usr/bin:/bin"})
    txt = log.read_text(errors="ignore")
    m = re.findall(r"DONE fold 0: best val AUC ([\d.]+) \(gold ([\d.]+)\)", txt)
    val_auc, gold_auc = (float(m[-1][0]), float(m[-1][1])) if m else (float("nan"), float("nan"))
    with lock:
        new = not Path(out_csv).exists()
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["trial", "gpu", "lr", "k_slices", "batch", "accum", "val_auc", "gold_auc", "minutes"])
            w.writerow([tid, gpu, cfg["lr"], cfg["k_slices"], b, a, val_auc, gold_auc,
                        round((time.time() - t0) / 60, 1)])
    print(f"trial {tid} (gpu{gpu}) {cfg} -> val {val_auc:.4f} gold {gold_auc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hours", type=float, default=2.5)
    args = ap.parse_args()
    (W / "results/sweep").mkdir(parents=True, exist_ok=True)
    out_csv = W / "results/sweep/trials.csv"
    deadline = time.time() + args.hours * 3600
    lock = threading.Lock()
    counter = itertools.count(sum(1 for _ in open(out_csv)) - 1 if out_csv.exists() else 0)

    def worker(gpu):
        while time.time() < deadline - 22 * 60:   # don't start a trial we can't finish
            tid = next(counter)
            run_trial(gpu, draw(tid), tid, out_csv, lock)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads: t.start()
    for t in threads: t.join()
    print("SWEEP WINDOW CLOSED", flush=True)


if __name__ == "__main__":
    main()
