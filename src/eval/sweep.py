"""Sensitivity of HOTA to the Kalman noise priors.

The point is not to find the best cell. It is to show how *flat or sharp* the optimum is:
a tracker whose HOTA swings 10 points across a plausible range of process noise is one
whose headline number is a tuning artifact. This is the plot that tells you whether to
believe the ablation table.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_cell(py, cfg, seq_list, frames_root, data_root, te_root, accel, meas, tracker_name,
             bench="SportsMOT", split="val"):
    out_dir = Path(data_root) / "trackers" / "mot_challenge" / f"{bench}-{split}" / tracker_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq in seq_list:
        cmd = [py, "-m", "src.run_tracker", "--config", cfg,
               "--frames-dir", str(Path(frames_root) / seq / "img1"),
               "--detections", f"outputs/dets/{seq}.jsonl",
               "--out", str(out_dir / f"{seq}.txt"), "--fps", "25",
               "--homography", f"configs/homography/{seq}.json",
               "--set", f"kalman.accel_std={accel}", f"kalman.meas_std={meas}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None
    res = Path("outputs/results/sweep_tmp.csv")
    cmd = [py, "-m", "src.eval.run_trackeval", "--trackeval-root", str(te_root),
           "--data-root", str(data_root), "--benchmark", bench, "--split", split,
           "--tracker", tracker_name, "--sequences", *seq_list,
           "--out", str(res), "--workers", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    summ = res.parent / (res.stem + "_summary.json")
    if r.returncode != 0 or not summ.exists():
        return None
    return json.loads(summ.read_text())


def main():
    ap = argparse.ArgumentParser(description="HOTA sensitivity over the Kalman priors")
    ap.add_argument("--config", default="configs/ablations/t6.yaml")
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--trackeval-root", required=True)
    ap.add_argument("--accel", nargs="+", type=float, default=[3, 10, 20, 40, 80])
    ap.add_argument("--meas", nargs="+", type=float, default=[0.2, 0.35, 0.6, 1.0])
    ap.add_argument("--out-json", default="outputs/results/sweep.json")
    ap.add_argument("--plot", default="outputs/figures/sensitivity.png")
    ap.add_argument("--metric", default="HOTA")
    a = ap.parse_args()

    grid = np.full((len(a.accel), len(a.meas)), np.nan)
    records = []
    for i, acc in enumerate(a.accel):
        for j, ms in enumerate(a.meas):
            d = run_cell(sys.executable, a.config, a.sequences, a.frames_root,
                         a.data_root, a.trackeval_root, acc, ms, "sweep_cell")
            if d:
                grid[i, j] = d.get(a.metric, np.nan)
                records.append({"accel_std": acc, "meas_std": ms, **d})
                print(f"  accel={acc:<6g} meas={ms:<5g} -> {a.metric} {grid[i,j]:.2f}", flush=True)
            else:
                print(f"  accel={acc:<6g} meas={ms:<5g} -> FAILED", flush=True)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(
        {"accel": a.accel, "meas": a.meas, "metric": a.metric,
         "grid": grid.tolist(), "records": records}, indent=2))

    finite = grid[np.isfinite(grid)]
    if finite.size:
        best = np.unravel_index(np.nanargmax(grid), grid.shape)
        print(f"\nbest {a.metric} {grid[best]:.2f} at accel_std={a.accel[best[0]]}, "
              f"meas_std={a.meas[best[1]]}")
        print(f"spread across the grid: {finite.min():.2f} - {finite.max():.2f} "
              f"({finite.max()-finite.min():.2f} points)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(a.meas)), [str(m) for m in a.meas])
        ax.set_yticks(range(len(a.accel)), [str(x) for x in a.accel])
        ax.set_xlabel("meas_std (m)")
        ax.set_ylabel("accel_std (m/s$^2$)")
        ax.set_title(f"{a.metric} sensitivity to the Kalman priors")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                            color="white", fontsize=9)
        fig.colorbar(im, ax=ax, label=a.metric)
        fig.tight_layout()
        Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.plot)
        print(f"plot -> {a.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
