"""Per-track speed and distance, in real units - only possible in ground space.

This is the payoff of tracking on the plane rather than in pixels: "player covered
412 m at a median 2.1 m/s and peaked at 7.8 m/s" is a sentence you cannot write from
pixel trajectories without knowing where in the frame each pixel was.

Speed is computed on an EMA-smoothed velocity and then clipped at `max_speed`, because
a single bad foot-point localisation on a 25 fps clip implies an impossible sprint.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def track_stats(trace, fps, max_speed=11.0, ema=0.6):
    """trace [N,5] = (frame, x, y, vx, vy) in metres -> dict of per-track stats"""
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 2:
        return None
    xy = trace[:, 1:3]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.maximum(np.diff(trace[:, 0]) / fps, 1e-6)
    inst = d / dt
    inst = np.clip(inst, 0, max_speed)
    sm = np.empty_like(inst)
    acc = inst[0]
    for i, v in enumerate(inst):
        acc = ema * acc + (1 - ema) * v
        sm[i] = acc
    return {
        "frames": int(len(trace)),
        "duration_s": float((trace[-1, 0] - trace[0, 0]) / fps),
        "distance_m": float(d.sum()),
        "speed_median_mps": float(np.median(sm)),
        "speed_mean_mps": float(sm.mean()),
        "speed_p95_mps": float(np.percentile(sm, 95)),
        "speed_max_mps": float(sm.max()),
        "sprints": int((sm > 5.0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="speed/distance analytics from ground traces")
    ap.add_argument("--traces", required=True, help="npz written by run_tracker --traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-speed", type=float, default=11.0)
    ap.add_argument("--min-frames", type=int, default=25)
    a = ap.parse_args()

    z = np.load(a.traces, allow_pickle=False)
    fps = float(z["fps"]) if "fps" in z else 25.0
    space = str(z["space"]) if "space" in z else "unknown"
    if space != "ground":
        print(f"WARNING: traces are in '{space}' space - distances are in pixels, not metres")

    per = {}
    for k in z.files:
        if k in ("fps", "space"):
            continue
        st = track_stats(z[k], fps, a.max_speed)
        if st and st["frames"] >= a.min_frames:
            per[k] = st

    if not per:
        raise SystemExit("no tracks long enough to analyse")
    med = float(np.median([v["speed_median_mps"] for v in per.values()]))
    mx = float(max(v["speed_max_mps"] for v in per.values()))
    tot = float(sum(v["distance_m"] for v in per.values()))
    summary = {"space": space, "fps": fps, "n_tracks": len(per),
               "median_of_track_median_speed_mps": round(med, 3),
               "max_speed_mps": round(mx, 3),
               "total_distance_m": round(tot, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "tracks": per}, indent=2))
    print(json.dumps(summary, indent=2))
    # plausibility check: field-sport players sit around 1-3 m/s median
    if space == "ground" and not (0.5 <= med <= 4.0):
        print(f"  WARNING: median speed {med:.2f} m/s is outside the plausible 0.5-4.0 band")


if __name__ == "__main__":
    main()
