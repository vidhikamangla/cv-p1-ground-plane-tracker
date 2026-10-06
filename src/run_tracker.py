"""Run the tracker over one cached sequence and write MOTChallenge results.

Frames are only decoded when something actually needs pixels (colour gate or CMC),
so the pure image-space IoU baseline runs at thousands of fps and the ablation is
cheap enough to re-run in full after every change.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .frames import list_frames
from .geometry.cmc import CMC, warp_magnitude
from .geometry.homography import dlt, reprojection_error
from .tracking.tracker import GroundTracker


def load_dets(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[int(d["frame"])] = (np.asarray(d["boxes"], dtype=np.float64).reshape(-1, 4),
                                    np.asarray(d["scores"], dtype=np.float64).reshape(-1))
    return out


def load_homography(path):
    """{ref_frame, img_pts, world_pts} -> (H_ref image->world, ref_frame, error)"""
    spec = json.loads(Path(path).read_text())
    img = np.asarray(spec["img_pts"], dtype=np.float64).reshape(-1, 2)
    world = np.asarray(spec["world_pts"], dtype=np.float64).reshape(-1, 2)
    H = dlt(img, world)
    mean_e, max_e = reprojection_error(H, img, world)
    return H, int(spec.get("ref_frame", 1)), (mean_e, max_e)


def main():
    ap = argparse.ArgumentParser(description="track one sequence")
    C.add_args(ap, default="configs/default.yaml")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--homography", default=None)
    ap.add_argument("--traces", default=None, help="optional npz of world-space traces")
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)

    frames = list_frames(a.frames_dir)
    dets = load_dets(a.detections)
    fps = a.fps or 25.0

    space = cfg.tracker.space
    use_cmc = bool(cfg.tracker.use_cmc)
    need_pixels = bool(cfg.tracker.use_color) or use_cmc

    H_ref = None
    ref_frame = 1
    if space == "ground":
        hpath = a.homography or cfg.paths.get("homography")
        if not hpath or not Path(hpath).exists():
            raise SystemExit(f"ground space needs --homography (got {hpath!r})")
        H_ref, ref_frame, err = load_homography(hpath)
        print(f"H_ref from {hpath}: reprojection mean {err[0]:.3f} m, max {err[1]:.3f} m")

    cmc = None
    if use_cmc:
        ref_img = frames[min(max(ref_frame - 1, 0), len(frames) - 1)]
        ref_gray = cv2.cvtColor(cv2.imread(str(ref_img)), cv2.COLOR_BGR2GRAY)
        cmc = CMC(ref_gray)

    # adaptive CMC: only trust the warp when the camera demonstrably moved
    cmc_min_shift = float(cfg.tracker.get("cmc_min_shift_px", 0) or 0)
    cmc_mags, n_cmc_skipped = [], 0

    tracker = GroundTracker(cfg, fps)
    lines, traces = [], {}
    t0 = time.perf_counter()

    for i, fp in enumerate(frames, start=1):
        boxes, scores = dets.get(i, (np.zeros((0, 4)), np.zeros((0,))))
        frame = cv2.imread(str(fp)) if need_pixels else None

        # Measurement transform for this frame (see GroundTracker._measure):
        #   W = CMC warp to the reference frame, H_ref = reference-frame homography
        H_t = H_inv = None
        W = None
        if cmc is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            W, _info = cmc.estimate(gray, boxes)
            if cmc_min_shift > 0:
                mag = warp_magnitude(W, frame.shape)
                cmc_mags.append(mag)
                if mag < cmc_min_shift:
                    W = None          # camera has not actually moved; do not add noise
                    n_cmc_skipped += 1
        if space == "ground":
            H_t = H_ref @ W if W is not None else H_ref   # W=None -> the T5 failure row
        elif W is not None:
            H_t = W
        if H_t is not None:
            try:
                H_inv = np.linalg.inv(H_t)
            except np.linalg.LinAlgError:
                H_inv = None

        active = tracker.update(i, boxes, scores, frame_bgr=frame, H_t=H_t, H_inv=H_inv)
        for tr in active:
            x0, y0, x1, y1 = tr.box
            lines.append(f"{i},{tr.id},{x0:.2f},{y0:.2f},{x1-x0:.2f},{y1-y0:.2f},"
                         f"{tr.score:.4f},-1,-1,-1")
            traces.setdefault(tr.id, []).append((i, float(tr.x[0]), float(tr.x[1]),
                                                 float(tr.x[2]), float(tr.x[3])))

    elapsed = time.perf_counter() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))

    if a.traces:
        Path(a.traces).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.traces, **{str(k): np.asarray(v, dtype=np.float64) for k, v in traces.items()},
                 fps=np.float64(fps), space=np.str_(space))

    msg = (f"{len(frames)} frames, {len(lines)} rows, {len(traces)} ids, "
           f"{elapsed:.1f}s ({len(frames)/max(elapsed,1e-9):.1f} fps)")
    if cmc is not None:
        msg += f"  cmc={cmc.stats}"
        if cmc_mags:
            import statistics
            msg += (f" median_shift={statistics.median(cmc_mags):.2f}px"
                    f" skipped={n_cmc_skipped}/{len(cmc_mags)}")
    print(msg)


if __name__ == "__main__":
    main()
