"""Turn typed pixel coordinates into a homography JSON, with a visual check.

A GUI clicker is useless here (remote box, no display), so the workflow is:
  1. `python -m src.geometry.calibrate grid --frames-dir ... --ref-frame 1`
     writes a PNG of the reference frame with a labelled 100 px grid.
  2. Read off the pixel coordinates of 4+ named court landmarks from that PNG.
  3. `python -m src.geometry.calibrate fit --pts "x,y;x,y;x,y;x,y"
         --landmarks corner_tl,corner_tr,corner_br,corner_bl ...`
     fits the homography, prints the reprojection error in metres, and writes an
     overlay PNG with the full court re-projected onto the frame so a bad click is
     obvious at a glance rather than three hours later in the HOTA table.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..frames import list_frames
from .homography import apply_h, dlt, reprojection_error

COURTS = Path(__file__).resolve().parents[2] / "configs" / "courts"


def _frame(frames_dir, ref_frame):
    frames = list_frames(frames_dir)
    i = min(max(ref_frame - 1, 0), len(frames) - 1)
    img = cv2.imread(str(frames[i]))
    if img is None:
        raise SystemExit(f"could not read {frames[i]}")
    return img, frames[i]


def cmd_grid(a):
    img, path = _frame(a.frames_dir, a.ref_frame)
    out = img.copy()
    h, w = out.shape[:2]
    step = a.step
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(out, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(out, str(y), (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    print(f"{path.name}  {w}x{h}  grid every {step}px -> {dst}")


def cmd_fit(a):
    court = json.loads((COURTS / f"{a.court}.json").read_text())[a.variant]
    lm = court["landmarks"]
    names = [s.strip() for s in a.landmarks.split(",")]
    pts = [[float(v) for v in p.split(",")] for p in a.pts.strip().strip(";").split(";")]
    if len(names) != len(pts):
        raise SystemExit(f"{len(names)} landmarks vs {len(pts)} points")
    unknown = [n for n in names if n not in lm]
    if unknown:
        raise SystemExit(f"unknown landmarks {unknown}; available: {sorted(lm)}")

    img_pts = np.asarray(pts, dtype=np.float64)
    world_pts = np.asarray([lm[n] for n in names], dtype=np.float64)
    H = dlt(img_pts, world_pts)
    mean_e, max_e = reprojection_error(H, img_pts, world_pts)
    print(f"reprojection error: mean {mean_e:.3f} m, max {max_e:.3f} m "
          f"({len(names)} correspondences)")
    if max_e > a.max_error:
        print(f"  WARNING: max error above {a.max_error} m - check the typed coordinates")

    spec = {"ref_frame": a.ref_frame, "court": a.court, "variant": a.variant,
            "landmarks": names, "img_pts": img_pts.tolist(),
            "world_pts": world_pts.tolist(),
            "reprojection_mean_m": mean_e, "reprojection_max_m": max_e}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    print(f"wrote {out}")

    if a.frames_dir and a.overlay:
        img, _ = _frame(a.frames_dir, a.ref_frame)
        Hi = np.linalg.inv(H)
        L, W = court["length"], court["width"]
        # re-project the court outline and the halfway line back into the image
        polys = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
        for poly in polys:
            ipts = apply_h(Hi, np.asarray(poly, dtype=np.float64))
            ipts = ipts[np.all(np.isfinite(ipts), axis=1)].astype(np.int32)
            if len(ipts) >= 2:
                cv2.polylines(img, [ipts], False, (0, 0, 255), 2, cv2.LINE_AA)
        # 1 m grid, to show whether the plane estimate is sane across the whole court
        for xm in np.arange(0, L + 1e-6, 2.0):
            seg = apply_h(Hi, np.array([[xm, 0.0], [xm, W]]))
            if np.all(np.isfinite(seg)):
                cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)),
                         (0, 180, 0), 1, cv2.LINE_AA)
        for p, n in zip(img_pts, names):
            cv2.circle(img, tuple(p.astype(int)), 6, (255, 255, 0), -1)
            cv2.putText(img, n, tuple((p + np.array([8, -8])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        ov = Path(a.overlay)
        ov.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ov), img)
        print(f"overlay -> {ov}  (the red outline must sit on the painted court lines)")


def main():
    ap = argparse.ArgumentParser(description="ground-plane calibration helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="write a coordinate-grid PNG of the reference frame")
    g.add_argument("--frames-dir", required=True)
    g.add_argument("--ref-frame", type=int, default=1)
    g.add_argument("--step", type=int, default=100)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_grid)

    f = sub.add_parser("fit", help="fit H from typed pixel coordinates")
    f.add_argument("--pts", required=True, help='"x,y;x,y;x,y;x,y" in pixels')
    f.add_argument("--landmarks", required=True, help="comma-separated landmark names")
    f.add_argument("--court", default="basketball")
    f.add_argument("--variant", default="fiba", help="court variant key inside the court json")
    f.add_argument("--ref-frame", type=int, default=1)
    f.add_argument("--out", required=True)
    f.add_argument("--frames-dir", default=None)
    f.add_argument("--overlay", default=None)
    f.add_argument("--max-error", type=float, default=0.5)
    f.set_defaults(func=cmd_fit)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
