"""Side-by-side render: annotated frames | bird's-eye view, plus an optional GIF.

This is the artifact that makes the project legible in five seconds, so it shows the
thing the project is actually about - the same players in pixels and in metres.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..analytics.bev import CourtCanvas, colour_for
from ..frames import list_frames


def load_tracks(path):
    """MOTChallenge txt -> {frame: [(id, x0,y0,x1,y1, conf)]}"""
    per = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        f, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        conf = float(p[6]) if len(p) > 6 else 1.0
        per[f].append((tid, x, y, x + w, y + h, conf))
    return per


def main():
    ap = argparse.ArgumentParser(description="render tracking video + BEV")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--traces", default=None, help="npz of ground traces for the BEV panel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--court-length", type=float, default=28.0)
    ap.add_argument("--court-width", type=float, default=15.0)
    ap.add_argument("--panel-height", type=int, default=720)
    ap.add_argument("--gif-every", type=int, default=4, help="sample every Nth frame")
    ap.add_argument("--gif-frames", type=int, default=70)
    ap.add_argument("--gif-scale", type=float, default=0.33)
    a = ap.parse_args()

    frames = list_frames(a.frames_dir)[:a.max_frames]
    tracks = load_tracks(a.tracks)

    ground = {}
    if a.traces and Path(a.traces).exists():
        z = np.load(a.traces, allow_pickle=False)
        for k in z.files:
            if k in ("fps", "space"):
                continue
            arr = z[k]
            for row in arr:
                ground.setdefault(int(row[0]), {})[int(k)] = (row[1], row[2], row[3], row[4])

    canvas = CourtCanvas(a.court_length, a.court_width) if ground else None
    court_x_max = a.court_length + 5.0
    court_y_max = a.court_width + 5.0
    trails = defaultdict(list)
    writer = None
    gif_frames = []

    for i, fp in enumerate(frames, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        for tid, x0, y0, x1, y1, conf in tracks.get(i, []):
            c = colour_for(tid)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
            cv2.putText(img, str(tid), (int(x0), max(12, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {i}  tracks {len(tracks.get(i, []))}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        scale = a.panel_height / img.shape[0]
        left = cv2.resize(img, (int(img.shape[1] * scale), a.panel_height))

        if canvas is not None:
            pos, spd = {}, {}
            for tid, (x, y, vx, vy) in ground.get(i, {}).items():
                # drop positions outside a sane court neighbourhood: a single bad foot
                # point projects metres away and would otherwise draw a line across the
                # whole court, hiding the tracks that are actually fine
                if not (-5.0 <= x <= court_x_max and -5.0 <= y <= court_y_max):
                    continue
                pos[tid] = (x, y)
                trails[tid].append((x, y))
                spd[tid] = float(np.hypot(vx, vy))
            # only trail tracks that are visible right now, so dead IDs stop drawing
            live = {t: trails[t] for t in pos}
            bev = canvas.render(pos, live, trail_len=20, speeds=spd)
            bs = a.panel_height / bev.shape[0]
            right = cv2.resize(bev, (int(bev.shape[1] * bs), a.panel_height))
            combo = np.hstack([left, right])
        else:
            combo = left

        if writer is None:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     a.fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        if a.gif and i % a.gif_every == 1 and len(gif_frames) < a.gif_frames:
            small = cv2.resize(combo, (int(combo.shape[1] * a.gif_scale),
                                       int(combo.shape[0] * a.gif_scale)))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    if writer:
        writer.release()
    print(f"wrote {a.out} ({len(frames)} frames)")

    if a.gif and gif_frames:
        try:
            import imageio.v2 as imageio
            Path(a.gif).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(a.gif, gif_frames, duration=0.12, loop=0)
            print(f"wrote {a.gif} ({len(gif_frames)} frames)")
        except ImportError:
            print("imageio not installed - skipped GIF")


if __name__ == "__main__":
    main()
