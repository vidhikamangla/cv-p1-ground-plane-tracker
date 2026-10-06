"""Cache detections once, so every tracker ablation reads the same boxes.

This matters for fairness as much as for speed: if each tracker row re-ran the
detector, run-to-run nondeterminism would show up in HOTA and be mistaken for a
tracking effect. Detection runs at a deliberately low confidence (0.10) because the
tracker's two-stage association is what does the filtering.
"""
import argparse
import json
from pathlib import Path

from . import config as C
from .frames import list_frames


def main():
    ap = argparse.ArgumentParser(description="cache detections for a sequence")
    C.add_args(ap, default="configs/default.yaml")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="*", type=int, default=None)
    a = ap.parse_args()
    cfg = C.load(a.config, a.sets)
    d = cfg.detector

    from ultralytics import YOLO
    model = YOLO(d.weights)

    frames = list_frames(a.frames_dir)
    classes = a.classes if a.classes is not None else list(d.classes)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = int(d.get("batch", 16))
    n_det = 0
    with out.open("w") as f:
        for start in range(0, len(frames), batch):
            chunk = frames[start:start + batch]
            res = model.predict([str(p) for p in chunk], imgsz=int(d.imgsz),
                                conf=float(d.conf), iou=float(d.iou), classes=classes,
                                device=d.get("device", 0), half=bool(d.get("half", True)),
                                verbose=False)
            for k, r in enumerate(res):
                b = r.boxes
                boxes = b.xyxy.cpu().numpy().tolist() if b is not None else []
                scores = b.conf.cpu().numpy().tolist() if b is not None else []
                labels = b.cls.cpu().numpy().astype(int).tolist() if b is not None else []
                n_det += len(boxes)
                f.write(json.dumps({"frame": start + k + 1,
                                    "file": chunk[k].name,
                                    "boxes": boxes, "scores": scores,
                                    "labels": labels}) + "\n")
    print(f"{len(frames)} frames, {n_det} detections -> {out}")


if __name__ == "__main__":
    main()
