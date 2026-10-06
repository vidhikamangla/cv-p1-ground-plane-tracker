"""Best-crop selection: pick the few frames of a track worth embedding.

A 300-frame track does not need 300 embeddings. Most of its crops are motion-blurred,
half-occluded, clipped by the frame edge, or 12 pixels tall, and averaging those in
actively destroys the identity signal. Scoring crops and keeping the top-K is both
cheaper and *more* accurate than embedding everything.

The score multiplies four independent quality signals, so any one of them being bad is
enough to reject the crop:

    detector confidence  - the detector's own opinion
    sqrt(area)           - resolution, square-rooted so a huge box does not dominate
    sharpness            - variance of the Laplacian, the standard blur proxy
    edge penalty         - boxes touching the frame border are usually truncated people
"""
import cv2
import numpy as np


def sharpness(gray):
    """Variance of the Laplacian; higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_penalty(box, frame_shape, margin=2):
    """1.0 if the box is comfortably inside the frame, decreasing as it touches a border."""
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = box[:4]
    touching = sum([x0 <= margin, y0 <= margin, x1 >= w - margin, y1 >= h - margin])
    return {0: 1.0, 1: 0.6, 2: 0.3}.get(touching, 0.15)


def crop_quality(frame, box, conf=1.0, min_size=8):
    """-> (score, crop) or (0.0, None) if the crop is unusable."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(box[2])), min(h, int(box[3]))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return 0.0, None
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    area = (x1 - x0) * (y1 - y0)
    s = (float(conf)
         * np.sqrt(area)
         * (1.0 + sharpness(gray))
         * edge_penalty(box, frame.shape))
    return float(s), crop


def select_best(frames, boxes, confs=None, k=5):
    """Pick the k best crops of one track.

    frames/boxes are parallel lists for a single track id.
    -> list of (index, score, crop), best first.
    """
    confs = confs if confs is not None else [1.0] * len(boxes)
    scored = []
    for i, (f, b, c) in enumerate(zip(frames, boxes, confs)):
        s, crop = crop_quality(f, b, c)
        if crop is not None and s > 0:
            scored.append((i, s, crop))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def aggregate(embeddings, mode="mean"):
    """Combine per-crop embeddings into one track embedding (L2-normalised)."""
    e = np.asarray(embeddings, dtype=np.float32).reshape(len(embeddings), -1)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    v = e.mean(0) if mode == "mean" else e.max(0)
    return v / max(float(np.linalg.norm(v)), 1e-12)
