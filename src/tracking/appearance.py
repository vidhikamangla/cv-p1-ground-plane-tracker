"""HSV torso histogram as a weak appearance cue.

Deliberately not a learned ReID embedding: on a 2-hour build a colour histogram is
honest, fast and explainable, and for team sports it captures the one thing that
matters most (kit colour). The torso crop is the middle 50% of the box width and the
20-60% band of its height - the head and legs are mostly background and grass.
"""
import cv2
import numpy as np

H_BINS, S_BINS = 8, 8


def torso_hist(frame, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    h, w = frame.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 2 or bh <= 4:
        return None
    cx0 = max(0, x0 + bw // 4); cx1 = min(w, x1 - bw // 4)
    cy0 = max(0, y0 + int(0.2 * bh)); cy1 = min(h, y0 + int(0.6 * bh))
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return None
    crop = frame[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).flatten().astype(np.float64)


def ema_hist(old, new, alpha=0.9):
    """Slow EMA: appearance should drift with lighting, not jump with one bad crop."""
    if old is None:
        return new
    if new is None:
        return old
    out = alpha * old + (1 - alpha) * new
    s = out.sum()
    return out / s if s > 0 else out
