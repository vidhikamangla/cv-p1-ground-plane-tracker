"""Bird's-eye-view render of ground-space tracks on a court diagram."""
import numpy as np

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None

PALETTE = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (240, 50, 230),
           (70, 240, 240), (210, 245, 60), (250, 190, 190), (170, 110, 40),
           (255, 225, 25), (0, 128, 128)]


def colour_for(tid):
    return PALETTE[int(tid) % len(PALETTE)]


class CourtCanvas:
    """Metres -> pixels for a top-down court, with a margin."""

    def __init__(self, length=28.0, width=15.0, px_per_m=26, margin_m=1.5):
        self.L, self.W = float(length), float(width)
        self.s = float(px_per_m)
        self.m = float(margin_m)
        self.w = int(round((self.L + 2 * self.m) * self.s))
        self.h = int(round((self.W + 2 * self.m) * self.s))

    def to_px(self, pt):
        x, y = float(pt[0]), float(pt[1])
        return (int(round((x + self.m) * self.s)), int(round((y + self.m) * self.s)))

    def base(self):
        img = np.full((self.h, self.w, 3), 32, np.uint8)
        tl, br = self.to_px((0, 0)), self.to_px((self.L, self.W))
        cv2.rectangle(img, tl, br, (200, 200, 200), 2)
        cv2.line(img, self.to_px((self.L / 2, 0)), self.to_px((self.L / 2, self.W)),
                 (200, 200, 200), 2)
        cv2.circle(img, self.to_px((self.L / 2, self.W / 2)),
                   int(1.8 * self.s), (200, 200, 200), 2)
        for xm in (0, self.L):
            cv2.circle(img, self.to_px((xm, self.W / 2)), int(1.25 * self.s),
                       (140, 140, 140), 1)
        return img

    def render(self, positions, trails=None, trail_len=50, speeds=None):
        """positions: {tid: (x_m, y_m)}; trails: {tid: [(x,y), ...]}"""
        img = self.base()
        if trails:
            for tid, pts in trails.items():
                pts = pts[-trail_len:]
                if len(pts) < 2:
                    continue
                arr = np.asarray([self.to_px(p) for p in pts], np.int32)
                cv2.polylines(img, [arr], False, colour_for(tid), 1, cv2.LINE_AA)
        for tid, p in positions.items():
            c = self.to_px(p)
            cv2.circle(img, c, 7, colour_for(tid), -1)
            cv2.circle(img, c, 7, (20, 20, 20), 1)
            label = str(tid)
            if speeds and tid in speeds:
                label += f" {speeds[tid]:.1f}"
            cv2.putText(img, label, (c[0] + 9, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, colour_for(tid), 1, cv2.LINE_AA)
        return img
