"""Track lifecycle.

tentative -> confirmed requires `min_hits` hits, which suppresses the one-frame false
positives that a deliberately low detector threshold lets through. confirmed -> lost ->
removed after `max_age` unmatched frames. Only confirmed tracks are written out, so the
detector can run at conf 0.10 without flooding the results file.

Size is carried as an EMA of width/height rather than being filtered as part of the
state. A 4-state constant-velocity filter on the foot point plus an EMA on the box size
is enough for association, and it keeps the state vector identical in image and ground
space - which is what makes the space ablation compare only the coordinate frame.
"""
import numpy as np

TENTATIVE, CONFIRMED, LOST, REMOVED = "tentative", "confirmed", "lost", "removed"


def box_from_foot(pt, wh):
    """(cx, bottom) + (w, h) -> xyxy box."""
    cx, bottom = float(pt[0]), float(pt[1])
    w, h = float(wh[0]), float(wh[1])
    return np.array([cx - w / 2.0, bottom - h, cx + w / 2.0, bottom], dtype=np.float64)


class Track:
    _next_id = 1

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1

    def __init__(self, kf, meas, box, score, frame_idx, hist=None):
        self.kf = kf
        self.x, self.P = kf.initiate(meas)
        self.box = np.asarray(box[:4], dtype=np.float64)
        self.wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float64)
        self.score = float(score)
        self.hist = hist
        self.id = Track._next_id
        Track._next_id += 1
        self.state = TENTATIVE
        self.hits = 1
        self.age = 0                 # frames since last match
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.trace = [(frame_idx, self.x[:2].copy())]

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:4].copy()

    def predicted_meas(self):
        """Predicted measurement point, in whatever plane the filter runs on."""
        return self.kf.project(self.x, self.P)[0]

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        self.age += 1

    def update(self, meas, box, score, frame_idx, hist=None, wh_alpha=0.6, R=None):
        self.x, self.P = self.kf.update(self.x, self.P, meas, R=R)
        box = np.asarray(box[:4], dtype=np.float64)
        self.box = box
        new_wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float64)
        self.wh = wh_alpha * self.wh + (1 - wh_alpha) * new_wh
        self.score = float(score)
        self.hits += 1
        self.age = 0
        self.last_frame = frame_idx
        if hist is not None:
            self.hist = hist
        self.trace.append((frame_idx, self.x[:2].copy()))

    def mark_missed(self, max_age):
        if self.state == TENTATIVE:
            self.state = REMOVED              # unconfirmed tracks die immediately
        elif self.age > max_age:
            self.state = REMOVED
        else:
            self.state = LOST

    def maybe_confirm(self, min_hits):
        if self.state in (TENTATIVE, LOST) and self.hits >= min_hits:
            self.state = CONFIRMED

    @property
    def is_active(self):
        return self.state in (TENTATIVE, CONFIRMED, LOST)
