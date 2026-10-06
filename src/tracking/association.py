"""Cost matrices and the assignment step.

Hungarian is scipy's: it is a solved problem and nobody is interviewing on LAPJV.
The costs and the gating are ours, because that is where tracking decisions live.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

INF = 1e5


def iou_matrix(a, b):
    """a [N,4], b [M,4] xyxy -> IoU [N,M]"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def bhattacharyya(h1, h2):
    """Distance in [0,1] between two L1-normalised histograms."""
    bc = np.sqrt(np.maximum(h1 * h2, 0)).sum()
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def colour_cost_matrix(track_hists, det_hists):
    n, m = len(track_hists), len(det_hists)
    C = np.zeros((n, m))
    for i, th in enumerate(track_hists):
        for j, dh in enumerate(det_hists):
            C[i, j] = 1.0 if th is None or dh is None else bhattacharyya(th, dh)
    return C


def linear_assign(cost, max_cost):
    """-> (matches [(i,j)], unmatched_rows, unmatched_cols), rejecting pairs above
    `max_cost` *after* solving, which is what keeps the gate from distorting the
    global optimum."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    c = np.where(np.isfinite(cost), cost, INF)
    ri, ci = linear_sum_assignment(c)
    matches, ur, uc = [], set(range(n)), set(range(m))
    for i, j in zip(ri, ci):
        if c[i, j] <= max_cost:
            matches.append((int(i), int(j)))
            ur.discard(int(i)); uc.discard(int(j))
    return matches, sorted(ur), sorted(uc)
