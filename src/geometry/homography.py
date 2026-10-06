"""Normalised DLT homography, implemented rather than calling findHomography.

Four point correspondences give 8 equations for the 8 degrees of freedom of H, so the
exact solution is the null space of a 8x9 matrix - obtained here from the smallest
singular vector. Hartley normalisation (translate to centroid, scale so mean distance
is sqrt(2)) is not optional: pixel coordinates around (900, 500) and world coordinates
around (14, 7.5) differ by two orders of magnitude, and the unnormalised system is so
badly conditioned that the SVD returns visible garbage.
"""
import numpy as np


def _normalise(pts):
    """-> (normalised pts, T) with T @ homogeneous(pts) == homogeneous(normalised)"""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    mean_d = d.mean()
    s = np.sqrt(2) / mean_d if mean_d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1]], dtype=np.float64)
    out = (pts - c) * s
    return out, T


def dlt(src, dst):
    """src/dst: [N>=4, 2]. -> H (3x3) mapping src -> dst, normalised so H[2,2] == 1."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError(f"need >=4 matched points, got {len(src)} and {len(dst)}")

    sn, Ts = _normalise(src)
    dn, Td = _normalise(dst)

    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_h(H, pts):
    """pts [N,2] -> [N,2]. Points that map behind the camera (w<=0) come back as NaN."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = hom @ np.asarray(H, dtype=np.float64).T
    w = out[:, 2:3]
    bad = np.abs(w[:, 0]) < 1e-9
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    res = out[:, :2] / w
    res[bad] = np.nan
    return res


def reprojection_error(H, src, dst):
    """Mean and max Euclidean error of H(src) vs dst, in dst units."""
    pred = apply_h(H, src)
    e = np.linalg.norm(pred - np.asarray(dst, dtype=np.float64).reshape(-1, 2), axis=1)
    return float(np.nanmean(e)), float(np.nanmax(e))


def foot_point(box):
    """Bottom-centre of an xyxy box: where the person meets the ground.

    The ground-plane homography is only valid for points *on* the plane, so the box
    centre is the wrong thing to project - it floats a metre above the floor and its
    projected position changes with the person's height and the camera angle.
    """
    x0, y0, x1, y1 = box[:4]
    return np.array([(x0 + x1) / 2.0, y1], dtype=np.float64)


def foot_points(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)


def jacobian(H, pts):
    """d(world)/d(image) of the projective map, evaluated at each image point.

    Needed because measurement noise is isotropic in PIXELS, not in metres: one pixel
    of foot-point error is a couple of centimetres at the near sideline and tens of
    centimetres at the far one. Propagating R_pixels through this Jacobian gives each
    detection its own measurement covariance,

        R_world = J @ R_pixels @ J.T

    which is the difference between a Kalman gain that is right everywhere and one
    that is right at exactly one depth in the frame.

    -> array [N, 2, 2]
    """
    H = np.asarray(H, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    w2 = w * w
    J = np.empty((len(pts), 2, 2))
    J[:, 0, 0] = (H[0, 0] * w - u * H[2, 0]) / w2
    J[:, 0, 1] = (H[0, 1] * w - u * H[2, 1]) / w2
    J[:, 1, 0] = (H[1, 0] * w - v * H[2, 0]) / w2
    J[:, 1, 1] = (H[1, 1] * w - v * H[2, 1]) / w2
    return J


def propagate_cov(H, pts, sigma_px):
    """Isotropic pixel noise -> per-point 2x2 world covariance. [N,2,2]"""
    J = jacobian(H, pts)
    Rp = (float(sigma_px) ** 2) * np.eye(2)
    return J @ Rp @ np.transpose(J, (0, 2, 1))
