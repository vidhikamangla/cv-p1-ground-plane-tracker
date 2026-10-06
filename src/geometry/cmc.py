"""Camera motion compensation: estimate W(t -> ref), the homography that warps the
current frame back onto the reference frame in which the court was calibrated.

Why this exists: SportsMOT cameras pan and zoom. A homography clicked once on frame
`ref_frame` is only valid for that frame. Composing it with W(t->ref) gives

    H_t = H_ref @ W(t->ref)        image(t) -> world (metres)

so a single calibration serves the whole clip. In ground space camera motion is
therefore *absorbed into the projection* and never reaches the Kalman filter, which is
the main argument for tracking on the plane: the filter only ever sees real player
motion, so its process noise stays physically meaningful.

Strategy: match ORB features directly against the reference frame, which avoids the
drift that frame-to-frame chaining accumulates over ~750 frames. When a direct match
is too weak (fast pan, motion blur), fall back to chaining from the previous frame so
the estimate degrades gracefully instead of jumping.

Player pixels are masked out before feature detection - matching on moving players is
exactly the failure mode this is meant to avoid.
"""
import cv2
import numpy as np


class CMC:
    def __init__(self, ref_gray, max_features=3000, ransac_thresh=3.0,
                 min_inliers=25, ratio=0.75):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ransac_thresh = float(ransac_thresh)
        self.min_inliers = int(min_inliers)
        self.ratio = float(ratio)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, None)
        self.prev_gray = ref_gray
        self.prev_W = np.eye(3)
        self.stats = {"direct": 0, "chained": 0, "identity": 0}

    @staticmethod
    def _mask_from_boxes(shape, boxes, pad=4):
        m = np.full(shape[:2], 255, np.uint8)
        for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
            x0 = max(0, int(b[0]) - pad); y0 = max(0, int(b[1]) - pad)
            x1 = min(shape[1], int(b[2]) + pad); y1 = min(shape[0], int(b[3]) + pad)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 0
        return m

    def _match(self, des_a, kp_a, des_b, kp_b):
        """-> (src_pts, dst_pts) from a to b, after Lowe ratio test."""
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return None, None
        knn = self.matcher.knnMatch(des_a, des_b, k=2)
        src, dst = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                src.append(kp_a[m.queryIdx].pt)
                dst.append(kp_b[m.trainIdx].pt)
        if len(src) < 8:
            return None, None
        return np.float32(src).reshape(-1, 1, 2), np.float32(dst).reshape(-1, 1, 2)

    def estimate(self, gray, det_boxes=()):
        """-> (W mapping current image -> reference image, info dict)"""
        mask = self._mask_from_boxes(gray.shape, det_boxes) if len(det_boxes) else None
        kp, des = self.orb.detectAndCompute(gray, mask)

        src, dst = self._match(des, kp, self.ref_des, self.ref_kp)
        if src is not None:
            W, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
            n_in = int(inl.sum()) if inl is not None else 0
            if W is not None and n_in >= self.min_inliers:
                self.prev_W, self.prev_gray = W, gray
                self.stats["direct"] += 1
                return W, {"mode": "direct", "inliers": n_in}

        # fall back: chain from the previous frame through the previous estimate
        kp_p, des_p = self.orb.detectAndCompute(self.prev_gray, None)
        src, dst = self._match(des, kp, des_p, kp_p)
        if src is not None:
            W_rel, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
            n_in = int(inl.sum()) if inl is not None else 0
            if W_rel is not None and n_in >= 8:
                W = self.prev_W @ W_rel
                self.prev_W, self.prev_gray = W, gray
                self.stats["chained"] += 1
                return W, {"mode": "chained", "inliers": n_in}

        self.stats["identity"] += 1
        return self.prev_W.copy(), {"mode": "identity", "inliers": 0}


def warp_magnitude(W, shape):
    """Mean pixel displacement of the image corners under W.

    Used to decide whether the camera actually moved. An ORB+RANSAC homography is never
    exactly the identity even on a locked-off camera, and that residual jitter is pure
    noise: harmless in pixels, amplified into metres once it passes through H_ref. So
    when the measured motion is below a threshold we snap W to the identity instead of
    trusting a few pixels of estimator noise.
    """
    import numpy as np
    h, w = shape[:2]
    corners = np.array([[0., 0.], [w, 0.], [w, h], [0., h]], dtype=np.float64)
    hom = np.concatenate([corners, np.ones((4, 1))], axis=1)
    out = hom @ np.asarray(W, dtype=np.float64).T
    ww = out[:, 2:3]
    ww = np.where(np.abs(ww) < 1e-9, 1e-9, ww)
    moved = out[:, :2] / ww
    return float(np.linalg.norm(moved - corners, axis=1).mean())
