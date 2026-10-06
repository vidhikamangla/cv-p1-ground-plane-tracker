"""The tracker. One implementation, two coordinate frames.

`space: image`  - the Kalman filter runs on the foot point in pixels.
`space: ground` - the foot point is projected through H_t = H_ref @ W(t->ref) first,
                  so the filter runs in metres and camera motion never reaches it.

Association is ByteTrack-shaped: high-confidence detections associate first, then the
leftover low-confidence detections get a second pass against still-unmatched tracks.
The second pass is what recovers partially occluded players that the detector scored at
0.2 - without it they become ID switches a few frames later.

Gates, all optional so the ablation can isolate each one:
  chi-square gate  - Mahalanobis distance under the predicted covariance (2 dof)
  colour gate      - Bhattacharyya distance between HSV torso histograms
  speed gate       - reject physically impossible displacements (ground space only,
                     where "11 m/s" actually means something)
"""
import numpy as np

from ..geometry.homography import apply_h, foot_points, propagate_cov
from .appearance import ema_hist, torso_hist
from .association import INF, colour_cost_matrix, iou_matrix, linear_assign
from .kalman import KalmanCV
from .track import CONFIRMED, Track, box_from_foot


class GroundTracker:
    def __init__(self, cfg, fps):
        t = cfg.tracker
        k = cfg.kalman
        self.space = t.space                     # image | ground
        self.cost = t.cost                       # mahalanobis | iou
        self.use_cmc = bool(t.use_cmc)
        self.use_color = bool(t.use_color)
        self.two_stage = bool(t.get("two_stage", True))
        self.high_thresh = float(t.high_thresh)
        self.low_thresh = float(t.low_thresh)
        self.match_thresh = float(t.match_thresh)
        self.iou_match_thresh = float(t.iou_match_thresh)
        self.color_gate = float(t.color_gate)
        self.max_age = int(t.max_age)
        self.min_hits = int(t.min_hits)
        self.max_speed = float(t.get("max_speed_mps", 0) or 0)
        # Heteroscedastic measurement model: detection noise is isotropic in PIXELS,
        # so in ground space each detection gets R = J Rpx J^T from the homography
        # Jacobian instead of one isotropic R in metres that is right at one depth only.
        self.per_point_R = bool(t.get("per_point_R", False))
        self.meas_std_px = float(k.get("meas_std_px", 8.0))
        dt = float(k.dt) if k.get("dt") else 1.0 / float(fps)
        self.dt = dt
        self.kf = KalmanCV(dt, float(k.accel_std), float(k.meas_std))
        self.tracks: list[Track] = []
        Track.reset_ids()

    # ------------------------------------------------------------------ geometry
    def _measure(self, boxes, H_t):
        """Detection boxes -> measurement points in the filter's plane.

        One transform covers every ablation row, which is why the four
        configurations share a single code path:

            image,  no CMC : M_t = I                  (pixels, current frame)
            image,  CMC    : M_t = W(t->ref)           (pixels, reference frame)
            ground, no CMC : M_t = H_ref               (metres, stale calibration)
            ground, CMC    : M_t = H_ref @ W(t->ref)   (metres, correct)

        `space` then only decides which gates are physically meaningful, not how the
        measurement is produced.
        """
        fp = foot_points(boxes)
        if self.space == "ground" and H_t is None:
            raise RuntimeError("ground space needs a homography for this frame")
        return apply_h(H_t, fp) if H_t is not None else fp

    def _predicted_boxes(self, tracks, H_inv):
        """Predicted image-space boxes, for IoU costs."""
        out = []
        for tr in tracks:
            p = tr.predicted_meas()
            if H_inv is None:
                # filter already runs in current-frame pixels
                out.append(box_from_foot(p, tr.wh))
                continue
            img_pt = apply_h(H_inv, p.reshape(1, 2))[0]
            if not np.all(np.isfinite(img_pt)):
                out.append(tr.box.copy())
            else:
                out.append(box_from_foot(img_pt, tr.wh))
        return np.asarray(out, dtype=np.float64).reshape(-1, 4)

    # ------------------------------------------------------------------ costs
    def _cost(self, tracks, boxes, meas, hists, H_inv, R=None, frame_idx=None):
        n, m = len(tracks), len(boxes)
        if n == 0 or m == 0:
            return np.zeros((n, m)), 0.0

        if self.cost == "mahalanobis":
            C = np.empty((n, m))
            for i, tr in enumerate(tracks):
                C[i] = tr.kf.gating_distance(tr.x, tr.P, meas, R=R)
            max_cost = self.match_thresh
            # NaN measurements (points projected behind the camera) can never match
            bad = ~np.all(np.isfinite(meas), axis=1)
            if bad.any():
                C[:, bad] = INF
        else:
            pboxes = self._predicted_boxes(tracks, H_inv)
            C = 1.0 - iou_matrix(pboxes, boxes)
            max_cost = 1.0 - self.iou_match_thresh

        if self.use_color and hists is not None:
            CC = colour_cost_matrix([tr.hist for tr in tracks], hists)
            C = np.where(CC > self.color_gate, INF, C)

        if self.max_speed > 0 and self.space == "ground":
            for i, tr in enumerate(tracks):
                d = np.linalg.norm(meas - tr.pos[None, :], axis=1)
                # Elapsed time since this track was last UPDATED, not one frame: a track
                # that has been lost for 10 frames has had 10 frames to move, and
                # dividing its displacement by a single dt overstates its speed 10x and
                # rejects exactly the re-identifications this gate should allow.
                gap = 1 if frame_idx is None else max(1, frame_idx - tr.last_frame)
                dt_eff = gap * self.dt
                C[i] = np.where(d / dt_eff > self.max_speed, INF, C[i])

        return C, max_cost

    # ------------------------------------------------------------------ main step
    def update(self, frame_idx, boxes, scores, frame_bgr=None, H_t=None, H_inv=None):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        for tr in self.tracks:
            tr.predict()

        meas_all = self._measure(boxes, H_t) if len(boxes) else np.zeros((0, 2))
        R_all = None
        if self.per_point_R and self.space == "ground" and H_t is not None and len(boxes):
            R_all = propagate_cov(H_t, foot_points(boxes), self.meas_std_px)
        hists_all = None
        if self.use_color and frame_bgr is not None and len(boxes):
            hists_all = [torso_hist(frame_bgr, b) for b in boxes]

        hi = np.where(scores >= self.high_thresh)[0]
        lo = np.where((scores < self.high_thresh) & (scores >= self.low_thresh))[0]

        active = [tr for tr in self.tracks if tr.is_active]
        matched_tracks = set()

        def sub(idx):
            h = [hists_all[i] for i in idx] if hists_all is not None else None
            r = R_all[idx] if R_all is not None else None
            return boxes[idx], meas_all[idx], h, r

        # --- stage 1: high-confidence detections
        b1, m1, h1, r1 = sub(hi)
        C1, mc1 = self._cost(active, b1, m1, h1, H_inv, r1, frame_idx)
        matches1, un_tr1, un_d1 = linear_assign(C1, mc1)
        for ti, di in matches1:
            tr = active[ti]
            j = hi[di]
            tr.update(m1[di], boxes[j], scores[j], frame_idx,
                      hist=ema_hist(tr.hist, hists_all[j]) if hists_all else None,
                      R=None if r1 is None else r1[di])
            tr.maybe_confirm(self.min_hits)
            matched_tracks.add(id(tr))

        # --- stage 2: the leftover low-confidence detections rescue existing tracks
        if self.two_stage and len(lo):
            rest = [active[i] for i in un_tr1]
            b2, m2, h2, r2 = sub(lo)
            C2, mc2 = self._cost(rest, b2, m2, h2, H_inv, r2, frame_idx)
            matches2, un_tr2, _ = linear_assign(C2, mc2)
            for ti, di in matches2:
                tr = rest[ti]
                j = lo[di]
                tr.update(m2[di], boxes[j], scores[j], frame_idx,
                          hist=ema_hist(tr.hist, hists_all[j]) if hists_all else None,
                          R=None if r2 is None else r2[di])
                tr.maybe_confirm(self.min_hits)
                matched_tracks.add(id(tr))

        # --- births: only from unmatched *high* detections
        for di in un_d1:
            j = hi[di]
            z = m1[di]
            if not np.all(np.isfinite(z)):
                continue
            born = Track(self.kf, z, boxes[j], scores[j], frame_idx,
                         hist=hists_all[j] if hists_all else None)
            self.tracks.append(born)
            # a track born this frame has not "missed" this frame - without this it is
            # marked missed and, being tentative, removed on the very frame it is created
            matched_tracks.add(id(born))

        # --- lifecycle
        for tr in self.tracks:
            if id(tr) not in matched_tracks:
                tr.mark_missed(self.max_age)
        self.tracks = [tr for tr in self.tracks if tr.is_active]

        return [tr for tr in self.tracks
                if tr.state == CONFIRMED and tr.last_frame == frame_idx]
