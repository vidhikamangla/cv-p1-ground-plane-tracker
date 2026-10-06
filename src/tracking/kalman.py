"""Constant-velocity Kalman filter, written out rather than imported.

State is [x, y, vx, vy] on the *measurement plane*: metres on the ground plane when
the tracker runs in ground space, pixels when it runs in image space. The same filter
serves both, which is what makes the space ablation a fair comparison - only the
coordinate frame changes, not the estimator.

Process noise is derived from a continuous white-noise-acceleration model, so
`accel_std` is a physical quantity (m/s^2) rather than a tuning knob with no units.
That is the practical argument for tracking on the ground plane: 3.0 m/s^2 is "a
sprinting human changing pace", whereas the pixel-space equivalent depends on where
in the frame the person happens to be standing.
"""
import numpy as np


class KalmanCV:
    ndim = 2            # measurement dimensionality (x, y)

    def __init__(self, dt, accel_std, meas_std):
        self.dt = float(dt)
        self.accel_std = float(accel_std)
        self.meas_std = float(meas_std)

        dt = self.dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        # integral of F(t) * [0,0,a,a]^T over the step, for white-noise acceleration
        q = self.accel_std ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        self.Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                               [0, dt4 / 4, 0, dt3 / 2],
                               [dt3 / 2, 0, dt2, 0],
                               [0, dt3 / 2, 0, dt2]], dtype=np.float64)
        self.R = (self.meas_std ** 2) * np.eye(2, dtype=np.float64)

    def initiate(self, z):
        """z = (x, y). Velocity unknown -> large but finite initial variance."""
        x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        # position uncertainty = measurement noise; velocity uncertainty = what a
        # person could plausibly be doing, squared
        v0 = (4.0 * self.accel_std * self.dt + 2.0) ** 2
        P = np.diag([self.meas_std ** 2, self.meas_std ** 2, v0, v0])
        return x, P

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def project(self, x, P, R=None):
        """-> (measurement mean, measurement covariance incl. sensor noise)

        `R` overrides the default isotropic noise with a per-measurement covariance,
        which is how ground-plane tracking gets a measurement model that is correct at
        every depth rather than at one.
        """
        Rm = self.R if R is None else np.asarray(R, dtype=np.float64).reshape(2, 2)
        return self.H @ x, self.H @ P @ self.H.T + Rm

    def update(self, x, P, z, R=None):
        """Joseph-form covariance update: stays symmetric positive-definite even after
        thousands of steps in float, which the textbook (I-KH)P form does not."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        Rm = self.R if R is None else np.asarray(R, dtype=np.float64).reshape(2, 2)
        zhat, S = self.project(x, P, Rm)
        K = P @ self.H.T @ np.linalg.inv(S)
        y = z - zhat
        x = x + K @ y
        I_KH = np.eye(4) - K @ self.H
        P = I_KH @ P @ I_KH.T + K @ Rm @ K.T
        return x, P

    def gating_distance(self, x, P, zs, R=None):
        """Squared Mahalanobis distance from the predicted measurement to each z.

        Compared against a chi-square threshold with 2 dof: 9.21 at p=0.99.
        Using the *projected* covariance (position + sensor noise) rather than raw
        position variance is what makes the gate scale-correct for a freshly born
        track, whose velocity is still unknown.
        """
        zs = np.asarray(zs, dtype=np.float64).reshape(-1, 2)
        if R is None:
            zhat, S = self.project(x, P)
            d = zs - zhat
            L = np.linalg.cholesky(S)
            return (np.linalg.solve(L, d.T) ** 2).sum(axis=0)
        # per-measurement R: S differs per detection, so solve one at a time
        R = np.asarray(R, dtype=np.float64).reshape(-1, 2, 2)
        zhat = self.H @ x
        base = self.H @ P @ self.H.T
        out = np.empty(len(zs))
        for i, (z, Ri) in enumerate(zip(zs, R)):
            S = base + Ri
            dd = (z - zhat).reshape(2, 1)
            try:
                L = np.linalg.cholesky(S)
                out[i] = float((np.linalg.solve(L, dd) ** 2).sum())
            except np.linalg.LinAlgError:
                out[i] = 1e9
        return out
        d = zs - zhat
        L = np.linalg.cholesky(S)
        sol = np.linalg.solve(L, d.T)
        return (sol ** 2).sum(axis=0)
