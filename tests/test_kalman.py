"""Kalman filter behaviour we can check by hand."""
import numpy as np

from src.tracking.kalman import KalmanCV


def test_constant_velocity_is_tracked_exactly_without_noise():
    kf = KalmanCV(dt=0.04, accel_std=3.0, meas_std=0.35)
    v = np.array([2.0, -1.0])
    p0 = np.array([5.0, 3.0])
    x, P = kf.initiate(p0)
    for k in range(1, 200):
        x, P = kf.predict(x, P)
        x, P = kf.update(x, P, p0 + v * (k * kf.dt))
    assert np.allclose(x[:2], p0 + v * (199 * kf.dt), atol=0.05)
    assert np.allclose(x[2:4], v, atol=0.05)


def test_covariance_stays_symmetric_and_positive_definite():
    """Joseph-form update must not let P drift asymmetric over a long run."""
    kf = KalmanCV(dt=0.04, accel_std=3.0, meas_std=0.35)
    rng = np.random.default_rng(0)
    x, P = kf.initiate(np.array([0.0, 0.0]))
    for _ in range(2000):
        x, P = kf.predict(x, P)
        x, P = kf.update(x, P, rng.normal(0, 0.35, 2))
        assert np.allclose(P, P.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(P) > 0)


def test_prediction_grows_uncertainty_and_update_shrinks_it():
    kf = KalmanCV(dt=0.04, accel_std=3.0, meas_std=0.35)
    x, P = kf.initiate(np.array([1.0, 1.0]))
    tr0 = np.trace(P)
    x, P = kf.predict(x, P)
    assert np.trace(P) > tr0
    tr1 = np.trace(P)
    x, P = kf.update(x, P, np.array([1.0, 1.0]))
    assert np.trace(P) < tr1


def test_gating_distance_matches_explicit_mahalanobis():
    kf = KalmanCV(dt=0.04, accel_std=3.0, meas_std=0.35)
    x, P = kf.initiate(np.array([10.0, 4.0]))
    x, P = kf.predict(x, P)
    zs = np.array([[10.0, 4.0], [11.0, 4.5], [30.0, 20.0]])
    d = kf.gating_distance(x, P, zs)
    zhat, S = kf.project(x, P)
    Si = np.linalg.inv(S)
    ref = np.array([(z - zhat) @ Si @ (z - zhat) for z in zs])
    assert np.allclose(d, ref, atol=1e-9)
    assert d[0] < d[1] < d[2]


def test_process_noise_scales_with_accel_std():
    """accel_std is a physical quantity: doubling it must quadruple Q (variance)."""
    a = KalmanCV(0.04, 3.0, 0.35)
    b = KalmanCV(0.04, 6.0, 0.35)
    assert np.allclose(b.Q, 4.0 * a.Q)
