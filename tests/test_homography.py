"""DLT homography: round trips and conditioning."""
import numpy as np

from src.geometry.homography import apply_h, dlt, foot_point, foot_points, reprojection_error


def test_exact_recovery_of_a_known_homography():
    H = np.array([[1.2, 0.1, 30.0],
                  [-0.05, 0.9, -12.0],
                  [0.0002, 0.0001, 1.0]])
    src = np.array([[0., 0.], [100., 0.], [100., 80.], [0., 80.], [43., 21.]])
    dst = apply_h(H, src)
    Hh = dlt(src, dst)
    assert np.allclose(Hh / Hh[2, 2], H / H[2, 2], atol=1e-6)
    mean_e, max_e = reprojection_error(Hh, src, dst)
    assert max_e < 1e-6


def test_round_trip_through_the_inverse():
    src = np.array([[100., 700.], [1200., 690.], [1150., 420.], [250., 430.]])
    dst = np.array([[0., 0.], [28., 0.], [28., 15.], [0., 15.]])       # a basketball court
    H = dlt(src, dst)
    back = apply_h(np.linalg.inv(H), apply_h(H, src))
    assert np.allclose(back, src, atol=1e-6)


def test_realistic_pixel_to_metre_scale_is_well_conditioned():
    """The whole reason for Hartley normalisation: pixels ~1e3, metres ~1e1."""
    img = np.array([[180., 980.], [1740., 970.], [1500., 520.], [420., 525.]])
    world = np.array([[0., 0.], [28., 0.], [28., 15.], [0., 15.]])
    H = dlt(img, world)
    mean_e, max_e = reprojection_error(H, img, world)
    assert max_e < 1e-6, f"max reprojection error {max_e}"


def test_foot_point_is_bottom_centre():
    box = np.array([10., 20., 30., 60.])
    assert np.allclose(foot_point(box), [20., 60.])
    assert np.allclose(foot_points(box[None, :]), [[20., 60.]])


def test_too_few_points_raises():
    try:
        dlt(np.zeros((3, 2)), np.zeros((3, 2)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for 3 correspondences")
