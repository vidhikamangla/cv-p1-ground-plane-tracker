"""Assignment: scipy's solver must agree with brute force on small problems."""
import itertools

import numpy as np

from src.tracking.association import bhattacharyya, iou_matrix, linear_assign


def brute_force(cost):
    n, m = cost.shape
    best, best_perm = np.inf, None
    for perm in itertools.permutations(range(m), n):
        c = sum(cost[i, perm[i]] for i in range(n))
        if c < best:
            best, best_perm = c, perm
    return best, best_perm


def test_matches_brute_force_optimum():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n, m = rng.integers(1, 6), rng.integers(1, 6)
        if n > m:
            n, m = m, n
        cost = rng.random((n, m)) * 10
        matches, _, _ = linear_assign(cost, max_cost=1e9)
        total = sum(cost[i, j] for i, j in matches)
        ref, _ = brute_force(cost)
        assert abs(total - ref) < 1e-9


def test_gate_rejects_expensive_pairs_without_breaking_the_rest():
    cost = np.array([[0.1, 9.0], [9.0, 0.2]])
    matches, ur, uc = linear_assign(cost, max_cost=1.0)
    assert sorted(matches) == [(0, 0), (1, 1)]
    cost2 = np.array([[0.1, 9.0], [9.0, 5.0]])
    matches2, ur2, uc2 = linear_assign(cost2, max_cost=1.0)
    assert matches2 == [(0, 0)] and ur2 == [1] and uc2 == [1]


def test_empty_inputs():
    m, ur, uc = linear_assign(np.zeros((0, 3)), 1.0)
    assert m == [] and ur == [] and uc == [0, 1, 2]
    m, ur, uc = linear_assign(np.zeros((2, 0)), 1.0)
    assert m == [] and ur == [0, 1] and uc == []


def test_iou_matrix_values():
    a = np.array([[0., 0., 10., 10.]])
    b = np.array([[0., 0., 10., 10.], [5., 5., 15., 15.], [100., 100., 110., 110.]])
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1.0) < 1e-9
    assert abs(m[0, 1] - 25.0 / 175.0) < 1e-9
    assert m[0, 2] < 1e-9


def test_bhattacharyya_bounds():
    h = np.array([0.5, 0.3, 0.2])
    assert bhattacharyya(h, h) < 1e-9
    other = np.array([0.0, 0.0, 1.0])
    d = bhattacharyya(h, other)
    assert 0.0 < d <= 1.0
