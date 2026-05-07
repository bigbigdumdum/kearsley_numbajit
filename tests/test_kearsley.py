import numpy as np
import pytest


def test_import():
    from kearsley import fit, fit_transform, transform, fill_rot_and_trans


def test_fit_transform_identity():
    """Fitting identical point sets should give RMSD ~ 0 and identity-like rotation."""
    from kearsley import fit_transform

    rng = np.random.default_rng(42)
    u = rng.random((10, 3))

    transformed_v, rmsd, so, rotation, translation = fit_transform(u, u.copy(), 1.0)

    assert rmsd < 1e-6, f"Expected near-zero RMSD for identical sets, got {rmsd}"
    assert so == pytest.approx(100, abs=1e-6), f"Expected SO=100, got {so}"
    np.testing.assert_allclose(transformed_v, u, atol=1e-6)


def test_fit_transform_translation():
    """A pure translation should be recovered exactly."""
    from kearsley import fit_transform

    rng = np.random.default_rng(0)
    u = rng.random((20, 3))
    shift = np.array([3.0, -1.5, 2.0])
    v = u + shift

    transformed_v, rmsd, so, rotation, translation = fit_transform(u, v, 1.0)

    assert rmsd < 1e-5, f"Expected near-zero RMSD after translation recovery, got {rmsd}"
    np.testing.assert_allclose(transformed_v, u, atol=1e-5)


def test_fit_returns_rmsd():
    from kearsley import fit

    rng = np.random.default_rng(7)
    u = rng.random((15, 3))
    v = rng.random((15, 3))

    rmsd, q, centroid_u, centroid_v = fit(u, v)

    assert isinstance(rmsd, float)
    assert rmsd >= 0.0
    assert q.shape == (4,)
    assert centroid_u.shape == (3,)
    assert centroid_v.shape == (3,)


def test_transform_different_size():
    """transform() should accept M != N."""
    from kearsley import fit_transform, transform

    rng = np.random.default_rng(99)
    u = rng.random((10, 3))
    v = rng.random((10, 3))
    w = rng.random((50, 3))  # larger set

    _, _, _, rotation, translation = fit_transform(u, v, 1.0)
    transformed_w = transform(w, rotation, translation)

    assert transformed_w.shape == (50, 3)
