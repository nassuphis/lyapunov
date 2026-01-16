"""Tests for field computation kernels."""

import pytest
import numpy as np


class TestLyapunovFields:
    """Test Lyapunov exponent field computations."""

    def test_lyapunov_field_1d_basic(self):
        from maps import build_map, seq_to_array
        from maps.fields import lyapunov_field_1d

        cfg = build_map("logistic")
        step = cfg["step"]
        deriv = cfg["deriv"]
        seq = seq_to_array("AB")
        domain = np.array([2.0, 2.0, 4.0, 2.0, 2.0, 4.0], dtype=np.float64)
        params = np.empty(0, dtype=np.float64)

        field = lyapunov_field_1d(
            step=step,
            deriv=deriv,
            seq=seq,
            domain=domain,
            pix=10,
            x0=0.5,
            n_transient=100,
            n_iter=100,
            eps=1e-12,
            params=params,
        )

        assert field.shape == (10, 10)
        assert field.dtype == np.float64
        assert np.all(np.isfinite(field))

    def test_lyapunov_field_1d_chaotic_region(self):
        """Test that chaotic region has positive Lyapunov exponent."""
        from maps import build_map, seq_to_array
        from maps.fields import lyapunov_field_1d

        cfg = build_map("logistic")
        # Domain where r=4 (fully chaotic)
        domain = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float64)
        seq = seq_to_array("AB")
        params = np.empty(0, dtype=np.float64)

        field = lyapunov_field_1d(
            step=cfg["step"],
            deriv=cfg["deriv"],
            seq=seq,
            domain=domain,
            pix=5,
            x0=0.5,
            n_transient=500,
            n_iter=500,
            eps=1e-12,
            params=params,
        )

        # At r=4, Lyapunov exponent should be ln(2) ≈ 0.693
        assert np.mean(field) > 0.5

    def test_lyapunov_field_1d_stable_region(self):
        """Test that stable region has negative Lyapunov exponent."""
        from maps import build_map, seq_to_array
        from maps.fields import lyapunov_field_1d

        cfg = build_map("logistic")
        # Domain where r=2.5 (stable fixed point)
        domain = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float64)
        seq = seq_to_array("AB")
        params = np.empty(0, dtype=np.float64)

        field = lyapunov_field_1d(
            step=cfg["step"],
            deriv=cfg["deriv"],
            seq=seq,
            domain=domain,
            pix=5,
            x0=0.5,
            n_transient=500,
            n_iter=500,
            eps=1e-12,
            params=params,
        )

        # At r=2.5, Lyapunov exponent should be negative
        assert np.mean(field) < 0


class TestHistogramFields:
    """Test histogram-based field computations."""

    def test_hist_field_1d_basic(self):
        from maps import build_map, seq_to_array
        from maps.fields import hist_field_1d

        cfg = build_map("logistic")
        seq = seq_to_array("AB")
        domain = np.array([2.0, 2.0, 4.0, 2.0, 2.0, 4.0], dtype=np.float64)
        params = np.empty(0, dtype=np.float64)

        field = hist_field_1d(
            step=cfg["step"],
            seq=seq,
            domain=domain,
            pix=10,
            x0=0.5,
            n_transient=100,
            n_iter=100,
            vcalc=0,  # copy
            hcalc=0,  # std
            hbins=32,
            params=params,
        )

        assert field.shape == (10, 10)
        assert field.dtype == np.float64
        assert np.all(np.isfinite(field))


class TestSpectralFields:
    """Test spectral entropy field computations."""

    def test_entropy_field_1d_basic(self):
        from maps import build_map, seq_to_array
        from maps.fields import entropy_field_1d

        cfg = build_map("logistic")
        seq = seq_to_array("AB")
        domain = np.array([2.0, 2.0, 4.0, 2.0, 2.0, 4.0], dtype=np.float64)
        params = np.empty(0, dtype=np.float64)
        # Frequencies
        omegas = np.linspace(0.01, np.pi, 16)

        field = entropy_field_1d(
            step=cfg["step"],
            seq=seq,
            domain=domain,
            pix=10,
            x0=0.5,
            n_transient=100,
            n_iter=100,
            omegas=omegas,
            params=params,
        )

        assert field.shape == (10, 10)
        assert field.dtype == np.float64
        assert np.all(np.isfinite(field))
        # Entropy should be in [0, 1]
        assert np.all(field >= 0)
        assert np.all(field <= 1)


class TestHistHelpers:
    """Test histogram helper functions."""

    def test_hist_fixed_bins_inplace(self):
        from maps.hist_helpers import hist_fixed_bins_inplace

        bins = np.zeros(10, dtype=np.int64)
        values = np.array([0.1, 0.2, 0.3, 0.5, 0.9], dtype=np.float64)
        hist_fixed_bins_inplace(bins, values, 0.0, 1.0)

        assert np.sum(bins) == 5  # All values should be binned
        assert bins[1] == 1  # 0.1 in bin 1
        assert bins[2] == 1  # 0.2 in bin 2
        assert bins[3] == 1  # 0.3 in bin 3
        assert bins[5] == 1  # 0.5 in bin 5
        assert bins[9] == 1  # 0.9 in bin 9

    def test_transform_values_copy(self):
        from maps.hist_helpers import transform_values

        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        vs = np.empty_like(xs)
        transform_values(0, xs, vs)  # vcalc=0 is copy

        np.testing.assert_array_equal(xs, vs)

    def test_transform_values_slope(self):
        from maps.hist_helpers import transform_values

        xs = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float64)
        vs = np.empty_like(xs)
        transform_values(1, xs, vs)  # vcalc=1 is slope

        # Slopes: [-, 1, 2, 3, 4], first element is copied from second
        assert vs[1] == 1.0
        assert vs[2] == 2.0
        assert vs[3] == 3.0
        assert vs[4] == 4.0


class TestMapLogicalToPhysical:
    """Test coordinate mapping."""

    def test_map_logical_corners(self):
        from maps.fields import map_logical_to_physical

        # Simple axis-aligned domain: [0,0] to [1,1]
        domain = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        # LL=(0,0), UL=(0,1), LR=(1,0)

        # Test corners
        A, B = map_logical_to_physical(domain, 0.0, 0.0)
        assert (A, B) == pytest.approx((0.0, 0.0))

        A, B = map_logical_to_physical(domain, 1.0, 0.0)
        assert (A, B) == pytest.approx((1.0, 0.0))

        A, B = map_logical_to_physical(domain, 0.0, 1.0)
        assert (A, B) == pytest.approx((0.0, 1.0))

        A, B = map_logical_to_physical(domain, 1.0, 1.0)
        assert (A, B) == pytest.approx((1.0, 1.0))

    def test_map_logical_center(self):
        from maps.fields import map_logical_to_physical

        domain = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 0.0], dtype=np.float64)

        A, B = map_logical_to_physical(domain, 0.5, 0.5)
        assert (A, B) == pytest.approx((1.0, 1.0))
