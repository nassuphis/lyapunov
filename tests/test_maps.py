"""Tests for the maps package."""

import pytest
import numpy as np


class TestMapsImports:
    """Test that all maps package imports work correctly."""

    def test_import_maps_package(self):
        import maps
        assert hasattr(maps, 'MAP_TEMPLATES')
        assert hasattr(maps, 'build_map')

    def test_import_defaults(self):
        from maps import (
            DEFAULT_MAP_NAME,
            DEFAULT_SEQ,
            DEFAULT_TRANS,
            DEFAULT_ITER,
            DEFAULT_X0,
            DEFAULT_EPS_LYAP,
            DEFAULT_CLIP,
            DEFAULT_GAMMA,
        )
        assert DEFAULT_MAP_NAME == "logistic"
        assert DEFAULT_SEQ == "AB"
        assert isinstance(DEFAULT_TRANS, int)
        assert isinstance(DEFAULT_ITER, int)

    def test_import_template_dicts(self):
        from maps import MAPS_STEP1D, MAPS_STEP2D, MAPS_STEP2D_AB, MAPS_STEP2D_XY0
        assert "logistic" in MAPS_STEP1D
        assert isinstance(MAPS_STEP2D, dict)
        assert isinstance(MAPS_STEP2D_AB, dict)
        assert isinstance(MAPS_STEP2D_XY0, dict)

    def test_import_build_functions(self):
        from maps import (
            sympy_deriv,
            sympy_jacobian_2d,
            build_map,
            substitute_common,
        )
        assert callable(sympy_deriv)
        assert callable(build_map)

    def test_import_sequence_functions(self):
        from maps import (
            SEQ_ALLOWED_RE,
            looks_like_sequence_token,
            decode_sequence_token,
            seq_to_array,
        )
        assert callable(looks_like_sequence_token)
        assert callable(seq_to_array)

    def test_import_fields(self):
        from maps.fields import (
            lyapunov_field_1d,
            lyapunov_field_2d,
            lyapunov_field_2d_ab,
            entropy_field_1d,
            hist_field_1d,
        )
        # These are numba-compiled functions
        assert callable(lyapunov_field_1d)

    def test_import_functions_namespace(self):
        from maps import functions
        assert hasattr(functions, 'NS')
        assert 'sin' in functions.NS
        assert 'cos' in functions.NS


class TestMapTemplates:
    """Test map template structure and content."""

    def test_map_templates_combined(self):
        from maps import MAP_TEMPLATES, MAPS_STEP1D, MAPS_STEP2D, MAPS_STEP2D_AB, MAPS_STEP2D_XY0
        # All individual dicts should be in MAP_TEMPLATES
        for name in MAPS_STEP1D:
            assert name in MAP_TEMPLATES
        for name in MAPS_STEP2D:
            assert name in MAP_TEMPLATES
        for name in MAPS_STEP2D_AB:
            assert name in MAP_TEMPLATES
        for name in MAPS_STEP2D_XY0:
            assert name in MAP_TEMPLATES

    def test_logistic_template(self):
        from maps import MAP_TEMPLATES
        logistic = MAP_TEMPLATES["logistic"]
        # Type is inferred from presence of expr vs expr_x/expr_y
        assert "expr" in logistic
        assert "domain" in logistic
        assert len(logistic["domain"]) == 4

    def test_henon_template(self):
        from maps import MAP_TEMPLATES
        if "henon" in MAP_TEMPLATES:
            henon = MAP_TEMPLATES["henon"]
            assert henon["type"] in ("step2d", "step2d_ab")
            assert "expr_x" in henon
            assert "expr_y" in henon


class TestBuildMap:
    """Test the build_map function."""

    def test_build_logistic(self):
        from maps import build_map
        cfg = build_map("logistic")
        assert "step" in cfg
        assert "deriv" in cfg
        assert callable(cfg["step"])
        assert callable(cfg["deriv"])

    def test_build_map_includes_domain(self):
        from maps import build_map
        cfg = build_map("logistic")
        assert "domain" in cfg
        assert len(cfg["domain"]) == 4

    def test_build_unknown_map_raises(self):
        from maps import build_map
        with pytest.raises(KeyError):
            build_map("nonexistent_map_xyz")


class TestSequenceFunctions:
    """Test A/B sequence handling."""

    def test_looks_like_sequence_token(self):
        from maps import looks_like_sequence_token
        assert looks_like_sequence_token("AB")
        assert looks_like_sequence_token("AABB")
        assert looks_like_sequence_token("ABBA")
        assert not looks_like_sequence_token("2.5")
        assert not looks_like_sequence_token("hello")

    def test_seq_to_array(self):
        from maps import seq_to_array
        arr = seq_to_array("AB")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype in (np.int8, np.int32, np.int64)  # dtype may vary
        assert len(arr) == 2
        assert arr[0] == 0  # A
        assert arr[1] == 1  # B

    def test_seq_to_array_longer(self):
        from maps import seq_to_array
        arr = seq_to_array("AABB")
        assert len(arr) == 4
        assert list(arr) == [0, 0, 1, 1]


class TestSymbolicDerivatives:
    """Test symbolic derivative computation."""

    def test_sympy_deriv(self):
        from maps import sympy_deriv
        # d/dx of r*x*(1-x) = r*(1-2x)
        deriv = sympy_deriv("r*x*(1-x)")
        assert "r" in deriv
        assert "x" in deriv

    def test_sympy_jacobian_2d(self):
        from maps import sympy_jacobian_2d
        jac = sympy_jacobian_2d("1 - a*x**2 + y", "b*x")
        assert len(jac) == 4  # dXdx, dXdy, dYdx, dYdy
