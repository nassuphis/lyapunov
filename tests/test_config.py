"""Tests for config module (spec parsing and map configuration)."""

import pytest
import numpy as np


class TestMakeCfg:
    """Test the make_cfg function."""

    def test_make_cfg_basic_1d(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4")

        assert "step" in cfg
        assert "deriv" in cfg
        assert "seq_arr" in cfg
        assert "domain" in cfg
        assert "x0" in cfg
        assert "n_tr" in cfg
        assert "n_it" in cfg
        assert cfg["type"] == "step1d"

    def test_make_cfg_domain_affine(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:3:3.5")

        # domain_affine has 6 elements for affine mapping
        domain_affine = cfg["domain_affine"]
        assert len(domain_affine) == 6

    def test_make_cfg_iter_override(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4,iter:500")
        assert cfg["n_it"] == 500

    def test_make_cfg_trans_override(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4,trans:1000")
        assert cfg["n_tr"] == 1000

    def test_make_cfg_x0_override(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4,x0:0.3")
        assert cfg["x0"] == pytest.approx(0.3)

    def test_make_cfg_histogram_mode(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4,hist:0:1:32")

        assert "vcalc" in cfg
        assert "hcalc" in cfg
        assert "hbins" in cfg
        assert cfg["vcalc"] == 0
        assert cfg["hcalc"] == 1
        assert cfg["hbins"] == 32

    def test_make_cfg_entropy_mode(self):
        from config import make_cfg

        cfg = make_cfg("map:logistic:AB:2:4:2:4,entropy:1,k:64")

        assert cfg.get("entropy_sign") == 1
        assert "omegas" in cfg
        assert len(cfg["omegas"]) == 64


class TestGetMapName:
    """Test the get_map_name function."""

    def test_get_map_name_basic(self):
        from config import get_map_name

        assert get_map_name("map:logistic:AB:2:4:2:4") == "logistic"

    def test_get_map_name_with_options(self):
        from config import get_map_name

        assert get_map_name("map:logistic:AB:2:4:2:4,iter:500,trans:100") == "logistic"


class TestAffine:
    """Test affine domain building."""

    def test_build_affine_domain_basic(self):
        from affine import build_affine_domain

        specdict = {}
        domain = build_affine_domain(specdict, 0.0, 0.0, 1.0, 1.0)

        assert len(domain) == 6
        # LL = (0, 0), UL = (0, 1), LR = (1, 0)
        assert domain[0] == pytest.approx(0.0)  # llx
        assert domain[1] == pytest.approx(0.0)  # lly
        assert domain[2] == pytest.approx(0.0)  # ulx
        assert domain[3] == pytest.approx(1.0)  # uly
        assert domain[4] == pytest.approx(1.0)  # lrx
        assert domain[5] == pytest.approx(0.0)  # lry

    def test_build_affine_domain_with_corners(self):
        from affine import build_affine_domain

        specdict = {
            "ll": ["1", "2"],
            "ur": ["5", "6"],
        }
        domain = build_affine_domain(specdict, 0.0, 0.0, 10.0, 10.0)

        # With ll and ur specified:
        # LL = (1, 2)
        # UL = (1, 6)  (same x as LL, y from UR)
        # LR = (5, 2)  (x from UR, same y as LL)
        assert domain[0] == pytest.approx(1.0)  # llx
        assert domain[1] == pytest.approx(2.0)  # lly
        assert domain[2] == pytest.approx(1.0)  # ulx
        assert domain[3] == pytest.approx(6.0)  # uly
        assert domain[4] == pytest.approx(5.0)  # lrx
        assert domain[5] == pytest.approx(2.0)  # lry

    def test_apply_rot_to_affine_domain(self):
        from affine import apply_rot_to_affine_domain

        # Unit square
        domain = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)

        # Rotate 0.25 turns (90 degrees)
        rotated = apply_rot_to_affine_domain(domain, 0.25)

        # After 90 degree rotation around center (0.5, 0.5):
        # LL (0,0) -> (1, 0)
        # UL (0,1) -> (0, 0)
        # LR (1,0) -> (1, 1)
        assert rotated[0] == pytest.approx(1.0, abs=1e-10)  # llx
        assert rotated[1] == pytest.approx(0.0, abs=1e-10)  # lly

    def test_apply_scale_to_affine_domain(self):
        from affine import apply_scale_to_affine_domain

        # Unit square
        domain = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)

        # Scale by 2x
        scaled = apply_scale_to_affine_domain(domain, 2.0)

        # Center is (0.5, 0.5), scaling by 2 around center:
        # LL (0,0) -> (-0.5, -0.5)
        # UL (0,1) -> (-0.5, 1.5)
        # LR (1,0) -> (1.5, -0.5)
        assert scaled[0] == pytest.approx(-0.5)  # llx
        assert scaled[1] == pytest.approx(-0.5)  # lly
