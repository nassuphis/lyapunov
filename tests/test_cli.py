"""Tests for CLI end-to-end functionality."""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path


class TestCLIBasic:
    """Test basic CLI invocations."""

    def test_cli_help(self):
        result = subprocess.run(
            ["uv", "run", "python", "lyapunov_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "lyapunov" in result.stdout.lower()

    def test_cli_render_1d_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "test.jpg")
            result = subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,iter:50",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            assert result.returncode == 0
            # Check that output file was created (with slot number)
            files = list(Path(tmpdir).glob("test_*.jpg"))
            assert len(files) >= 1

    def test_cli_render_histogram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "hist.jpg")
            result = subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,hist:0:1:16,iter:50",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            assert result.returncode == 0
            files = list(Path(tmpdir).glob("hist_*.jpg"))
            assert len(files) >= 1

    def test_cli_render_entropy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "entropy.jpg")
            result = subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,entropy:1,k:8,iter:50",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            assert result.returncode == 0
            files = list(Path(tmpdir).glob("entropy_*.jpg"))
            assert len(files) >= 1

    def test_cli_creates_spec_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "test.jpg")
            subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,iter:50",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            # Check for companion .spec file
            spec_files = list(Path(tmpdir).glob("test_*.spec"))
            assert len(spec_files) >= 1

            # Check spec file contains the spec string
            spec_content = spec_files[0].read_text()
            assert "logistic" in spec_content


class TestCLIOptions:
    """Test CLI option handling."""

    def test_cli_gamma_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "gamma.jpg")
            result = subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,iter:50,gamma:2.0",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            assert result.returncode == 0

    def test_cli_clip_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "clip.jpg")
            result = subprocess.run(
                [
                    "uv", "run", "python", "lyapunov_cli.py",
                    "map:logistic:AB:2:4:2:4,iter:50,clip:3.0",
                    "--pix", "20",
                    "--out", outfile,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=120,
            )

            assert result.returncode == 0
