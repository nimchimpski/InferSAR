from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

_process_helpers = pytest.importorskip(
    "scripts.process.process_helpers",
    reason="preproc tests require project import path and preprocess helpers",
)

detect_input_is_linear = _process_helpers.detect_input_is_linear
detect_input_is_linear_multiband = _process_helpers.detect_input_is_linear_multiband


pytestmark = [pytest.mark.preproc, pytest.mark.data]


LINEAR_FIXTURE = Path(
    "data/1raw/pc_bangladesh/S1A_IW_GRDH_1SDV_20251231T121235_20251231T121300_062559_07D726_rtc_vv_vh_stacked.tif"
)
DB_FIXTURE = Path("data/4final/sen1floods11/S1Hand/Bolivia_60373_S1Hand.tif")


def _require_multiband(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    with rasterio.open(path) as src:
        if src.count < 2:
            pytest.skip(f"Fixture must be 2+ bands: {path}")
    return path


def _write_single_band(src_path: Path, dst_path: Path, band_index: int) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        band = src.read(band_index)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(band, 1)


def _assert_stats_close(stats_a: dict, stats_b: dict, tol: float = 1e-5) -> None:
    for key in ("min", "max", "p1", "p50", "p99", "frac_lt_zero"):
        va = float(stats_a.get(key, np.nan))
        vb = float(stats_b.get(key, np.nan))
        assert np.isfinite(va), f"Non-finite stat for {key} in first result: {va}"
        assert np.isfinite(vb), f"Non-finite stat for {key} in second result: {vb}"
        assert abs(va - vb) <= tol, f"Stat mismatch for {key}: {va} vs {vb}"


def _run_parity_check(scene_path: Path, sample_size: int = 50000, seed: int = 42) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        vv_path = tmp_dir / "vv_single.tif"
        vh_path = tmp_dir / "vh_single.tif"

        _write_single_band(scene_path, vv_path, band_index=1)
        _write_single_band(scene_path, vh_path, band_index=2)

        pair_is_linear, pair_stats = detect_input_is_linear(
            vv_path,
            vh_path,
            sample_size=sample_size,
            seed=seed,
            return_stats=True,
        )
        multi_is_linear, multi_stats = detect_input_is_linear_multiband(
            scene_path,
            band_indices=(1, 2),
            sample_size=sample_size,
            seed=seed,
            return_stats=True,
        )

    assert pair_is_linear == multi_is_linear
    _assert_stats_close(pair_stats, multi_stats)


def test_scale_detection_wrapper_parity_linear_fixture() -> None:
    scene = _require_multiband(LINEAR_FIXTURE)
    _run_parity_check(scene)


def test_scale_detection_wrapper_parity_db_fixture() -> None:
    scene = _require_multiband(DB_FIXTURE)
    _run_parity_check(scene)


def test_expected_linear_fixture_classification() -> None:
    scene = _require_multiband(LINEAR_FIXTURE)
    is_linear = detect_input_is_linear_multiband(scene)
    assert is_linear is True


def test_expected_db_fixture_classification() -> None:
    scene = _require_multiband(DB_FIXTURE)
    is_linear = detect_input_is_linear_multiband(scene)
    assert is_linear is False