from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio

from scripts.process.process_helpers import (
    detect_input_is_linear,
    detect_input_is_linear_multiband,
)


def _find_candidate_multiband() -> Path:
    candidates = [
        Path("data/4final/sen1floods11/S1Hand/Bolivia_60373_S1Hand.tif"),
        Path("data/1raw/pc_bangladesh/S1A_IW_GRDH_1SDV_20251231T121235_20251231T121300_062559_07D726_rtc_vv_vh_stacked.tif"),
    ]

    for path in candidates:
        if path.exists():
            return path

    found = sorted(Path("data").glob("**/*.tif"))
    for path in found:
        try:
            with rasterio.open(path) as src:
                if src.count >= 2:
                    return path
        except Exception:
            continue

    raise FileNotFoundError("No readable 2-band TIFF found under data/ for scale check test.")


def _write_single_band(src_path: Path, dst_path: Path, band_index: int) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        band = src.read(band_index)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(band, 1)


def _assert_stats_compatible(stats_a: dict, stats_b: dict, tol: float = 1e-5) -> None:
    for key in ("min", "max", "p1", "p50", "p99", "frac_lt_zero"):
        va = float(stats_a.get(key, np.nan))
        vb = float(stats_b.get(key, np.nan))
        if not np.isfinite(va) or not np.isfinite(vb):
            raise AssertionError(f"Non-finite stat for key={key}: {va}, {vb}")
        if abs(va - vb) > tol:
            raise AssertionError(
                f"Stat mismatch for {key}: single-band={va}, multiband={vb}, diff={abs(va - vb)}"
            )


def main() -> None:
    sample_size = 50000
    seed = 42

    scene_path = _find_candidate_multiband()
    print(f"Using scene: {scene_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        vv_path = tmp_dir / "vv_single.tif"
        vh_path = tmp_dir / "vh_single.tif"

        _write_single_band(scene_path, vv_path, band_index=1)
        _write_single_band(scene_path, vh_path, band_index=2)

        is_linear_pair, stats_pair = detect_input_is_linear(
            vv_path,
            vh_path,
            sample_size=sample_size,
            seed=seed,
            return_stats=True,
        )
        is_linear_multi, stats_multi = detect_input_is_linear_multiband(
            scene_path,
            band_indices=(1, 2),
            sample_size=sample_size,
            seed=seed,
            return_stats=True,
        )

    if is_linear_pair != is_linear_multi:
        raise AssertionError(
            f"Decision mismatch: pair={is_linear_pair}, multiband={is_linear_multi}"
        )

    _assert_stats_compatible(stats_pair, stats_multi)

    print("Scale detection consistency: PASS")
    print(f"Decision: {'linear' if is_linear_pair else 'dB'}")
    print(
        "Stats: "
        f"min={stats_pair['min']:.6f}, max={stats_pair['max']:.6f}, "
        f"p50={stats_pair['p50']:.6f}, p99={stats_pair['p99']:.6f}, "
        f"frac_lt_zero={stats_pair['frac_lt_zero']:.6f}"
    )


if __name__ == "__main__":
    main()