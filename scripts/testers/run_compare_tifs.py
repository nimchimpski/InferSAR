
from pathlib import Path
import logging

import numpy as np
import rasterio


logging.basicConfig(
    level=logging.INFO,
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_pc_image() -> Path:
    pc_dir = Path("data/1raw/pc_bangladesh")
    candidates = [
        pc_dir / "S1A_IW_GRDH_1SDV_20251231T121235_20251231T121300_062559_07D726_rtc_vv_vh_stacked.tif",
        pc_dir / "S1A_IW_GRDH_1SDV_20251231T121235_20251231T121300_062559_07D726_vv_vh_linear_stacked.tif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(
        path for path in pc_dir.glob("*.tif")
        if path.stem.endswith("_stacked")
    )
    if matches:
        return matches[-1]

    raise FileNotFoundError("No Planetary Computer stacked TIFF found in data/1raw/pc_bangladesh")


def band_summary(src, band_index: int) -> dict:
    band = src.read(band_index).astype(np.float32)
    nodata = src.nodata

    valid_mask = np.isfinite(band)
    if nodata is not None:
        valid_mask &= band != nodata

    valid = band[valid_mask]
    zero_fraction = float(np.mean(band == 0))
    nan_fraction = float(np.mean(~np.isfinite(band)))

    if valid.size == 0:
        return {
            "name": src.descriptions[band_index - 1] or f"Band_{band_index}",
            "dtype": src.dtypes[band_index - 1],
            "nodata": nodata,
            "valid_pixels": 0,
            "zero_fraction": zero_fraction,
            "nan_fraction": nan_fraction,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
        }

    p01, p50, p99 = np.percentile(valid, [1, 50, 99])
    return {
        "name": src.descriptions[band_index - 1] or f"Band_{band_index}",
        "dtype": src.dtypes[band_index - 1],
        "nodata": nodata,
        "valid_pixels": int(valid.size),
        "zero_fraction": zero_fraction,
        "nan_fraction": nan_fraction,
        "min": float(valid.min()),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "max": float(valid.max()),
    }


def raster_summary(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "path": path,
            "count": src.count,
            "dtype": src.dtypes,
            "crs": str(src.crs) if src.crs else None,
            "width": src.width,
            "height": src.height,
            "transform": tuple(src.transform) if src.transform else None,
            "resolution": src.res,
            "descriptions": src.descriptions,
            "tags": src.tags(),
            "nodata": src.nodata,
            "bands": [band_summary(src, index) for index in range(1, src.count + 1)],
        }


def print_raster_summary(label: str, summary: dict) -> None:
    print(f"\n{'=' * 24} {label} {'=' * 24}")
    print(f"Path:         {summary['path']}")
    print(f"Band count:   {summary['count']}")
    print(f"Descriptions: {summary['descriptions']}")
    print(f"CRS:          {summary['crs']}")
    print(f"Size:         {summary['width']} x {summary['height']}")
    print(f"Resolution:   {summary['resolution']}")
    print(f"Transform:    {summary['transform']}")
    print(f"Nodata:       {summary['nodata']}")
    print(f"Tags:         {summary['tags']}")

    for band_index, band in enumerate(summary["bands"], start=1):
        print(f"\nBand {band_index}: {band['name']}")
        print(f"  dtype:        {band['dtype']}")
        print(f"  nodata:       {band['nodata']}")
        print(f"  valid pixels: {band['valid_pixels']}")
        print(f"  zero frac:    {band['zero_fraction']:.6f}")
        print(f"  nan frac:     {band['nan_fraction']:.6f}")
        print(f"  min/p01/p50:  {band['min']} / {band['p01']} / {band['p50']}")
        print(f"  p99/max:      {band['p99']} / {band['max']}")


def print_comparison(left: dict, right: dict) -> None:
    print(f"\n{'=' * 22} Compatibility Summary {'=' * 22}")
    checks = [
        ("band_count", left["count"], right["count"]),
        ("crs", left["crs"], right["crs"]),
        ("resolution", left["resolution"], right["resolution"]),
        ("descriptions", left["descriptions"], right["descriptions"]),
        ("nodata", left["nodata"], right["nodata"]),
    ]
    for name, left_value, right_value in checks:
        status = "MATCH" if left_value == right_value else "DIFF"
        print(f"{name:12} {status:>5} | left={left_value} | right={right_value}")

    for band_index, (left_band, right_band) in enumerate(zip(left["bands"], right["bands"]), start=1):
        print(f"\nBand {band_index} comparison")
        print(f"  names:          {left_band['name']} | {right_band['name']}")
        print(f"  dtype:          {left_band['dtype']} | {right_band['dtype']}")
        print(f"  nodata:         {left_band['nodata']} | {right_band['nodata']}")
        print(f"  zero frac:      {left_band['zero_fraction']:.6f} | {right_band['zero_fraction']:.6f}")
        print(f"  nan frac:       {left_band['nan_fraction']:.6f} | {right_band['nan_fraction']:.6f}")
        print(f"  min/p50/max:    {left_band['min']} / {left_band['p50']} / {left_band['max']}")
        print(f"                   {right_band['min']} / {right_band['p50']} / {right_band['max']}")


def main():
    print("++++++++++TIFF COMPARISON+++++++++++++")
    pc_path = find_pc_image()
    train_path = Path("data/4final/sen1floods11/S1Hand/Bolivia_60373_S1Hand.tif")

    left = raster_summary(pc_path)
    right = raster_summary(train_path)

    print_raster_summary("Planetary Computer", left)
    print_raster_summary("Sen1Floods11", right)
    print_comparison(left, right)


if __name__ == "__main__":
    main()

