#!/usr/bin/env python
"""
Download Sentinel-1 RTC from Planetary Computer for the 2022 Pakistan floods
(Sindh province, Dadu district) and stack VV/VH into a single inference input raster.
"""

from pystac_client import Client
import planetary_computer
import requests
from osgeo import gdal
import numpy as np
from pathlib import Path
import os

# ========== CONFIGURATION ==========
CENTER_LAT = 26.7
CENTER_LON = 67.8
BOUNDS = [CENTER_LON - 0.25, CENTER_LAT - 0.25,  # bbox [W, S, E, N]
          CENTER_LON + 0.25, CENTER_LAT + 0.25]
DATE_START = "2022-08-20"
DATE_END = "2022-09-15"
MAX_ITEMS = 5
OUTPUT_DIR = Path("data/1raw/pc_pakistan_floods_2022")
KEEP_INTERMEDIATE_BANDS = True
# ===================================

def setup_environment():
    """Ensure Planetary Computer is authenticated."""
    token = os.getenv("PC_AUTH_TOKEN")
    if not token:
        print("Warning: PC_AUTH_TOKEN not set. Using anonymous access.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def search_scenes(client):
    """Search for Sentinel-1 RTC scenes over the Dadu/Sindh flood area."""
    search = client.search(
        collections=["sentinel-1-rtc"],
        bbox=BOUNDS,
        datetime=f"{DATE_START}/{DATE_END}",
        max_items=MAX_ITEMS
    )
    items = list(search.items())
    print(f"Found {len(items)} scenes")
    return items

def download_band(item, band_name):
    """Download single band (VV or VH) as GeoTIFF."""
    if band_name not in item.assets:
        return None

    filename = OUTPUT_DIR / f"{item.id}_{band_name}.tif"
    if filename.exists():
        print(f"Reusing existing file: {filename}")
        return str(filename)

    asset = item.assets[band_name]
    signed_url = planetary_computer.sign(asset.href)
    response = requests.get(signed_url, timeout=120)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

    return str(filename)

def stack_vv_vh(vv_path, vh_path, output_path):
    """Stack VV and VH bands into 2-channel GeoTIFF."""
    vv_ds = gdal.Open(str(vv_path))
    vh_ds = gdal.Open(str(vh_path))
    vv_data = vv_ds.ReadAsArray()
    vh_data = vh_ds.ReadAsArray()

    stacked = np.stack([vv_data, vh_data], axis=0)  # [bands, h, w]
    vv_nodata = vv_ds.GetRasterBand(1).GetNoDataValue()
    vh_nodata = vh_ds.GetRasterBand(1).GetNoDataValue()

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(output_path),
                           vv_ds.RasterXSize, vv_ds.RasterYSize,
                           2, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(stacked[0])
    out_ds.GetRasterBand(2).WriteArray(stacked[1])
    if vv_nodata is not None:
        out_ds.GetRasterBand(1).SetNoDataValue(vv_nodata)
    if vh_nodata is not None:
        out_ds.GetRasterBand(2).SetNoDataValue(vh_nodata)
    out_ds.SetGeoTransform(vv_ds.GetGeoTransform())
    out_ds.SetProjection(vv_ds.GetProjection())
    out_ds.GetRasterBand(1).SetDescription("VV")
    out_ds.GetRasterBand(2).SetDescription("VH")
    out_ds.SetMetadataItem("source_collection", "sentinel-1-rtc")
    out_ds.SetMetadataItem("source_vv", Path(vv_path).name)
    out_ds.SetMetadataItem("source_vh", Path(vh_path).name)
    out_ds.FlushCache()
    print(f"Stacked VV+VH: {output_path}")

def main():
    print("=" * 40)
    print("Pakistan 2022 Floods (Sindh/Dadu) Sentinel-1 Inference Pipeline")
    print("Fetching data from Planetary Computer...")
    print("=" * 40)

    setup_environment()
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    items = search_scenes(client)
    if not items:
        print("No scenes found. Try expanding date range or bbox.")
        return

    item = items[0]
    print(f"\nDownloading scene: {item.id}")

    vv_path = download_band(item, "vv")
    vh_path = download_band(item, "vh")

    if not vv_path or not vh_path:
        print("Failed to download bands")
        return

    final_output = OUTPUT_DIR / f"{item.id}_vv_vh_stacked.tif"
    stack_vv_vh(vv_path, vh_path, str(final_output))

    if not KEEP_INTERMEDIATE_BANDS:
        for f in [vv_path, vh_path]:
            try:
                os.remove(f)
            except OSError:
                pass

    print("\n" + "=" * 40)
    print(f"Inference-ready stacked image: {final_output}")
    print("Format: 2 bands (VV, VH), Float32, RTC georeferenced source")
    print("=" * 40)

if __name__ == "__main__":
    main()
