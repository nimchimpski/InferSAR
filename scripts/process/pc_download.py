#!/usr/bin/env python
"""
Download Sentinel-1 RTC from Planetary Computer for Bangladesh
and stack VV/VH into a single inference input raster.
"""

from pystac_client import Client
import planetary_computer
import requests
from osgeo import gdal
import numpy as np
from pathlib import Path
import os

# ========== CONFIGURATION ==========
CENTER_LAT = 22.096
CENTER_LON = 89.2241
BOUNDS = [CENTER_LON - 0.25, CENTER_LAT - 0.25,  # bbox [W, S, E, N]
          CENTER_LON + 0.25, CENTER_LAT + 0.25]
DATE_START = "2025-01-01"
DATE_END = "2025-12-31"
MAX_ITEMS = 5
OUTPUT_DIR = Path("data/1raw/pc_bangladesh")
KEEP_INTERMEDIATE_BANDS = True
# ===================================

def setup_environment():
    """Ensure Planetary Computer is authenticated."""
    token = os.getenv("PC_AUTH_TOKEN")
    if not token:
        print("Warning: PC_AUTH_TOKEN not set. Using anonymous access.")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def search_scenes(client):
    """Search for Sentinel-1 RTC scenes over Bangladesh."""
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
    # convert to numpy arrays
    vv_data = vv_ds.ReadAsArray()
    vh_data = vh_ds.ReadAsArray()
    
    # Stack to [height, width, 2] or [2, height, width]
    stacked = np.stack([vv_data, vh_data], axis=0)  # [bands, h, w]
    vv_nodata = vv_ds.GetRasterBand(1).GetNoDataValue()
    vh_nodata = vh_ds.GetRasterBand(1).GetNoDataValue()
    vv_desc = "VV"
    vh_desc = "VH"
    
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
    # Set band descriptions and source metadata.
    out_ds.GetRasterBand(1).SetDescription(vv_desc)
    out_ds.GetRasterBand(2).SetDescription(vh_desc)
    out_ds.SetMetadataItem("source_collection", "sentinel-1-rtc")
    out_ds.SetMetadataItem("source_vv", Path(vv_path).name)
    out_ds.SetMetadataItem("source_vh", Path(vh_path).name)
    out_ds.FlushCache()
    print(f"Stacked VV+VH: {output_path}")

def main():
    fetch_new = False
    print("=" * 40)
    print("Bangladesh Sentinel-1 Inference Pipeline")
    if fetch_new:
        print("Fetching new data from Planetary Computer...")
    else:
        print("Using existing data in data/1raw/pc_bangladesh...")
    print("=" * 40)

    if fetch_new:
        setup_environment()
        client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        # Search
        items = search_scenes(client)
        if not items:
            print("No scenes found. Try expanding date range or bbox.")
            return

        # Download first scene's VV and VH
        item = items[0]
        print(f"\nDownloading scene: {item.id}")

        vv_path = download_band(item, "vv")
        vh_path = download_band(item, "vh")

        if not vv_path or not vh_path:
            print("Failed to download bands")
            return

        vv_path = OUTPUT_DIR / f"{item.id}_vv.tif"
        vh_path = OUTPUT_DIR / f"{item.id}_vh.tif"
    else:
        # Use existing files
        item_id = "S1A_IW_GRDH_1SDV_20251231T121235_20251231T121300_062559_07D726"
        
        vv_path = OUTPUT_DIR / f"{item_id}_rtc_vv.tif"
        vh_path = OUTPUT_DIR / f"{item_id}_rtc_vh.tif"
        if not vv_path.exists() or not vh_path.exists():
            print("Existing band files not found. Set fetch_new=True to download.")
            return

    # Build scene id from the VV filename so this works for both fetched and offline modes.
    # scene_id = vv_path.stem.removesuffix("_vv")

    # Stack bands
    # final_output = OUTPUT_DIR / f"{scene_id}_vv_vh_stacked.tif"
    # stack_vv_vh(vv_path, vh_path, str(final_output))
    
    # Cleanup (optional; keep by default for offline reuse on slow connections)
    if not KEEP_INTERMEDIATE_BANDS:
        for f in [vv_path, vh_path]:
            try:
                os.remove(f)
            except OSError:
                pass
    
    print("\n" + "=" * 40)
    # print(f"Inference-ready stacked image: {final_output}")
    # print("Format: 2 bands (VV, VH), Float32, RTC georeferenced source")
    print("Format: 2 single files, vv and vh Float32, RTC georeferenced source")
    print("=" * 40)

if __name__ == "__main__":
    main()