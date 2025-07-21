#!/usr/bin/env python3
import sys
import rasterio
import numpy as np
from scripts.process.process_helpers import print_tiff_info_TSX

def describe_raster(path):
    with rasterio.open(path) as src:
        print(f"\n=== {path} ===")
        # Basic info
        print(f"CRS:           {src.crs}")
        print(f"Width×Height:  {src.width} × {src.height}")
        print(f"Transform:     {src.transform}")
        px, py = src.res
        print(f"Pixel size:    {px} × {py}")
        print(f"Data type:     {src.dtypes[0]}")
        print(f"Band count:    {src.count}")
        print(f'shape: {src.shape}')
        logger.info(f'---resolution= {src.res}')


        for i in range(1, src.count + 1):
            print(f"Band {i} name: {src.descriptions[i - 1]}")
            print(f"Band {i} nodata: {src.nodata}")
            # Read entire band into memory (only do this if file fits in RAM)
            arr = src.read(i).astype(np.float64)  # float for full stats
            valid = arr[~src.dataset_mask().astype(bool)]  # mask out nodata if any

            # Compute stats
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            mean, std = np.nanmean(arr), np.nanstd(arr)
            print(f"Min / Max:     {mn:.3f} / {mx:.3f}")
            print(f"Mean / StdDev: {mean:.3f} / {std:.3f}\n")
    

if __name__ == "__main__":
    print("SAR Image Info Tool")
    if len(sys.argv) < 2:
        print("Usage: describe.py <raster1.tif> [raster2.tif ...]")
        sys.exit(1)
    for pth in sys.argv[1:]:
        # describe_raster(pth)
        print_tiff_info_TSX(pth)
