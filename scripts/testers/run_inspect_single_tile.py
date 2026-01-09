

from pathlib import Path
import rasterio
import numpy as np
import logging
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process.process_helpers import get_band_name, min_max_vals, num_band_vals, datatype_check, check_single_tile, print_tiff_info_TSX, nan_check

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    print('++++++++++SINGLE TILE CHECK+++++++++++++X')
    tile_path = Path('data/tiling_tests/listening.tiff')

    with rasterio.open(tile_path) as src:
        data = src.read()
        nan_check(data)
        resolution = src.res  # Or alternatively src.transform.a, src.transform.e
        print(f"Band count:    {src.count}")
        print('----------------------------------')
        for i in range(1, src.count + 1):
            band_data = src.read(i)
            valid_data = band_data[np.isfinite(band_data)]
            min, max = min_max_vals(valid_data)
            name = get_band_name(i, src)
            numvals =  num_band_vals(valid_data)
            print(f"Band name: {name}: Min={min}, Max={max}")
            print(f"num unique vals = {numvals}")
            print(f"CRS: {src.crs}")
            print(f"Width×Height:  {src.width} × {src.height}")
            print(f"Transform:     {src.transform}")
            px, py = src.res
            print(f"Pixel size:    {px} × {py}")
            print(f"Data type:     {src.dtypes[0]}")
            print(f'resolution= {src.res}')
            print('----------------------------------')

                
if __name__ == "__main__":
    main()

