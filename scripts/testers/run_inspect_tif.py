

from asyncio.log import logger
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
    '''
    1 indexd: src.read(1), src.tags(1)
    0-based: src.descriptions[0], src.dtypes[0]
    Anything that's an array or list →
    '''
    print('++++++++++SINGLE TILE CHECK+++++++++++++X')
    tile_path = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR/data/4final/predict_input/2026-01-09-00:00_2026-01-09-23:59_Sentinel-1_IW_VV+VH_VH_(Raw).tiff')

    with rasterio.open(tile_path) as src:
        data = src.read()
        if np.isnan(data).any():
            logger.warning("Warning: NaN values found in the data.")

        resolution = src.res  # Or alternatively src.transform.a, src.transform.e
        print(f"---Band count:    {src.count}")
        print('---Tags:',src.tags())  # Look for orbit/pass direction metadata
        print('---Meta:', src.meta)
        print("---Descriptions:", src.descriptions)
        for i in range(src.count):
            print(f'--- Band {i} ---')
            band_data = src.read(i+1)
            valid_data = band_data[np.isfinite(band_data)]
            min, max = min_max_vals(valid_data)

            # Handle missing/empty descriptions
            name = src.descriptions[i].lower() if src.descriptions[i] else f"Band_{i}"

            numvals =  num_band_vals(valid_data)
            print(f"-Band name: {name}: Min={min}, Max={max}")
            print(f"---num unique vals = {numvals}")
            print(f"---CRS: {src.crs}")
            print(f"---Width×Height:  {src.width} × {src.height}")
            print(f"---Transform:     {src.transform}")
            px, py = src.res
            print(f"---Pixel size:    {px} × {py}")
            print(f"---Data type:     {src.dtypes[0]}")
            print(f'---resolution= {src.res}')
            print('----------------------------------')


                
if __name__ == "__main__":
    main()

