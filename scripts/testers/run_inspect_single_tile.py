

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process.process_helpers import get_band_name, min_max_vals, num_band_vals, datatype_check, check_single_tile, print_tiff_info_TSX

def main():
    print('++++++++++SINGLE TILE CHECK+++++++++++++X')
    tile_path = Path("data/4final/predictions/tiles/tile_0006_512_512.tif")
    # check_single_tile(tile_path)
    print_tiff_info_TSX(tile_path,)

                
if __name__ == "__main__":
    main()

