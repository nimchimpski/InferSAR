

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process.process_helpers import print_tiff_info_TSX
from scripts.process.process_dataarrays import *

repo_root = Path('/Users/alexwebb/Library/Mobile Documents/com~apple~CloudDocs/Documents/coding/floodai/UNOSAT_FloodAI_v2')


dataset = repo_root / 'data/2interim/TSX_TILES/NORM_TILES_FOR_SELECT_AND_SPLIT_INPUT/ST1_20190906_normalized_tiles_logclipmm_g_pcnf100'
# dir = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\COMPLETE_previous_train_inputs\mtnweighted_NO341_3_full\train")
num = 0
for tile in dataset.iterdir():
    print_tiff_info_TSX(tile)
    num += 1
    if num == 2:
        print(f'---has no mask px ={has_no_mask_pixels(tile)}')
        break

