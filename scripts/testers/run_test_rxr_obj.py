from pathlib import Path

from scripts.process.process_helpers import dataset_type, print_dataarray_info, open_dataarray
from scripts.process.process_dataarrays import *

repo_root = Path('/Users/alexwebb/Library/Mobile Documents/com~apple~CloudDocs/Documents/coding/floodai/UNOSAT_FloodAI_v2')

# cube = repo_root / 'data/2interim/SAR_to_process_INPUT/ST1_20190906_VNM/ST1_20190906_VNM_extracted/ST1_20190906_VNM.nc'

dataset = repo_root / 'data/2interim/SAR_to_process_INPUT/ST1_20190906_VNM/ST1_20190906_VNM_extracted'

# obj_path = dataset / 'ST1_20190906_VNM.nc'

# print(f'---Processing rxr_obj: {rxr_obj.name}')


cubes = list(dataset.rglob('*.nc'))
for cube in cubes:
    print(f'---Processing cube: {cube.name}')
    da = open_dataarray(cube)
    dataset_type(da)
    # print_dataarray_info(da)

    # #open the cube and get the min mix values for each band
    # hhmin, hhmax = calculate_global_min_max_nc(cube, 'hh')
    # print(f"hh min: {hhmin}, max: {hhmax}")
    # maskmin, maskmax = calculate_global_min_max_nc(cube, 'mask')
    # print(f"mask min: {maskmin}, max: {maskmax}")

    print(f'no mask px ={has_no_mask_pixels(da)}')