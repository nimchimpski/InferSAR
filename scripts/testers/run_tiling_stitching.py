import rasterio
import os
import shutil
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import suppress

from rasterio.windows import Window
from pathlib import Path
from scripts.process.process_tiffs import tile_geotiff_directly
from scripts.inference_functions import stitch_tiles, stitch_tiles_raw

start_time = time.time()   
# vv_image = Path('data/4final/predict_input/rhone_leman_Sentinel-1_IW_VV_(Raw).tiff')
# vh_image = Path('data/4final/predict_input/rhone_leman_Sentinel-1_IW_VH_(Raw).tiff')
tiling_tests_path = Path('data/tiling_tests/')
tiles_path = tiling_tests_path / 'tiles'
if tiles_path.exists():
    if tiles_path.is_dir():
        shutil.rmtree(tiles_path)
    else:
        tiles_path.unlink()
tiles_path.mkdir(parents=True, exist_ok=True)
metadata = []
tile_size = 512
stride = 256
threshold = 0.5
image_path = Path('data/tiling_tests/listening.tiff')

def main():
    print('//////////////////////////////')

    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height

        print(f'Image dimensions: width={width}, height={height}')

        tile_idx = 0
        for y in range(0, height, stride):  # Changed: iterate through full height
            # print(f'ROW y={y}')
            for x in range(0, width, stride):  # Changed: iterate through full width
                # Calculate this_tile tile bounds (handles edge tiles correctly)
                # print(f'  COL x={x}')
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                this_tile_width = x_end - x
                this_tile_height = y_end - y

                # Create window with this_tile dimensions
                window = Window(x, y, this_tile_width, this_tile_height)
                
                # print(f'tile width={window.width}, tile height={window.height}')

                # Create tile filename (using y, x for consistency with xarray version)
                tile_name = f"tile_{tile_idx:04d}_{y}_{x}.tif"
                tile_path = tiles_path / tile_name

                #  create the data to write
                profile = src.profile.copy()
                profile.update({
                    "height": this_tile_height,
                    "width": this_tile_width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                # Write tile
                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(src.read(window=window))

                # tiles.append(tile_path)

                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                metadata.append({
                    'tile_name': tile_name,
                    'x_start': x,
                    'y_start': y,
                    'x_end': x_end,
                    'y_end': y_end,
                    'width': x_end - x,      # Direct calculation
                    'height': y_end - y      # Direct calculation
                })
                tile_idx += 1

    print(f'max x_end={max([m["x_end"] for m in metadata])}, max y_end={max([m["y_end"] for m in metadata])}')
    print(f'Tiles saved to: {tiles_path}')
    # print(f'Metadata: {metadata}')
    #  STITCHING (using raw image stitching for simple .tiff test)
    save_path=tiling_tests_path / 'test_stitched_image.tiff'
    final_img = stitch_tiles_raw(metadata, tiles_path, save_path, image_path, tile_size, stride)
    print(f'final image dims: {final_img.shape}')

    end_time = time.time()
    run_time = end_time - start_time
    print(f"Run time: {run_time:.2f} seconds")

if __name__ == "__main__":
    main()