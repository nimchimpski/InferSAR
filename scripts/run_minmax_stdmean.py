import json
import logging
import sys
import rasterio
import numpy as np
from pathlib import Path
'''
writes the vals to .json.
skips any tiles that are all nans.
Consider adding headroom of 1-2 units to max to avoid clipping to the .json file
'''
# Add project directory to Python path for imports
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from scripts import train
from scripts.process.process_helpers import  write_minmax_to_json

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

mode = 'train' 
# mode = 'train' 


def main():
    """
    Compute the min and max values for each band in a dataset.
    """
    # Define the dataset path
    if mode == 'train':
        dataset_path = project_path / 'data' / '4final' / 'dataset' / 'S1Hand'  #
    elif mode == 'inference':
        dataset_path = project_path / 'data' / '4final' / 'predict_input'  #
        
    print(f"mode ={mode} dataset= {dataset_path}")
    # Compute min and max values


    global_min = float('inf')
    global_max = float('-inf')
    all_vv_vals = []
    all_vh_vals = []
    
    # Iterate through all image files in each event
    ok=0
    tiles=0
    for image in dataset_path.iterdir():
        if image.is_file() and image.suffix.lower() in ['.tif', '.tiff'] and image.suffix.lower() not in ['.aux.xml']:
            if mode == 'inference':
                print(f'Processing tile: {image.name}')
            tiles+=1
            # logger.info(f"Processing {image.name}")
            try:
                with rasterio.open(image) as src:
                    for band_to_read in range(1, src.count + 1):
                        # Read the data as a NumPy array
                        # print(f"Processing band {band_to_read}" )
                        if mode == 'train':
                            if tiles < 2:
                                desc = src.descriptions[band_to_read - 1].lower()
                                print(f'band to read= {band_to_read} desc= {desc}')
                        data = src.read(band_to_read)  # Read the first band
                        valid_data = data[np.isfinite(data)]  # create new np array excluding NaN values
                        if len(valid_data) == 0:
                            logger.info(f"All values are NaN in {image.name}, skipping...")
                            continue
                        if desc == 'vv':
                            all_vv_vals.append(valid_data)
                        elif desc == 'vh':
                            all_vh_vals.append(valid_data)
                        lmin, lmax = valid_data.min(), valid_data.max()
                        logger.debug(f"local: Min: {int(lmin)}, Max: {int(lmax)}")
                        global_min = min(global_min, lmin)
                        global_max = max(global_max, lmax)
                # logger.info(f'global_min={global_min}, global_max={global_max}')
                ok+=1
            except Exception as e:
                logger.info(f"Error processing {image}: {e}")
                continue
    vv_all = np.concatenate(all_vv_vals)
    vh_all = np.concatenate(all_vh_vals)
    vv_mean = vv_all.mean()
    vv_std  = vv_all.std()
    vh_mean = vh_all.mean()
    vh_std  = vh_all.std()
    print(f"VV Mean: {vv_mean}, VV Std: {vv_std}")
    print(f"VH Mean: {vh_mean}, VH Std: {vh_std}")
    # logger.info(f"Global Min: {global_min}, Global Max: {global_max}")
    print(f"num tiles processed= {ok} out of {tiles}")




    print(f"Raw Global Min: {global_min}, Global Max: {global_max}")
    global_min=global_min - 1
    global_max=global_max + 1

    # Print the results
    print(f"Global Min-1: {global_min}")
    print(f"Global Max+1: {global_max}")

    if mode == 'train':
        output_path= project_path / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the dictionary to the JSON file
        with open(output_path, 'w') as json_file:
            json.dump({'db_min': global_min, 'db_max' : global_max, 'vv_mean': vv_mean, 'vv_std': vv_std, 'vh_mean': vh_mean, 'vh_std': vh_std}, json_file, indent=4)
    
    print(f"Min, max, std, mean values saved to {output_path}")


if __name__ == "__main__":
    main()