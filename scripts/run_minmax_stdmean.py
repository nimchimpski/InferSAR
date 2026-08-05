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

# mode = 'inference' 
mode = 'train' 


def main():
    """
    Compute the min and max values for each band in a dataset.
    To get the minmax stdmean for the training dataset choose 'train'mode.
    To get the value of the inference dataset choose 'inference' mode.
    """
    # Define the dataset path
    if mode == 'train':
        dataset_path = project_path / 'data' / '4final' / 'dataset' / 'S1Hand'  #
    elif mode == 'inference':
        dataset_path = project_path / 'data' / '4final' / 'predict_input'  #
        
    print(f"mode = {mode} \ndataset = {dataset_path.name}")
    # Compute min and max values


    global_min = float('inf')
    global_max = float('-inf')
    all_vv_vals = []
    all_vh_vals = []
    
    # Iterate through all image files in each event
    ok=0
    tiles=0
    outlier_tiles = []
    for image in dataset_path.iterdir():
        if image.is_file() and image.suffix.lower() in ['.tif', '.tiff'] and image.suffix.lower() not in ['.aux.xml']:
            if mode == 'inference':
                print(f'Processing tile: {image.name}')
            tiles+=1
            # logger.info(f"Processing {image.name}")
            try:
                with rasterio.open(image) as src:
                    # Inspect basic raster metadata for sanity
                    # print(f"  dtype={src.dtypes}, count={src.count}, crs={src.crs}")
                    for band_to_read in range(1, src.count + 1):
                        # Read the data as a NumPy array
                        # print(f"Processing band {band_to_read}" )
                        desc = src.descriptions[band_to_read - 1]
                        if desc:
                            desc = desc.lower()
                        else:
                            name= image.name.lower()
                            if 'vv' in name:
                                desc = 'vv'
                            elif 'vh' in name:
                                desc = 'vh'
                            elif src.count ==1:
                                desc = 'vv' if band_to_read ==1 else'vh'
                            else:
                                desc = f'band_{band_to_read}'
                            # print(f'band to read= {band_to_read} desc= {desc}')
                        
                        data = src.read(band_to_read)  # Read the band
                        # Show scale/offset/statistics tags if present
                        band_tags = src.tags(band_to_read)
                        brief_tags = {k: band_tags[k] for k in band_tags.keys() if k.upper() in {"SCALE", "OFFSET", "UNITTYPE", "STATISTICS_MINIMUM", "STATISTICS_MAXIMUM", "STATISTICS_MEAN", "STATISTICS_STDDEV"}}
                        # if brief_tags:
                        #     print(f"    band {band_to_read} ({desc}) tags: {brief_tags}")
                        raw_min, raw_max = data.min(), data.max()
                        # CREATE A VALID MASK
                        valid = src.dataset_mask().astype(bool)
                        # MASK INVALID DATA WITH NAN
                        data = np.where(valid, data, np.nan)  
                        # FILTER OUT NaN VALUES
                        valid_data = data[np.isfinite(data)]
                        # Report distribution percentiles in linear and dB spaces
                        if valid_data.size > 0:
                            lin_p = np.nanpercentile(valid_data, [0.1, 1, 5, 50, 95, 99, 99.9])
                            db_vals = 10.0 * np.log10(np.clip(valid_data, 1e-6, None))
                            db_p = np.nanpercentile(db_vals, [0.1, 1, 5, 50, 95, 99, 99.9])
                            # print(
                            #     "    percentiles lin(dB): "
                            #     f"lin={np.round(lin_p, 6).tolist()} | dB={np.round(db_p, 3).tolist()}"
                            # )
                        if mode == 'train':
                            # CLAMP TO REALISTIC SAR dB RANGE
                            valid_data = valid_data[(valid_data >= -60) & (valid_data <= 10)]

                            if valid_data.size == 0:
                                outlier_tiles.append((image.name, desc, raw_min, raw_max))
                                logger.info(f"All values outside [-60, 10] in {image.name} band {desc}: raw min={raw_min:.2f}, max={raw_max:.2f}")
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
    
    if outlier_tiles:
        print(f"\n⚠️  OUTLIER TILES (values outside [-60, 10] dB):")
        for fname, band, raw_min, raw_max in outlier_tiles:
            print(f"  {fname} {band}: min={raw_min:.2f}, max={raw_max:.2f}")
        print(f"Total outlier bands: {len(outlier_tiles)}")

    print(f"\nRaw Global Min: {global_min}, Global Max: {global_max}")
    global_min=global_min - 1
    global_max=global_max + 1

    # Print the results
    if mode == 'train':
        print(f"Global Min-1: {global_min}")
        print(f"Global Max+1: {global_max}")


    global_min = round(np.float32(global_min).item(), 2)
    global_max = round(np.float32(global_max).item(), 2)
    vv_mean = round(np.float32(vv_mean).item(), 2)
    vv_std = round(np.float32(vv_std).item(), 2)
    vh_mean = round(np.float32(vh_mean).item(), 2)
    vh_std = round(np.float32(vh_std).item(), 2)
    if mode == 'train':
        output_path= project_path / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the dictionary to the JSON file
        with open(output_path, 'w') as json_file:
            json.dump({'db_min': global_min, 'db_max' : global_max, 'vv_mean': vv_mean, 'vv_std': vv_std, 'vh_mean': vh_mean, 'vh_std': vh_std}, json_file, indent=4)
    
        print(f"Min, max, std, mean values saved to {output_path}")

    # In inference mode, compare dB-converted values against training JSON stats
    if mode == 'train':
        stats_path = project_path / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"\nTraining stats loaded from {stats_path.name}: {stats}")
            db_min = float(stats.get('db_min', -60))
            db_max = float(stats.get('db_max', 10))
            vv_mean_db = float(stats.get('vv_mean', -12))
            vv_std_db = float(stats.get('vv_std', 4))
            vh_mean_db = float(stats.get('vh_mean', -18))
            vh_std_db = float(stats.get('vh_std', 5))

            # Convert concatenated linear arrays to dB
            vv_db = 10.0 * np.log10(np.clip(vv_all, 1e-6, None)) if vv_all.size else np.array([])
            vh_db = 10.0 * np.log10(np.clip(vh_all, 1e-6, None)) if vh_all.size else np.array([])

            if vv_db.size:
                vv_db_p = np.nanpercentile(vv_db, [0.1, 1, 5, 50, 95, 99, 99.9])
                vv_out_frac = np.mean((vv_db < db_min) | (vv_db > db_max))
                vv_norm = (np.clip(vv_db, db_min, db_max) - vv_mean_db) / max(vv_std_db, 1e-6)
                vv_norm_p = np.nanpercentile(vv_norm, [1, 5, 50, 95, 99])
                print("\nVV compatibility:")
                print(f"  dB percentiles [0.1,1,5,50,95,99,99.9]: {np.round(vv_db_p, 3).tolist()}")
                print(f"  frac outside [{db_min}, {db_max}]: {vv_out_frac:.6f}")
                print(f"  normalized percentiles [1,5,50,95,99]: {np.round(vv_norm_p, 3).tolist()}")

            if vh_db.size:
                vh_db_p = np.nanpercentile(vh_db, [0.1, 1, 5, 50, 95, 99, 99.9])
                vh_out_frac = np.mean((vh_db < db_min) | (vh_db > db_max))
                vh_norm = (np.clip(vh_db, db_min, db_max) - vh_mean_db) / max(vh_std_db, 1e-6)
                vh_norm_p = np.nanpercentile(vh_norm, [1, 5, 50, 95, 99])
                print("\nVH compatibility:")
                print(f"  dB percentiles [0.1,1,5,50,95,99,99.9]: {np.round(vh_db_p, 3).tolist()}")
                print(f"  frac outside [{db_min}, {db_max}]: {vh_out_frac:.6f}")
                print(f"  normalized percentiles [1,5,50,95,99]: {np.round(vh_norm_p, 3).tolist()}")
        except FileNotFoundError:
            print(f"\nWarning: training stats JSON not found at {stats_path}. Skipping compatibility check.")
        except Exception as e:
            print(f"\nWarning: failed to load/parse training stats JSON: {e}")


if __name__ == "__main__":
    main()