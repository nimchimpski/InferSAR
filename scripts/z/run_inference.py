import torch
from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
import time
import os
import xarray as xr
import json
import matplotlib.pyplot as plt
import click
import yaml
import gc
import logging
import signal

from rasterio.plot import show
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import OrderedDict
from skimage.morphology import binary_erosion
from torch.utils.data import Dataset, DataLoader

from scripts.train.train_classes import UnetModel, Sen1Dataset
from scripts.train.train_helpers import pick_device
from scripts.process.process_tiffs import  create_event_datacube_copernicus,reproject_to_4326_gdal, make_float32_inf, resample_tiff_gdal
from scripts.process.process_dataarrays import tile_datacube_rxr_inf
from scripts.process.process_helpers import  print_tiff_info_TSX, check_single_input_filetype, rasterize_kml_rasterio, compute_image_minmax, process_raster_minmax, path_not_exists, read_minmax_from_json, normalize_imagedata_inf, read_raster, write_raster
from scripts.process.process_helpers import handle_interrupt
from scripts.inference_functions import make_prediction_tiles, stitch_tiles, clean_checkpoint_keys

start=time.time()

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)
logging.getLogger('scripts.process.process_helpers').setLevel(logging.INFO)

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

@click.command()
@click.option('--no_config', is_flag=True, help='loading from no_config settings', show_default=False)

def main(no_config=False):

    device = pick_device()
    logger.info(f">>> Using device: {device}")

    logger.info(f'no_config mode = {no_config}')


    # VARIABLES................................................................
    mode = 'inference' # 'inference' = input_is_linear = True
    norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
    stats = 0
    db_min = -30.0
    db_max = 0.0
    MAKE_TIFS = 0
    MAKE_DATAARRAY= 0
    # stride = tile_size
    ############################################################################
    # DEFINE PATHS
    # DEFINE THE WORKING FOLDER FOR I/O
    predict_input = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/data/4final/predict_input")
    logger.info(f'working folder: {predict_input.name}')
    if path_not_exists(predict_input):
        return
    
    minmax_path = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/global_minmax_INPUT/global_minmax.json")
    if path_not_exists(minmax_path):
        return

    ckpt_path = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/checkpoints/ckpt_INPUT")

    ############################################################################
    if not config:
        threshold =  0.5 # PREDICTION CONFIDENCE THRESHOLD
        tile_size = 256 # TILE SIZE FOR INFERENCE
        # Normalize all paths in the config
        # image = check_single_input_filetype(predict_input, 'image', '.tif', '.tiff')
        # if image is None:
        #     logger.info(f"---No input image found in {predict_input}")
        #     return
        # else:
        #     logger.info(f'found input image: {image.name}')
        output_folder = predict_input
        output_filename = '_name'
        # analysis_extent = Path('Users/alexwebb/floodai/InferSAR/data/4final/predict_INPUT/extent_INPUT')  

    # READ CONFIG
    elif config:
        config_path = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/floodaiv2_config.yaml')
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        threshold = config["threshold"] # PREDICTION CONFIDENCE THRESHOLD
        tile_size = config["tile_size"] # TILE SIZE FOR INFERENCE
        # Normalize all paths in the config
        input_file = Path(config['input_file'])
        output_folder = Path(config['output_folder'])
        output_filename = Path(config['output_filename'])
        # analysis_extent = Path(config['analysis_extent'])

    stride = tile_size # STRIDE FOR TILING, SAME AS TILE SIZE

    # logger.info(f' config = {config}')
    logger.info(f'threshold: {threshold}') 
    logger.info(f'tile_size: {tile_size}')
    logger.info(f'output_folder= {output_folder.name}')
    logger.info(f'output_filename= {output_filename}')
    # logger.info(f'alalysis_extent= {analysis_extent}')
    # logger.info(f' IF TRAINING: CHECK LAYERDICT NAMES=FILENAMES IN FOLDER <<<')
    # FIND THE CKPT
    ckpt = next(ckpt_path.rglob("*.ckpt"), None)
    if ckpt is None:
        logger.info(f"---No checkpoint found in {ckpt_path}")
        return
    logger.info(f'ckpt: {ckpt.name}')

    # poly = check_single_input_filetype(predict_input,  'poly', '.kml')
    # if poly is None:
        # return

    # GET REGION CODE FROM MASK TODO
    # sensor = image.parents[1].name.split('_')[:1]
    sensor = 'sensor'
    # logger.info(f'datatype= ',sensor[0])
    # date = image.parents[1].name.split('_')[0]
    date = 'date'
    # logger.info(f'date= ',date)
    image_code = "code"
    # image_code = "_".join(image.parents[3].name.split('_')[4:])
    # image_code = "_".join(image.parents[1].name.split('_')[1])
    # parts = image.name.split('_')
    # image_code = "_".join(parts[:-1])
    # logger.info(f'image_code= ',image_code)
    stiched_img = output_folder / f'{sensor}_{image_code}_{date}_{tile_size}_{threshold}{output_filename}_WATER_AI.tif'

    logger.info(f'output filename: {stiched_img.name}')
    if stiched_img.exists():
        logger.info(f"---overwriting existing file! : {stiched_img}")

    # CREATE THE EXTRACTED FOLDER
    extracted = predict_input / f'{image_code}_extracted'

    logger.info(f' MAKE_TIFS = {MAKE_TIFS}')

    if MAKE_TIFS:
        if extracted.exists():
            # logger.info(f"--- Deleting existing extracted folder: {extracted}")
            # delete the folder and create a new one
            shutil.rmtree(extracted)
        extracted.mkdir(exist_ok=True)

        ###### CHANGE DATATYPE TO FLOAT32
        # logger.info('CHANGING DATATYPE')
        # image_32 = extracted / f'{image_code}_32.tif'
        # make_float32_inf(image, image_32)
        # logger.info_tiff_info_TSX(image_32, 1)

        ##### RESAMPLE TO 2.5
        # logger.info('RESAMPLING')
        # resamp_image = extracted / f'{image_32.stem}_resamp'
        # resample_tiff_gdal(image_32, resamp_image, target_res=2.5)
        # logger.info_tiff_info_TSX(resamp_image, 2)

        # with rasterio.open(image) as src:
            # logger.info(f'src shape= ',src.shape)

        ##### SORT OUT ANALYSIS EXTENT

        # ex_extent = extracted / f'{image_code}_extent.tif'
        # create_extent_from_mask(image, ex_extent)
        # rasterize_kml_rasterio( poly, ex_extent, pixel_size=0.0001, burn_value=1)

        ####### REPROJECT IMAGE TO 4326
        # logger.info('REPROJECTING')
        # final_image = extracted / 'final_image.tif'
        # reproject_to_4326_gdal(image_32, final_image, resampleAlg = 'bilinear')
        # logger.info_tiff_info_TSX(final_image, 3)

        # reproj_extent = extracted / f'{image_code}_4326_extent.tif'
        # reproject_to_4326_gdal(ex_extent, reproj_extent)
        # fnal_extent = extracted / f'{image_code}_32_final_extent.tif'
        # make_float32_inf(reproj_extent, final_extent

    # final_image = extracted / 'final_image.tif'

    # GET THE TRAINING MIN MAX STATS
    statsdict =  read_minmax_from_json(minmax_path)
    stats = (statsdict["min"], statsdict["max"])
    logger.info(f'---stats: {stats}')

    if MAKE_DATAARRAY:
        create_event_datacube_copernicus(predict_input, image_code)

    cube = next(predict_input.rglob("*.nc"), None)
    if cube is None:
        logger.info(f"---No data cube found in {predict_input.name}")
        return  
    save_tiles_path = predict_input /  f'{image_code}_tiles'

    if save_tiles_path.exists():
        # logger.info(f" Deleting existing tiles folder: {save_tiles_path}")
        # delete the folder and create a new one
        shutil.rmtree(save_tiles_path)
        save_tiles_path.mkdir(exist_ok=True, parents=True)
        # CALCULATE THE STATISTICS

    # DO THE TILING
    tiles, metadata = tile_datacube_rxr_inf(cube, save_tiles_path, tile_size=tile_size, stride=stride, norm_func=norm_func, stats=stats, percent_non_flood=0, inference=True) 
    # logger.info(f"{len(tiles)} tiles saved to {save_tiles_path}")
    # logger.info(f"{len(metadata)} metadata saved to {save_tiles_path}")
    # metadata = Path(save_tiles_path) / 'tile_metadata.json'

    # NOW YOU HAVE FOLDER OF TILES - SAME STATE AS TRAINING

    # INITIALIZE THE MODEL
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=pick_device()
    model = UnetModel( encoder_name="resnet34", in_channels=2, classes=1, pretrained=True.to(device)
    )   
    # LOAD THE CHECKPOINT
    ckpt_path = Path(ckpt)
    checkpoint = torch.load(ckpt_path)

    # EXTRACT THE MODEL STATE DICT
    cleaned_state_dict = clean_checkpoint_keys(checkpoint["state_dict"])

    # LOAD THE MODEL STATE DICT
    model.load_state_dict(cleaned_state_dict)

    # MAKE DATASET  CLASS INSTANCE

    if mode == 'inference':
        bs = 1,
        shuffle = False,
        input_is_linear = True

    infer_ds = Sen1Dataset(
        mode='inference',
        csv_path=save_tiles_path / 'tile_metadata.json',
        root_dir=save_tiles_path,
        input_is_linear=input_is_linear,
        db_min=db_min,
        db_max=db_max)
    
        # MAKE DATALOADER FROM DATASET

    infer_dl = DataLoader( infer_ds,
        batch_size=bs,  # Batch size of 1 for inference 
        shuffle=shuffle,  # No need to shuffle for inference
        num_workers=4,  # Adjust based on your system
        persistent_workers=True,  # Keep workers alive for faster loading   
        )

    # prediction_tiles = make_prediction_tiles(save_tiles_path, metadata, model, device, threshold)

    # ----------MAKE PREDICTIONS IN BATCHES
    model.eval()

    # DEFINE THE PREDICTION TILES FOLDER
    predictions_folder = save_tiles_path.parent / f'{save_tiles_path.stem}_predictions'
    # DELETE THE PREDICTION FOLDER IF IT EXISTS
    if predictions_folder.exists():
        logger.info(f"--- Deleting existing predictions folder: {predictions_folder}")
        shutil.rmtree(predictions_folder)
    predictions_folder.mkdir(exist_ok=True)

    with torch.no_grad():
        for imgs, _, valids, fnames in tqdm(infer_dl, desc="Predict"):
            imgs   = imgs.to(device)            # [B,2,H,W]
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu()  # back to CPU for numpy/rasterio
            preds  = (probs > threshold).float()  # [B,1,H,W]

            for b, name in enumerate(fnames):
                out = preds[b, 0].numpy()                 # 2-D
                out[~valids[b, 0].numpy().astype(bool)] = 0  # mask invalid px

                src_path = save_tiles_path / name
                with rasterio.open(src_path) as src:
                    profile = src.profile
                profile.update(dtype="float32", count=1)

                dst_path = predictions_folder / name
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(out.astype("float32"), 1)


    # STITCH PREDICTION TILES
    input_image = next(predict_input.rglob("*.tif"), None) if name in ['vv', 'vh'] else None
    prediction_img = stitch_tiles(metadata, predictions_folder, stiched_img, input_image)
    # logger.info prediction_img size
    # logger.info(f'prediction_img shape:',prediction_img.shape)
    # display the prediction mask
    # plt.imshow(prediction_img, cmap='gray')
    # plt.show()

    del model

    torch.cuda.empty_cache()
    gc.collect()

    end = time.time()
    # time taken in minutes to 2 decimal places
    logger.info(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == "__main__":
    main()