"""
Flood detection model training, testing, and inference script.

This script uses a centralized ProjectPaths class to manage all directory and file paths,
making the code more maintainable and easier to understand.

Usage:
    python run_master_new.py --train    # Train the model
    python run_master_new.py --test     # Test the model  
    python run_master_new.py --inference # Run inference
    python run_master_new.py --config   # Use config file
    python run_master_new.py --fine_tune # Fine-tune from training checkpoint
    python run_master_new.py --ckpt_input # Specify checkpoint input folder

If training on multiple regions - consider enabling 'dynamic weighting' to deal with batch differences - and then disable the weighting here:-
 smp_bce =  smp.losses.SoftBCEWithLogitsLoss(ignore_index=255, reduction='mean',pos_weight=torch.tensor([8.0]))
"""
import logging
logging.basicConfig(
    level=logging.INFO, 
    force=True,                           
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
import sys
import os
import click
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional, List
import time
import wandb
import random 
import numpy as np
import os.path as osp
import yaml
import shutil
import rasterio
from pathlib import Path 
from dotenv import load_dotenv  
import sys
# import handle interupt
import signal
import json
import warnings
# .............................................................
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as Func
from torchvision import transforms
from torch import Tensor, einsum
from pytorch_lightning import seed_everything
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
# from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from pytorch_lightning.tuner.tuning import Tuner
# .............................................................
import tifffile as tiff
import matplotlib.pyplot as plt
import signal
from PIL import Image
from tqdm import tqdm
from operator import itemgetter, mul
from functools import partial
from wandb import Artifact
from datetime import datetime
# .............................................................

# Add project directory to Python path for imports
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from scripts.process.process_helpers import handle_interrupt, read_minmax_from_json, print_tiff_info_TSX
from scripts.process.process_tiffs import create_event_datacube_copernicus, reproject_to_4326_gdal, make_float32_inf, resample_tiff_gdal, tile_geotiff_directly
from scripts.process.process_dataarrays import tile_datacube_rxr_inf
from scripts.train.train_helpers import is_sweep_run, pick_device
from scripts.train.train_classes import  UnetModel,   Segmentation_training_loop, Sen1Dataset
from scripts.train.train_functions import  loss_chooser, wandb_initialization, job_type_selector, create_subset
from scripts.inference_functions import create_weight_matrix, make_prediction_tiles, stitch_tiles, clean_checkpoint_keys, create_inference_csv, write_df_to_csv
from scripts.paths_class import ProjectPaths

start = time.time()

# Suppress num_workers warning - we've tested and num_workers=0 is optimal for our dataset
warnings.filterwarnings('ignore', '.*does not have many workers.*')

# logging.getLogger('scripts.process.process_helpers').setLevel(logging.WARNING)
# logging.getLogger('scripts.train.train_classess').setLevel(logging.WARNING)

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

@click.command()
@click.option('--train', is_flag=True,  help="Train the model")
@click.option('--test', is_flag=True, help="Test the model")
@click.option('--inference', is_flag=True, help="Run inference on a copernicus S1 image")
@click.option('--config', is_flag=True, help='loading from config')
@click.option('--fine_tune', is_flag=True, default=None, help="fine tune from training ckpt")
@click.option('--ckpt_input', is_flag=True, default=None, help="ckpt path is 'training folder or input folder'")

# //////////////////   MAIN   ///////////////////////

def main(train, test, inference, config, fine_tune, ckpt_input):
    print("\n" + "/"*40 + "\nRUNNING INFERSAR")
    n = 0
    for i in train, test, inference:
        if i:
            n += 1
    if n > 1 or n == 0:
        print("==========\nYOU MUST  SPECIFY ONE OF --TRAIN, --TEST OR --INFERENCE.\n==========")
        return
    # print click options
    print(f'\nwith flags:\n--train: {train}\n--test: {test}\n--inference: {inference}\n--config: {config}\n--fine_tune: {fine_tune}\n--ckpt_input: {ckpt_input}')    

    if  test:
        job_type =  "test"
        print("\n========== TESTING MODE ==========")
        # print('\ntesting ckpt in INPUT folder\n')
    elif train:
        job_type = "train" 
        print("========== TRAINING MODE ==========")
 
        if fine_tune:
            logger.info("\nFINE TUNING FROM TRAINING CKPT\n")
    elif inference:
        job_type = "inference"
        print("\n" + "="*40 + "\nINFERENCE MODE\n" + "="*40 )
        logger.info("\nNEEDS TO 'MAKE TIFFS' TO ENABLE NORMALISATION. WILL NOT MAKE DATACUBE. TILE DIRECTLY FROM THE NORAMLISED TIFS.\n" + "="*50 + "\n")

    device = pick_device()                       
    logger.info(f" Using device: {device}")

    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    
    #......................................................
    # INITIALIZE PATHS
    # project_path = Path(__file__).resolve().parent.parent
    logger.info(f"Project directory: {project_path}")
    
    paths = ProjectPaths(project_path=project_path)
    # Validate paths for the current job type
    path_errors = paths.validate_paths(job_type)
    if path_errors:
        for error in path_errors:
            logger.error(error)
        raise FileNotFoundError("Required paths validation failed. See errors above.")
    #......................................................
    # CONFIGURATION PARAMETERS

    DUAL_BAND_INPUT = False # True for dual band VV+VH input, False for single band (multi-file) input
    input_is_linear = True  # True for copernicus direct downloads, False for Sen1floods11
    # Training parameters
    dataset_name = "sen1floods11"  # "sen1floods11" or "copernicus_floods"
    run_name = "_rhoneleman"
    TRAINING_DATA_PRETILED = True
    subset_fraction = 1
    batch_size = 8 # 8 is tested as optimal for the macbook
    max_epoch = 15
    early_stop = True
    patience = 3
    num_workers = 0 
    # Logging and model parameters
    WandB_online = True
    LOGSTEPS = 50
    PRETRAINED = True
    in_channels = 2 # TODO ???
    DEVRUN = 0
    # Loss function parameters
    loss_description = 'bce_dice' # 'smp_bce' #  ''focal'

    focal_alpha = 0.8
    focal_gamma = 8
    bce_weight = 0.35 # FOR BCE_DICE
    # Data processing parameters
    # db_min = None
    # db_max = None
    tile_size = 512 
    stride = 512
    
    # Initialize variables
    stitched_img_path = None  # Will be set later based on mode
    # INFERENCE CONFIGURATION
    output_filename = '_sauvbelin'
    # sensor = 'S1'
    # date= '030126'
    threshold = 0.5 # THRESHOLD FOR METRICS + STITCHING. used in train class and inference stitching
    # ........................................................
    if input_is_linear:
        print("="*40 +f'\nINPUT IS LINEAR = {input_is_linear}')
    else:
        print("="*40 +f'\nINPUT IS dB')
    if DUAL_BAND_INPUT:
        print("="*40 +f'\nDUAL BAND INPUT = {DUAL_BAND_INPUT}')
    else:
        print("="*40 +f'\nINPUT IS SEPERATE , SINGLE BAND FILES')
    print("="*40 +f'\nWandB ONLINE = {WandB_online}')
    if WandB_online:
        print("."*40)
    
    MAKE_TIFS = None # INFERENCE NEEDS THIS
    MAKE_DATAARRAY = None # INFERENCE DOES NOT NEED THIS
    MAKE_TILES = None # INFERENCE NEEDS THIS
    if TRAINING_DATA_PRETILED:
        logger.info(" USING PRE-TILED TRAINING DATASET ")
        MAKE_TIFS = False
    if inference:
        # DO NOT CHANGE THESE 
        WandB_online = False
        MAKE_TIFS = True
        MAKE_TILES = True
        subset_fraction = 1
        batch_size = 1
        shuffle = False
        # DEFINE INFERENCE PATHS
        inference_paths = paths.get_inference_paths(
            tile_size=tile_size, 
            threshold=threshold,
            output_filename=output_filename
        )
        file_list_csv_path = inference_paths['file_list_csv_path']
        image_tiles_path = inference_paths['image_tiles_path']
        extracted = inference_paths['extracted_path']
        metadata_path = inference_paths['metadata_path']
        # print('\n' + '='*50 + '\nCHECK THESE PATHS ARE CORRECT FOR YOUR SETUP')
        # for p in inference_paths:
        #     path_value = inference_paths[p]
        #     display_path = path_value.relative_to(project_path / 'data' / '4final') if isinstance(path_value, Path) else path_value
        #     print(f'\n{p}: {display_path}')
        # print('='*50)

        image_code = paths.image_code
        logger.info(f"Image code: {image_code}")
    
        working_path = paths.predictions_path
        stitched_img_path = inference_paths['stitched_image']
        stitched_img_path = working_path /  f'{dataset_name}_{timestamp}_{tile_size}_{threshold}_{output_filename}.tif'
        if stitched_img_path.exists():
            logger.info(f"overwriting existing file! : {stitched_img_path}")

    if config:
        if not paths.main_config.exists():
            raise FileNotFoundError(f"Config file not found: {paths.main_config}")
        logger.info(f"Loading config from: {paths.main_config}")
        with open(paths.main_config, "r") as file:
            config_data = yaml.safe_load(file)
        threshold = config_data["threshold"] 
        tile_size = config_data["tile_size"] 
        input_file = Path(config_data['input_file'])
        output_folder = Path(config_data['output_folder'])
        output_filename = Path(config_data['output_filename'])

    persistent_workers = num_workers > 0

    if paths.env_file.exists():
        load_dotenv(paths.env_file)
    else:
        logger.info("Warning: .env not found; using shell environment")

    # if loss_description != 'focal':
    #     focal_alpha = None
    #     focal_gamma = None
    # if loss_description != 'bce_dice':
    #     bce_weight = None

    #####       WANDB INITIALIZATION + CONFIG       ###########
    project = "mac_py_package"
    wandb_config = {
        "name": run_name,
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "batch_size":batch_size,
        "loss_description": loss_description,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "bce_weight": bce_weight,
        "max_epoch": max_epoch,
    }
    wandb_logger = wandb_initialization(
        job_type, paths.project_path, project, dataset_name, run_name,
        paths.train_csv, paths.val_csv, paths.test_csv, wandb_config, WandB_online
    )
    config = wandb.config
    if loss_description == "focal":
        logger.info(f"focal_alpha: {wandb.config.get('focal_alpha', 'Not Found')}")
        logger.info(f"focal_gamma: {wandb.config.get('focal_gamma', 'Not Found')}")
        loss_desc = f"{loss_description}_{config.focal_alpha}_{config.focal_gamma}" 
    elif loss_description == "bce_dice":
        loss_desc = f"{loss_description}_{config.bce_weight}"
    else:
        loss_desc = loss_description

    run_name = f"{job_type}_{dataset_name}_{timestamp}_BS{config.batch_size}_s{config.subset_fraction}_{loss_desc}"  
    wandb.run.name = run_name

    # wandb.run.save()

    if is_sweep_run():
        logger.info(" IN SWEEP MODE <<<")
    

    #........................................................

    # Load dataset statistics (calculated beforehand)
    stats_file = paths.project_path / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        db_min = stats['global_minmax']['db_min']
        db_max = stats['global_minmax']['db_max']
        logger.info(f"Loaded dataset stats: db_min={db_min:.2f}, db_max={db_max:.2f}")
        print(F'............................\nUSING GLOBAL dB MIN {db_min} AND dB MAX {db_max}')
    else:
        logger.warning(f"No dataset statistics found at {stats_file}")
        logger.warning(f"Using hardcoded db_min={db_min}, db_max={db_max}")
        logger.warning("Consider running scan_dataset_minmax() first!")

    if train:
        file_list_csv_path = paths.train_csv
        training_paths = paths.get_training_paths()
        image_tiles_path = training_paths['image_tiles_path']
    elif test:
        file_list_csv_path = paths.test_csv
        training_paths = paths.get_training_paths()
        image_tiles_path = training_paths['image_tiles_path']

    # Only delete and recreate folders for inference mode
    print(f'\nMAKE_TIFS = {MAKE_TIFS},\nMAKE_DATAARRAY = {MAKE_DATAARRAY}, \nMAKE_TILES = {MAKE_TILES}\n')
    logger.info(f'training threshold = {threshold}, tile_size = {tile_size}, stride = {stride}')


    if MAKE_TIFS:
        # CREATES 2 SEPARATE TIFS IN EXTRACTED DIR, FOR VV AND VH CHANNELS (OR COPIES IF ALREADY SEPARATE)
        logger.info('EXTRACTING VV AND VH CHANNELS')
        predict_input = inference_paths['predict_input_path']

        if DUAL_BAND_INPUT:
            # Filter for .tif files specifically, ignoring system files
            print('='*40 +'DUAL BAND INPUT')

            valid_files = [f for f in predict_input.iterdir() 
                           if f.is_file() 
                           and not f.name.startswith('.') 
                           and f.suffix.lower() in ['.tif', '.tiff']]

            if not valid_files:
                logger.error(f"No valid .tif image files found in {predict_input}")
                raise FileNotFoundError(f"No valid .tif image files found in {predict_input}")
        
            # check there is only one file
            if len(valid_files) > 1:
                logger.error(f"DUAL_BAND_INPUT is True but multiple .tif files found in {predict_input}")
                raise ValueError(f"DUAL_BAND_INPUT is True but multiple .tif files found in {predict_input}")
        
            image = valid_files[0]  # Take the first valid .tif file
            logger.info(f"--- Image to process: {image}")

            if not image or not image.is_file():
                logger.error(f"Image not found: {image}")
                raise FileNotFoundError(f"Image not found: {image}")

            # Extract the vv and vh channels from the geotiff and make 2 separate tifs
            with rasterio.open(image) as src:
                logger.info(f"--- Image shape: {src.shape}")
                logger.info(f"--- Image CRS: {src.crs}")
                logger.info(f"--- Image bounds: {src.bounds}")
                logger.info(f"--- Image transform: {src.transform}")

                # extract the VV and VH bands
                vv_band = src.read(1)  # Assuming VV is the first band
                vh_band = src.read(2)  # Assuming VH is the second band

                # Create new TIF file_paths for VV and VH
                vv_image_path = extracted / f"{image.stem}_VV.tif"
                vh_image_path = extracted / f"{image.stem}_VH.tif"

                logger.info(f'extracted= {extracted}')
                logger.info(f"--- VV image path: {vv_image_path}")
                logger.info(f"--- VH image path: {vh_image_path}")

                # Write the VV band to a new TIF file
                with rasterio.open(vv_image_path, 'w', driver='GTiff', height=vv_band.shape[0], width=vv_band.shape[1],
                                   count=1, dtype=vv_band.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(vv_band, 1)
                logger.info(f"--- VV band saved to {vv_image_path}")

                # Write the VH band to a new TIF file
                with rasterio.open(vh_image_path, 'w', driver='GTiff', height=vh_band.shape[0], width=vh_band.shape[1],
                                   count=1, dtype=vh_band.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(vh_band, 1)
                logger.info(f"--- VH band saved to {vh_image_path}")

                # Check if the VV and VH images were created successfully
                if not vv_image_path.exists() or not vh_image_path.exists():
                    logger.error(f"Failed to create VV or VH images")
                    raise RuntimeError(f"Failed to create VV or VH images")

                logger.info(f"wrote VV tif at vv_image_path: {vv_image_path}")
                logger.info(f"wrote VH tiff at vh_image_path: {vh_image_path}")
        else:
            print("="*40 + "\nDUAL IMAGE INPUT")
            vv_image_path = None
            vh_image_path = None

            for file in predict_input.iterdir():
                if file.suffix.lower() in ['.tif', '.tiff']:
                    if '_vv_' in file.name.lower():
                        vv_image_path = file
                    elif '_vh_' in file.name.lower():
                        vh_image_path = file

            # Verify both files were found
            if not vv_image_path or not vh_image_path:
                raise FileNotFoundError(f"Missing VV or VH files in {predict_input}")
            
            logger.info(f"Found inference VV: {vv_image_path}")
            logger.info(f"Found inference VH: {vh_image_path}")
            # copy files to extracted folder
            if not extracted.exists():
                extracted.mkdir(parents=True, exist_ok=True)

            shutil.copy(vv_image_path, extracted / 'vv_copy.tif')
            vv_image_path = extracted / 'vv_copy.tif'
            logger.info(f"Copied VV to: {vv_image_path}")

            shutil.copy(vh_image_path, extracted / 'vh_copy.tif')
            vh_image_path = extracted / 'vh_copy.tif'

            logger.info(f"Copied VH to: {vh_image_path}")


        if True:
            pass
            # THIS is processing steps for TERRASAR-X DATA - NOT USING FOR COPERNICUS S1 DATA
            # TODO MAKE SURE EXTENSION = TIF NOT TIFF
            # if extracted.exists():
                # logger.info(f"--- Deleting existing extracted folder: {extracted}")
                # delete the folder and create a new one
            #     shutil.rmtree(extracted)
            # extracted.mkdir(exist_ok=True)

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

    # # Initialize cube variable for inference mode
    # cube = None
    # metadata = None
    
    if MAKE_DATAARRAY:
        print(f'=========== MAKING DATAARRAY =========')
        print(f'WORKING IN EXTRACTED AT:{extracted}')

        if not extracted.exists():
            # create the directory if it does not exist
            extracted.mkdir(parents=True, exist_ok=True)
            logger.info(f"--- Created extract directory: {extracted}")


            # raise FileNotFoundError(f"extract directory not found: {extracted}")
        create_event_datacube_copernicus(extracted, paths.image_code)
        if cube is None:
            raise FileNotFoundError(f"No data cube found in {extracted}")

    if MAKE_TILES:
        print(f'='*40 + '\nMAKING TILES')
        if inference:
            # INFERENCE: Tile directly from GeoTIFF (no datacube needed)
            logger.info("Tiling directly from GeoTIFF for inference")

            if image_tiles_path.exists():
                logger.info(f"Deleting existing inference tiles: {image_tiles_path}")
                shutil.rmtree(image_tiles_path)
            image_tiles_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"About to tile from:")
            logger.info(f"  VV: {vv_image_path}")
            logger.info(f"  VH: {vh_image_path}")
            logger.info(f"  Output: {image_tiles_path}")
            logger.info(f"  Tile size: {tile_size}, Stride: {stride}")
    
            # Check files exist
            if not vv_image_path.exists():
                logger.error(f"VV file does not exist: {vv_image_path}")
            if not vh_image_path.exists():
                logger.error(f"VH file does not exist: {vh_image_path}")

            # Check image dimensions
            with rasterio.open(vv_image_path) as src:
                logger.info(f"VV image dimensions: {src.shape}")
                logger.info(f"VV image size: height={src.height}, width={src.width}")

            # Tile directly from GeoTIFF
            tiles, metadata = tile_geotiff_directly(
                vv_image=vv_image_path,
                vh_image=vh_image_path,
                output_path=image_tiles_path,
                tile_size=tile_size,
                stride=stride
            )
            logger.info(f'metadata path: {metadata_path}')
            logger.info(f'metadata: {metadata}')
        else:
            cube = next(extracted.rglob("*.nc"), None)
            if cube is None:
                logger.error("Cannot create tiles: no data cube available")
                return
            if image_tiles_path.exists():
                logger.info(f"--- Deleting existing inference tiles folder: {image_tiles_path}")
                shutil.rmtree(image_tiles_path)
            image_tiles_path.mkdir(exist_ok=True, parents=True)
            extracted.mkdir(exist_ok=True, parents=True)
            # TILE DATACUBE
            #  meteada is a dict used for stcihing later
            tiles, metadata = tile_datacube_rxr_inf(cube, image_tiles_path, tile_size=tile_size, stride=stride, percent_non_flood=0, inference = False) 

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"---Saved {len(metadata)} tile_metadata to {metadata_path}")
        logger.info(f"{len(tiles)} tiles saved to {image_tiles_path}")

        # CREATE CSV FOR INFERENCE DATALOADER USING THE METADATA

        inference_list_df = create_inference_csv(metadata)

        inference_list_df.to_csv(file_list_csv_path, index=False, header=False)
        logger.info(f"---Inference CSV created at {file_list_csv_path}")
        # write_df_to_csv(inference_list_dataframe, file_list_csv_path)

    # VERIFY CSV EXISTS OR WAS MADE  
    if not file_list_csv_path.exists():
        logger.error(f"Failed to create inference CSV file: {file_list_csv_path}")
        raise FileNotFoundError(f"CSV file does not exist / was not created: {file_list_csv_path}")
        ########     INITIALISE THE MODEL     #########
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED).to(device)

    #........................................................
    # CREATE DATALOADERS BASED ON JOB TYPE
    # EXISTING CKPTSONLY NEEDED FOR FINETUNING
    if ckpt_input:
        ckpt_path = next(paths.ckpt_input_path.rglob("*.ckpt"), None)
    else:
        # get latest checkpoint in training folder
        ckpt_path = max(paths.ckpt_training_path.rglob("*.ckpt"), key=os.path.getctime, default=None)
    if ckpt_path is None:
        logger.error(f"*No checkpoint found in: {paths.ckpt_input_path}")
        return
    if train:
        # Training dataset
        # normalisation handled in dataset class
        train_dataset = Sen1Dataset(
            job_type="train",
            working_path=paths.training_path,
            images_path=paths.images_path,
            labels_path=paths.labels_path,
            csv_path=paths.train_csv,
            image_code=paths.image_code,
            input_is_linear=input_is_linear,
            db_min=db_min,
            db_max=db_max)
        
        # Validation dataset
        val_dataset = Sen1Dataset(
            job_type="val",
            working_path=paths.training_path,
            images_path=paths.images_path,
            labels_path=paths.labels_path,
            csv_path=paths.val_csv,
            image_code=paths.image_code,
            input_is_linear=input_is_linear,
            db_min=db_min,
            db_max=db_max)
        
        # Apply subset if needed
        if subset_fraction < 1:
            train_subset_indices = random.sample(range(len(train_dataset)), int(subset_fraction * len(train_dataset)))
            train_dataset = Subset(train_dataset, train_subset_indices)
            # Use larger subset for validation to ensure class diversity
            val_subset_fraction = max(subset_fraction, 0.8)  # Use at least 80% of validation data
            val_subset_indices = random.sample(range(len(val_dataset)), int(val_subset_fraction * len(val_dataset)))
            val_dataset = Subset(val_dataset, val_subset_indices)
            logger.info(f"Using {subset_fraction} of training data and {val_subset_fraction} of validation data")
        
        # Create dataloaders
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
        
    elif test:

        # Test dataset
        test_dataset = Sen1Dataset(
            job_type="test",
            working_path=paths.test_path,
            images_path=paths.images_path,
            labels_path=paths.labels_path,
            csv_path=paths.test_csv,
            image_code=paths.image_code,
            input_is_linear=input_is_linear,
            db_min=db_min,
            db_max=db_max)
        
        # Apply subset if needed
        if subset_fraction < 1:
            test_subset_indices = random.sample(range(len(test_dataset)), int(subset_fraction * len(test_dataset)))
            test_dataset = Subset(test_dataset, test_subset_indices)
        
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
        
    elif inference:
        logger.debug(">>>>>>>>> CREATING INFERENCE DATALOADER")
        # Inference dataset
        inference_dataset = Sen1Dataset(
            job_type="inference",
            working_path=working_path,
            images_path=paths.images_path,
            labels_path=paths.labels_path,
            csv_path=file_list_csv_path,
            image_code=image_code,
            input_is_linear=input_is_linear,
            db_min=db_min,
            db_max=db_max)
        
        subset_indices = random.sample(range(len(inference_dataset)), int(subset_fraction * len(inference_dataset)))
        inference_subset = Subset(inference_dataset, subset_indices)
        
        dataloader = DataLoader(inference_subset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, shuffle=False)
    
    print(f'='*40 + f'\nCKPT NAME: {ckpt_path.name}\n' + '='*40)

    if test or inference:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            return
        # EXTRACT THE MODEL STATE DICT
        cleaned_state_dict = clean_checkpoint_keys(ckpt["state_dict"])

        # LOAD THE MODEL STATE DICT
        try:
            model.load_state_dict(cleaned_state_dict)
            print(f"\nCHECKPOINT:  {ckpt_path.name}\n")
        except Exception as e:
            logger.error(f"Failed to load model state dict: {e}")
            return


    # .........................................................
    # CHOOSE LOSS FUNCTION 
    if train or test:
        loss_fn = loss_chooser(loss_description, config.focal_alpha, config.focal_gamma, config.bce_weight, device=device)
        

        
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,  # Stop if no improvement for 3 consecutive epochs
            mode="min",
    )
        # .........................................................
        # SETUP TRAINING LOOP    
        ckpt_save_dir = paths.ckpt_training_path
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_save_dir,
            filename=run_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        if early_stop:
            callbacks=[checkpoint_callback, early_stopping]
        else:
            callbacks=[checkpoint_callback]
        # ---------- precision & accelerator automatically picked ------------- #
        #  – MPS  : bf16 autocast, accelerator="mps"
        #  – CUDA : fp16 autocast, accelerator="gpu"
        #  – CPU  : fp32
        #
        logger.info(f'device type = {device.type}')
        accelerator = (
            "mps"
            if device.type == "mps"
            else "gpu"
            if device.type == "cuda"
            else "cpu"
        )

        precision = (
            "bf16-mixed"
            if device.type == "mps"
            else "16-mixed"
            if device.type == "cuda"
            else 32
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=LOGSTEPS,
            max_epochs=config.max_epoch,
            accelerator=accelerator,
            devices=1,
            precision=32, # precision=precision once pytorch is upgraded to enable mixed which is faster on m2
            fast_dev_run=DEVRUN,
            num_sanity_val_steps=2,
            callbacks=callbacks,
        )
        # get info about train_dl

        # imgs, masks, valids, fnames = next(iter(train_dl))
        # → torch.Size([B, 2, H, W]) torch.Size([B, 1, H, W]) torch.Size([B, 1, H, W])


        # Training or Testing
        if train:
            logger.info(" Starting training")
            training_loop = Segmentation_training_loop(model, loss_fn, stitched_img_path, loss_description, metric_threshold=threshold)
            trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl)

        elif test:
            logger.debug(f" Starting testing.....")
            training_loop = Segmentation_training_loop.load_from_checkpoint(
                checkpoint_path=ckpt_path,  model=model, loss_fn=loss_fn, save_path=stitched_img_path, loss_description=loss_description, metric_threshold=threshold
            )
            trainer.test(model=training_loop, dataloaders=test_dl)

    # //////////////////////////////////////////////////////////////
    # INFERENCE LOOP
    elif inference:

        pred_tiles_path = inference_paths['pred_tiles_path']
        # DELETE THE PREDICTION FOLDER IF IT EXISTS
        if pred_tiles_path.exists():
            logger.info(f"Deleting existing predictions folder: {pred_tiles_path}")
            shutil.rmtree(pred_tiles_path)
        pred_tiles_path.mkdir(exist_ok=True)

        #  # MAKE PREDICTION ON TILES ////////////////////////////////

        logger.info("\n\n========== MAKING PREDICTIONS ON TILES==========\n")
            # CREATE A WEIGHT MATRIX FOR BLENDING

        with torch.no_grad():
            t = 1
            global_min = float('inf')
            global_max = float('-inf')
            for imgs, valids, fnames in tqdm(dataloader, desc="Predict"):
                logger.debug(f'/////Processing batch {t} with {imgs.shape[0]} tiles')
                t += 1
                logger.debug(f'images shape: {imgs.shape}, valids shape: {valids.shape}, fnames: {fnames}')
                logger.debug(f"First image filename: {fnames[0]}")

                first_img = imgs[0]

                logger.debug(f"First image tensor shape: {first_img.shape}")
                logger.debug(f"First image dtype: {first_img.dtype}")
                logger.debug(f"First image min: {first_img.min().item()}, max: {first_img.max().item()}")
        
                # Inspect each channel
                vv = first_img[0]
                vh = first_img[1]
                logger.debug(f"VV min/max: {vv.min().item()}/{vv.max().item()}")
                logger.debug(f"VH min/max: {vh.min().item()}/{vh.max().item()}")

                lmin, lmax = min(vv.min(),vh.min()), max(vv.max(), vh.max())
                global_min = min(global_min, lmin)
                global_max = max(global_max, lmax)
                
                imgs   = imgs.to(device)            # [B,2,H,W]
                logits = model(imgs)
                probs  = torch.sigmoid(logits).cpu()  # back to CPU for numpy/rasterio
                preds  = (probs > threshold).float()  # [B,1,H,W]
                for b, name in enumerate(fnames):
                    out = preds[b, 0].numpy()                 # 2-D
                    out[~valids[b, 0].numpy().astype(bool)] = 0  # mask invalid px

                    src_path = image_tiles_path / name
                    with rasterio.open(src_path) as src:
                        profile = src.profile
                    profile.update(dtype="float32", count=1)

                    dst_path = pred_tiles_path / name
                    with rasterio.open(dst_path, "w", **profile) as dst:
                        dst.write(out.astype("float32"), 1)  
            print(f'GLOBAL MIN MAX OVER ALL TILES: {global_min} , {global_max}')
            ims_list = list(pred_tiles_path.glob("*.tif"))
            if len(ims_list) > 0:
                # Load metadata for stitching
                logger.info(f"Loading metadata from: {metadata_path}")
                if not metadata_path.exists():
                    return

                with open(metadata_path, "r") as f:
                    metadata = json.load(f) 

        # STITCH PREDICTION TILES /////////////////////////////////////////////

        input_image = next(extracted.rglob("*.tif"), None) if extracted.exists() else None
        if input_image and ('vv' in input_image.name.lower() or 'vh' in input_image.name.lower()) and input_image.suffix.lower() == '.tif':
            logger.info(f"---input_image: {input_image}")
            logger.info(f"---pred_tiles_path: {pred_tiles_path}")
            logger.info(f'---extracted folder: {extracted}')
            stitch_tiles(metadata, pred_tiles_path, stitched_img_path, input_image, tile_size, stride, threshold)
        else:
            logger.warning(f"No suitable input image found in {extracted} for stitching")
            logger.info("Prediction tiles created but stitching skipped")  
  

    # Cleanup
    run_time = (time.time() - start) / 60
    print(f" Total runtime: {run_time:.2f} minutes")
    wandb.finish()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    end = time.time()
    print(f"Time taken inc wand sync: {((end - start) / 60):.2f} minutes")

if __name__ == '__main__':
    main()
