"""
Flood detection model training, testing, and inference script.

This script uses a centralized ProjectPaths class to manage all directory and file paths,
making the code more maintainable and easier to understand.

Usage:
    python run_master_copy.py --train    # Train the model
    python run_master_copy.py --test     # Test the model  
    python run_master_copy.py --inference # Run inference
    python run_master_copy.py --config   # Use config file
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
from scripts.inference_functions import make_prediction_tiles, stitch_tiles, clean_checkpoint_keys, create_inference_csv, write_df_to_csv

start = time.time()


logging.getLogger('scripts.process.process_helpers').setLevel(logging.INFO)
logging.getLogger('scripts.train.train_classess').setLevel(logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

class ProjectPaths:
    """Centralized path management for the flood detection project"""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._image_code = None
        
        # Main working directories
        self.dataset_path = self.project_path / "data" /  "4final" / "dataset"
        self.training_path = self.project_path / "data" / "4final" / "training"
        self.predictions_path = self.project_path / "data" / "4final" / 'predictions'
        self.test_path = self.project_path / "data" / "4final" / "testing"
        
        # Data subdirectories
        self.images_path = self.dataset_path / 'S1Hand'
        self.labels_path = self.dataset_path / 'LabelHand'
        
        # CSV files
        self.train_csv = self.dataset_path / "flood_train_data.csv"
        self.val_csv = self.dataset_path / "flood_valid_data.csv"
        self.test_csv = self.dataset_path / "flood_test_data.csv"
        
        # Checkpoint directories (consolidated)
        self.ckpt_input_path = project_path / "checkpoints" / 'ckpt_input'
        self.ckpt_training_path = self.project_path / "checkpoints" / "ckpt_training"
        
        # Config files
        self.main_config = project_path / "configs" / "floodaiv2_config.yaml"
        self.minmax_config = project_path / "configs" / "global_minmax_INPUT" / "global_minmax.json"
        
        # Environment file
        self.env_file = self.project_path / ".env"

    @property
    def image_code(self) -> str:
        if self._image_code is None:  
            predict_input = self.project_path / "data" / "4final" / "predict_input"
            # DEFINE PREDICT_INPUT
            if not predict_input.exists():
                raise FileNotFoundError(f"Predict input not found: {predict_input}")
            # FIND THE INPUT IMAGE TO EXTRACT IMAGE CODE
            input_names = [f for f in predict_input.iterdir() 
                           if f.is_file() 
                           and not f.name.startswith('.') 
                           and f.suffix.lower() in ['.tif', '.tiff']]
            if input_names:
                # extract file name
                splits = input_names[0].stem.split('_')
                raw_code = '_'.join(splits[:2])
                # Sanitize: replace colons with hyphens or underscores 
                self._image_code = raw_code.replace(':', '-')  # ← ADD THIS   
     

            # OTHERWISE GET IT FROM THE TILE FOLDER NAME
            else:
                logger.info(f"No '.tif' imput image found in {predict_input}\nso looking  in tile folder name in {self.predictions_path} for image_code...")
   

                tile_folders = [f for f in self.predictions_path.iterdir() if not f.name.startswith('.') and  f.is_dir() and "tiles" in f.name.lower()]

                if not tile_folders:
                    raise FileNotFoundError(f"No input files or tile folders found in {self.predictions_path}")
                # Extract image code from folder name
                # e.g., "Ghana_313799_tiles" -> "Ghana_313799"
                folder_name = tile_folders[0].name
                self._image_code = folder_name.replace("_tiles", "")
                logger.info(f"Extracted image_code from pre-tiled folder: {self._image_code}")
        return self._image_code

 

    
    def get_inference_paths(self, sensor: str = 'sensor', date: str = 'date', tile_size: int =512, threshold: float = 0.5, output_filename: str = '_name') -> dict:
        # GRAB OUTPUT_FILENAME
      
        image_tiles_path = self.predictions_path / f'tiles'
        return {
            'predict_input_path': self.project_path / "data" / "4final" / "predict_input",
            'pred_tiles_path': self.predictions_path / f'{output_filename}_predictions',
            'image_tiles_path': image_tiles_path,
            'extracted_path': self.predictions_path / 'extracted',
            'file_list_csv_path': self.predictions_path / "predict_tile_list.csv",
            'stitched_image': self.predictions_path / f'{sensor}_{self.image_code}_{date}_{tile_size}_{threshold}_{output_filename}_WATER_AI.tif',
            'metadata_path': image_tiles_path / 'tile_metadata.json'
        }
    
    def get_training_paths(self):
        """Get training/testing specific paths"""
        return {
            'image_tiles_path': self.dataset_path,
            'metadata_path': self.dataset_path / 'tile_metadata_pth.json'
        }
    
    def validate_paths(self, job_type: str):
        """Validate that required paths exist for the given job type"""
        errors = []
        required_paths = []
        if job_type in ('train', 'test'):
            required_paths = [
                (self.dataset_path, "Dataset directory"),
                (self.images_path, "Images directory"),
                (self.labels_path, "Labels directory"),
            ]
            
            if job_type == 'train':
                required_paths.extend([
                    (self.train_csv, "Training CSV"),
                    (self.val_csv, "Validation CSV")
                ])
            elif job_type == 'test':
                required_paths.append((self.test_csv, "Test CSV"))
                
        elif job_type == 'inference':
            # Inference paths are created dynamically, less validation needed
            pass
            
        # Always check checkpoint folder
        required_paths.append((self.ckpt_input_path, "Checkpoint input directory"))
        
        for path, description in required_paths:
            if not path.exists():
                errors.append(f"{description} not found: {path}")
        
        # Check for checkpoint files
        if not any(self.ckpt_input_path.rglob("*.ckpt")):
            errors.append(f"No checkpoint files found in: {self.ckpt_input_path}")
            
        return errors

@click.command()
@click.option('--train', is_flag=True,  help="Train the model")
@click.option('--test', is_flag=True, help="Test the model")
@click.option('--inference', is_flag=True, help="Run inference on a copernicus S1 image")
@click.option('--config', is_flag=True, help='loading from config')
@click.option('--fine_tune', is_flag=True, default=None, help="fine tune from training ckpt")
@click.option('--ckpt_input', is_flag=True, default=None, help="ckpt path is 'training folder or input folder'")

# //////////////////   MAIN   ///////////////////////

def main(train, test, inference, config, fine_tune, ckpt_input):
    n = 0
    for i in train, test, inference:
        if i:
            n += 1
    if n > 1 or n == 0:
        print("==========\nYOU MUST  SPECIFY ONE OF --TRAIN, --TEST OR --INFERENCE.\n==========")
        return

    if  test:
        job_type =  "test"
        print("========== TESTING MODE ==========")
        logger.info(' ARE YOU TESTING THE CORRECT CKPT? <<<')
    elif train:
        job_type = "train" 
        print("========== TRAINING MODE ==========")
        if fine_tune:
            logger.info("\nFINE TUNING FROM TRAINING CKPT\n")
    elif inference:
        job_type = "inference"
        print("\n" + "="*20 + "INFERENCE MODE" + "="*20 )
        logger.info("\nNEEDS TO 'MAKE TIFFS' TO ENABLE NORMALISATION. WILL NOT MAKE DATACUBE. TILE DIRECTLY FROM THE NORAMLISED TIFS.\n" + "="*50 + "\n")
    
    print(f"YOU ARE USING A CONFIG FILE: {config}")

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
    dataset_name = "sen1floods11"  # "sen1floods11" or "copernicus_floods"
    run_name = "_runname"
    SINGLE_INPUT = False  # True for VV+VH input, False for single band
    input_is_linear = False   # True for copernicus direct downloads, False for Sen1floods11
    # Training parameters
    subset_fraction = 1
    batch_size = 8
    max_epoch = 15
    early_stop = True
    patience = 3
    num_workers = 0 # 8 TODO
    threshold = 0.5  # Default threshold for training
    # Logging and model parameters
    WandB_online = True
    LOGSTEPS = 50
    PRETRAINED = True
    in_channels = 2
    DEVRUN = 0
    # Loss function parameters
    loss_description = 'bce_dice' #'smp_bce' # 'bce_dice' #'focal' # 'bce_dice' # focal'
    focal_alpha = 0.8
    focal_gamma = 8
    bce_weight = 0.35 # FOR BCE_DICE
    # Data processing parameters
    db_min = -30.0
    db_max = 0.0
    tile_size = 512
    stride = tile_size
    # Processing flags - default to False, set to True in specific modes
    MAKE_TIFS = None # INFERENCE NEEDS THIS
    MAKE_DATAARRAY = None # INFERENCE DOES NOT NEED THIS
    MAKE_TILES = None # INFERENCE NEEDS THIS
    # Initialize variables
    stitched_image = None  # Will be set later based on mode


    # ...........................................................
    # MODE-SPECIFIC CONFIGURATION
    if inference:
        # DO NOT CHANGE THESE 
        MAKE_TIFS = True
        MAKE_TILES = True
        subset_fraction = 1
        batch_size = 1
        shuffle = False
        # DEFINE INFERENCE PATHS
        inference_paths = paths.get_inference_paths(
            sensor='sensor', 
            date='date', 
            tile_size=tile_size, 
            threshold=threshold,
            output_filename= '_name'
        )
        file_list_csv_path = inference_paths['file_list_csv_path']
        image_tiles_path = inference_paths['image_tiles_path']
        extracted = inference_paths['extracted_path']
        metadata_path = inference_paths['metadata_path']
        print('\n' + '='*50 + '\nCHECK THESE PATHS ARE CORRECT FOR YOUR SETUP')
        for p in inference_paths:
            path_value = inference_paths[p]
            display_path = path_value.relative_to(project_path / 'data' / '4final') if isinstance(path_value, Path) else path_value
            print(f'\n{p}: {display_path}')
        print('='*50)

        input_is_linear = True  # NEEDS TO BE EXACT SAME AS TRAINING. S1 COPERNICUS IS LINEAR, s1floods11 IS NOT
        threshold = 0.5
        image_code = paths.image_code
        logger.info(f"Image code: {image_code}")
    
        working_path = paths.predictions_path
        stitched_image = inference_paths['stitched_image']

        if stitched_image.exists():
            logger.info(f"overwriting existing file! : {stitched_image}")

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

    run_name = f"{dataset_name}_{timestamp}_BS{config.batch_size}_s{config.subset_fraction}_{loss_desc}"  
    wandb.run.name = run_name
    # wandb.run.save()

    if is_sweep_run():
        logger.info(" IN SWEEP MODE <<<")
    

    #........................................................
    #########    TRAIN / TEST - CREATE DATA LOADERS    #########
    if train:
        file_list_csv_path = paths.train_csv
        training_paths = paths.get_training_paths()
        image_tiles_path = training_paths['image_tiles_path']
    elif test:
        file_list_csv_path = paths.test_csv
        training_paths = paths.get_training_paths()
        image_tiles_path = training_paths['image_tiles_path']

    # Only delete and recreate folders for inference mode
    logger.info(f'\n\nMAKE_TIFS = {MAKE_TIFS},\nMAKE_DATAARRAY = {MAKE_DATAARRAY}, \nMAKE_TILES = {MAKE_TILES}\n')
    logger.info(f'training threshold = {threshold}, tile_size = {tile_size}, stride = {stride}')


    if MAKE_TIFS:
        logger.info('EXTRACTING VV AND VH CHANNELS')
        predict_input = inference_paths['predict_input_path']

        if SINGLE_INPUT:
            # Filter for .tif files specifically, ignoring system files
            print('=====SINGLE MULTIBAND INPUT======')

            valid_files = [f for f in predict_input.iterdir() 
                           if f.is_file() 
                           and not f.name.startswith('.') 
                           and f.suffix.lower() in ['.tif', '.tiff']]

            if not valid_files:
                logger.error(f"No valid .tif image files found in {predict_input}")
                raise FileNotFoundError(f"No valid .tif image files found in {predict_input}")
        
            # check there is only one file
            if len(valid_files) > 1:
                logger.error(f"SINGLE_INPUT is True but multiple .tif files found in {predict_input}")
                raise ValueError(f"SINGLE_INPUT is True but multiple .tif files found in {predict_input}")
        
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
            print('=====DUAL IMAGE INPUT======')
            vv_image_path = None
            vh_image_path = None

            for file in predict_input.iterdir():
                if file.suffix.lower() in ['.tif', '.tiff']:
                    if 'vv' in file.name.lower():
                        vv_image_path = file
                    elif 'vh' in file.name.lower():
                        vh_image_path = file

            # Verify both files were found
            if not vv_image_path or not vh_image_path:
                raise FileNotFoundError(f"Missing VV or VH files in {predict_input}")
            
            logger.info(f"Found VV: {vv_image_path}")
            logger.info(f"Found VH: {vh_image_path}")
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
        print(f'=========== MAKING TILES =========')
        if inference:
            # INFERENCE: Tile directly from GeoTIFF (no datacube needed)
            logger.info("Tiling directly from GeoTIFF for inference")

            # Get the VV or VH image (either works for georeferencing)
            # vv_image_path = next(extracted.glob("*_VV.tif"), None)
            # if not vv_image_path or not vv_image.exists():
            #     logger.error(f"VV image not found in {extracted}")
            #     raise FileNotFoundError(f"VV image not found in {extracted}")

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
    if train and fine_tune:
        # INPUT bool helper
        if ckpt_input:
            ckpt_path = next(paths.ckpt_input_path.rglob("*.ckpt"), None)
        else:
            # get latest checkpoint in training folder
            ckpt_path = max(paths.ckpt_training_path.rglob("*.ckpt"), key=os.path.getctime, default=None)
        if ckpt_path is None:
            logger.error(f"No checkpoint found in: {paths.ckpt_input_path}")
            return

        # Training dataset
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
        # INPUT bool helper
        if ckpt_input:
            ckpt_path = next(paths.ckpt_input_path.rglob("*.ckpt"), None)
        else:
            # get latest checkpoint in training folder
            ckpt_path = max(paths.ckpt_training_path.rglob("*.ckpt"), key=os.path.getctime, default=None)
        if ckpt_path is None:
            logger.error(f"No checkpoint found in: {paths.ckpt_input_path}")
            return

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

        ckpt_path = next(paths.ckpt_input_path.rglob("*.ckpt"), None)
        if ckpt_path is None:
            logger.error(f"No checkpoint found in: {paths.ckpt_input_path}")
            return

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
            print(f"CHECKPOINT->{ckpt_path.name}")
        except Exception as e:
            logger.error(f"Failed to load model state dict: {e}")
            return


    # .........................................................
    # CHOOSE LOSS FUNCTION 
    if train or test:
        loss_fn = loss_chooser(loss_description, config.focal_alpha, config.focal_gamma, config.bce_weight)
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
            training_loop = Segmentation_training_loop(model, loss_fn, stitched_image, loss_description)
            trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl)

        elif test:
            logger.info(f" Starting testing with checkpoint: {ckpt_path}")
            training_loop = Segmentation_training_loop.load_from_checkpoint(
                checkpoint_path=ckpt_path,  model=model, loss_fn=loss_fn, save_path=stitched_image, loss_description=loss_description
            )
            trainer.test(model=training_loop, dataloaders=test_dl)

    # .........................................................
    # INFERENCE LOOP
    elif inference:

        pred_tiles_path = inference_paths['pred_tiles_path']
        # DELETE THE PREDICTION FOLDER IF IT EXISTS
        if pred_tiles_path.exists():
            logger.info(f"Deleting existing predictions folder: {pred_tiles_path}")
            shutil.rmtree(pred_tiles_path)
        pred_tiles_path.mkdir(exist_ok=True)

        #  # MAKE PREDICTION TILES
        logger.info("\n\n========== MAKING PREDICTION TILES==========\n")
        with torch.no_grad():
            t = 1
            for imgs, valids, fnames in tqdm(dataloader, desc="Predict"):
                logger.info(f'/////Processing batch {t} with {imgs.shape[0]} tiles')
                t += 1
                logger.info(f'images shape: {imgs.shape}, valids shape: {valids.shape}, fnames: {fnames}')
                logger.info(f"First image filename: {fnames[0]}")

                first_img = imgs[0]

                logger.info(f"First image tensor shape: {first_img.shape}")
                logger.info(f"First image dtype: {first_img.dtype}")
                logger.info(f"First image min: {first_img.min().item()}, max: {first_img.max().item()}")
        
                # Inspect each channel
                vv = first_img[0]
                vh = first_img[1]
                logger.info(f"VV min/max: {vv.min().item()}/{vv.max().item()}")
                logger.info(f"VH min/max: {vh.min().item()}/{vh.max().item()}")
                
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

            ims_list = list(pred_tiles_path.glob("*.tif"))
            if len(ims_list) > 0:
                # Load metadata for stitching
                logger.info(f"Loading metadata from: {metadata_path}")
                if not metadata_path.exists():
                    return

                with open(metadata_path, "r") as f:
                    metadata = json.load(f) 

        # STITCH PREDICTION TILES
        input_image = next(extracted.rglob("*.tif"), None) if extracted.exists() else None
        if input_image and ('vv' in input_image.name.lower() or 'vh' in input_image.name.lower()) and input_image.suffix.lower() == '.tif':
            logger.info(f"---input_image: {input_image}")
            logger.info(f"---pred_tiles_path: {pred_tiles_path}")
            logger.info(f'---extracted folder: {extracted}')
            stitch_tiles(metadata, pred_tiles_path, stitched_image, input_image)
        else:
            logger.warning(f"No suitable input image found in {extracted} for stitching")
            logger.info("Prediction tiles created but stitching skipped")  

    # Cleanup
    run_time = (time.time() - start) / 60
    logger.info(f" Total runtime: {run_time:.2f} minutes")
    wandb.finish()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    end = time.time()
    logger.info(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == '__main__':
    main()
