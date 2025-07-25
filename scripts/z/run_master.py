import sys
import os
import click
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional, List
import time
import wandb
import random 
import numpy as np
import os.path as osp
import logging
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

from scripts.process.process_helpers import handle_interrupt, read_minmax_from_json
from scripts.process.process_tiffs import create_event_datacube_copernicus, reproject_to_4326_gdal, make_float32_inf, resample_tiff_gdal
from scripts.process.process_dataarrays import tile_datacube_rxr_inf
from scripts.train.train_helpers import is_sweep_run, pick_device
from scripts.train.train_classes import  UnetModel,   Segmentation_training_loop, Sen1Dataset
from scripts.train.train_functions import  loss_chooser, wandb_initialization, job_type_selector, create_subset
from scripts.inference_functions import make_prediction_tiles, stitch_tiles, clean_checkpoint_keys, create_inference_csv, write_df_to_csv

start = time.time()

logging.basicConfig(
    level=logging.WARNING,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)
logging.getLogger('scripts.process.process_helpers').setLevel(logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_interrupt)

@click.command()
@click.option('--train', is_flag=True,  help="Train the model")
@click.option('--test', is_flag=True, help="Test the model")
@click.option('--inference', is_flag=True, help="Run inference on a copernicus S1 image")
@click.option('--config', is_flag=True, help='loading from config')

def main(train, test, inference, config):

    n = 0
    for i in train, test, inference:
        if i:
            n += 1
    if n > 1:
        raise ValueError("You can only specify one of --train, --test or --inference.")

    if  test:
        logger.info(' ARE YOU TESTING THE CORRECT CKPT? <<<')
    if train:
        job_type = "train" 
    elif test:
        job_type =  "test"
    elif inference:
        job_type = "inference"

    logger.info(f"train={train}, test={test}, inference={inference}\n, config = ")

    device = pick_device()                       
    logger.info(f" Using device: {device}")

    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)
    #......................................................
    # USER DEFINITIONS
    root_dir = Path('/Users/alexwebb/laptop_coding/floodai/INFERSAR/data/4final')
    logger.info(f"repo root: {root_dir}")
    env_file = root_dir / ".env"
    dataset_pth = root_dir / "dataset" 
    working_dir = root_dir / "training"
    ckpt_folder = root_dir / "checkpoints" / "ckpt_input"
    project = "mac_py_package"
    dataset_name = "sen1floods11"  # "sen1floods11" or "copernicus_floods"
    image_code = "0000"  # Placeholder for image code, can be set in config
    images_dir = dataset_pth/ 'S1HAnd'
    labels_dir = dataset_pth/ 'LabelHand'
    train_list = dataset_pth / "flood_train_data.csv"
    val_list = dataset_pth / "flood_valid_data.csv"
    test_list = dataset_pth / "flood_test_data.csv"
    run_name = "_"
    input_is_linear = False   # True for copernicus direct downloads, False for Sen1floods11
    subset_fraction = 1
    batch_size = 8
    max_epoch =15
    early_stop = True
    patience=3
    num_workers = 8
    WandB_online = False
    LOGSTEPS = 50
    PRETRAINED = True
    # inputs = ['vv', 'vh', 'mask']
    in_channels = 2
    DEVRUN = 0
    user_loss = 'bce_dice' #'smp_bce' # 'bce_dice' #'focal' # 'bce_dice' # focal'
    focal_alpha = 0.8
    focal_gamma = 8
    bce_weight = 0.35 # FOR BCE_DICE
    minmax_path = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/global_minmax_INPUT/global_minmax.json")
    output_filename = '_name' # OVERRIDDDEN IN CONFIG
    db_min = -30.0
    db_max = 0.0
    MAKE_TIFS = 0
    MAKE_DATAARRAY= 0
    MAKE_TILES = 0
    tile_size = 512
    stride = tile_size
#.......................................................
    if inference:
        input_is_linear = True  # For inference, we assume input is linear
        # INFERENCE
        threshold = 0.3
        predict_input = working_dir / "predict_input"
        working_dir = root_dir / 'predictions'
        file_list = working_dir / "predict_tile_list.csv"
        MAKE_TIFS = 0
        MAKE_DATAARRAY = 1
        MAKE_TILES = 1
        sensor = 'sensor'
        date = 'date'
        subset_fraction = 1
        batch_size = 1
        shuffle = False
        # TODO GET ABOVE VALUES FROM SOMEWHERE EG FILENAME
        stitched_image = working_dir / f'{sensor}_{image_code}_{date}_{tile_size}_{threshold}{output_filename}_WATER_AI.tif'
        if stitched_image.exists():
            logger.info(f"---overwriting existing file! : {stitched_image}")
        logger.info(f'config mode = {config}')

    if config:
        config_path = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/floodaiv2_config.yaml')
        logger.info(f"---config.name: {config.name}")
        logger.info(f"---Current config: {wandb.config}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        threshold = config["threshold"] 
        tile_size = config["tile_size"] 
        input_file = Path(config['input_file'])
        output_folder = Path(config['output_folder'])
        output_filename = Path(config['output_filename'])

    persistent_workers = num_workers > 0

    if env_file.exists():
        load_dotenv(env_file)
    else:
        logger.info("Warning: .env not found; using shell environment")

    # if user_loss != 'focal':
    #     focal_alpha = None
    #     focal_gamma = None
    # if user_loss != 'bce_dice':
    #     bce_weight = None

    #####       WAND INITIALISEATION + CONFIG       ###########
    wandb_config = {
        "name": run_name,
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "batch_size":batch_size,
        "user_loss": user_loss,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "bce_weight": bce_weight,
        "max_epoch": max_epoch,
    }
    wandb_logger = wandb_initialization(job_type, root_dir, project, dataset_name, run_name,train_list, val_list, test_list, wandb_config, WandB_online)
    config = wandb.config
    if user_loss == "focal":
        logger.info(f"---focal_alpha: {wandb.config.get('focal_alpha', 'Not Found')}")
        logger.info(f"---focal_gamma: {wandb.config.get('focal_gamma', 'Not Found')}")
        loss_desc = f"{user_loss}_{config.focal_alpha}_{config.focal_gamma}" 
    elif user_loss == "bce_dice":
        loss_desc = f"{user_loss}_{config.bce_weight}"
    else:
        loss_desc = user_loss

    run_name = f"{dataset_name}_{timestamp}_BS{config.batch_size}_s{config.subset_fraction}_{loss_desc}"  
    wandb.run.name = run_name
    # wandb.run.save()

    if is_sweep_run():
        logger.info(" IN SWEEP MODE <<<")
    #........................................................
    ckpt = next(ckpt_folder.rglob("*.ckpt"), None)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_folder}")
    
# ##################################################################
    #########    TRAIN / TEST - CREATE DATA LOADERS    #########
    if  train:
        file_list = train_list
    if test:
        file_list = test_list
    if train or test:
        save_tiles_path = dataset
        training_tiles = save_tiles_path
    if inference:
        save_tiles_path = working_dir /  f'{image_code}_tiles'
        tiles_dir = save_tiles_path
    metadata_path = save_tiles_path / 'tile_metadata_pth.json'

    # CREATE THE EXTRACTED FOLDER
    extracted = working_dir / f'{image_code}_extracted'
    if save_tiles_path.exists():
        # logger.info(f" Deleting existing tiles folder: {save_tiles_path}")
        # delete the folder and create a new one
        shutil.rmtree(save_tiles_path)
        save_tiles_path.mkdir(exist_ok=True, parents=True)

    logger.info(f' MAKE_TIFS = {MAKE_TIFS}, MAKE_DATAARRAY = {MAKE_DATAARRAY}, MAKE_TILES = {MAKE_TILES}   ')

    if MAKE_TIFS:
        pass
        # TODO MAKE SURE EXTENSION = TIF NOT TOFF
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

    if MAKE_DATAARRAY:
        create_event_datacube_copernicus(predict_input, image_code)
        cube = next(predict_input.rglob("*.nc"), None)
        if cube is None:
            logger.info(f"---No data cube found in {predict_input.name}")
            return  

    if MAKE_TILES:
        # TILE DATACUBE
        tiles, metadata = tile_datacube_rxr_inf(cube, save_tiles_path, tile_size=tile_size, stride=stride, percent_non_flood=0, inference = inference) 

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"---Saved metadata to {metadata_path}")
        logger.info(f"{len(tiles)} tiles saved to {save_tiles_path}")
        logger.info(f"{len(metadata)} metadata saved to {save_tiles_path}")
    # metadata = Path(save_tiles_path) / 'tile_metadata_pth.json'
        inference_list_dataframe = create_inference_csv(metadata)
        write_df_to_csv(inference_list_dataframe, file_list)
    
        ########     INITIALISE THE MODEL     #########
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED).to(device)
    # LOAD THE CHECKPOINT
    checkpoint = torch.load(ckpt)

    # EXTRACT THE MODEL STATE DICT
    cleaned_state_dict = clean_checkpoint_keys(checkpoint["state_dict"])

    # LOAD THE MODEL STATE DICT
    model.load_state_dict(cleaned_state_dict)

    logger.info(" Creating data loaders")
    logger.info(f"---tileS_dir: {tiles_dir}")
    logger.info(f"---file_list: {file_list}")
    logger.info(f'---input_is_linear: {input_is_linear}')

# ////////////////////////////////////////////////////////////

    # TRAIN + INFERENCE : CREATE THE  DATALOADER
    dataset = Sen1Dataset(
        job_type = job_type,
        tiles_dir = tiles_dir,
        images_dir = images_dir,
        labels_dir = labels_dir,
        csv_path = file_list,
        image_code=image_code,
        input_is_linear=input_is_linear,
        db_min=db_min,
        db_max=db_max)
    
    subset_indices = random.sample(range(len(dataset)), int(subset_fraction * len(dataset)))
    subset = Subset(dataset, subset_indices)
        # MAKE DATALOADER FROM DATASET

    dataloader = DataLoader( subset,
        batch_size = batch_size,  # Batch size of 1 for inference 
        num_workers = 4,
        persistent_workers = True,  # Keep workers alive for faster loading  
  # Adjust based on your system
        shuffle = job_type in ("train", "test") ,)

    if train or test:
        ########.     CHOOE LOSS FUNCTION     #########
        loss_fn = loss_chooser(user_loss, config.focal_alpha, config.focal_gamma, config.bce_weight)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,  # Stop if no improvement for 3 consecutive epochs
            mode="min",
    )
        ###########    SETUP TRAINING LOOP    #########
        ckpt_dir = root_dir / "checkpoints" / "ckpt_training"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
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
            training_loop = Segmentation_training_loop(model, loss_fn, stitched_image, user_loss)
            trainer.fit(training_loop, train_dataloaders=train_dl, val_dataloaders=val_dl)

        elif test:
            logger.info(f" Starting testing with checkpoint: {ckpt}")
            training_loop = Segmentation_training_loop.load_from_checkpoint(
                ckpt, model=model, loss_fn=loss_fn, stitched_image=stitched_image
            )
            trainer.test(model=training_loop, dataloaders=test_dl)


    elif inference:

        # DEFINE THE PREDICTION TILES FOLDER
        predictions_folder = save_tiles_path.parent / f'{save_tiles_path.stem}_predictions'
        # DELETE THE PREDICTION FOLDER IF IT EXISTS
        if predictions_folder.exists():
            logger.info(f"--- Deleting existing predictions folder: {predictions_folder}")
            shutil.rmtree(predictions_folder)
        predictions_folder.mkdir(exist_ok=True)

        #  # MAKE PREDICTION TILES
        with torch.no_grad():
            for imgs, valids, fnames in tqdm(dataloader, desc="Predict"):
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
        input_image = next(extracted.rglob("*.tif"), None) 
        if ('vv' in input_image.name.lower() or 'vh' in input_image.name.lower()) and input_image.suffix.lower() == '.tif':
            logger.info(f"---input_image: {input_image}")
            logger.info(f"---predictions_folder: {predictions_folder}")
            logger.info(f'---input_folder: {input_folder}')
            stitch_tiles(metadata, predictions_folder, stitched_image, input_image)  

    
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