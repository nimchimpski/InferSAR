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

# .............................................................
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from scripts.inference_helpers import make_prediction_tiles, stitch_tiles, clean_checkpoint_keys

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

    if inference:
            logger.info(f'config mode = {config}')

    if test and train:
        raise ValueError("You can only specify one of --train or --test.")
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
    repo_root = Path(__file__).resolve().parents[1]
    logger.info(f"repo root: {repo_root}")
    env_file = repo_root / ".env"
    dataset_path = repo_root / "data" / "4final" / "train_input" 
    if test:
        dataset_path = repo_root / "data" / "4final" / "test_input"

    ckpt_folder = repo_root / "checkpoints" / "ckpt_input"
    stitched_image = repo_root / "results"

    project = "mac_py_package"
    dataset_name = "sen1floods11"  # "sen1floods11" or "copernicus_floods"
    image_code = "code"  # Placeholder for image code, can be set in config
    train_list = dataset_path / "flood_train_data.csv"
    val_list = dataset_path / "flood_valid_data.csv"
    test_list = dataset_path / "flood_test_data.csv"
    run_name = "_"
    mode = "train"
    input_is_linear = False   # True for copernicus direct downloads, False for Sen1floods11
    subset_fraction = 1
    bs = 8
    max_epoch =15
    early_stop = True
    patience=3
    num_workers = 8
    WandB_online = True
    LOGSTEPS = 50
    PRETRAINED = True
    # inputs = ['vv', 'vh', 'mask']
    in_channels = 2
    DEVRUN = 0
    user_loss = 'bce_dice' #'smp_bce' # 'bce_dice' #'focal' # 'bce_dice' # focal'
    focal_alpha = 0.8
    focal_gamma = 8
    bce_weight = 0.35 # FOR BCE_DICE
    # INFERENCE
    threshold =  0.5 # OVERRIDDDEN IN CONFIG
    tile_size = 256 # OVERRIDDDEN IN CONFIG

    predict_input = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/data/4final/predict_input")
    minmax_path = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/global_minmax_INPUT/global_minmax.json")
    
    output_folder = predict_input # OVERRIDDDEN IN CONFIG
    output_filename = '_name' # OVERRIDDDEN IN CONFIG
    db_min = -30.0
    db_max = 0.0
    MAKE_TIFS = 0
    MAKE_DATAARRAY= 0
    MAKE_TILES = 0
    stride = tile_size
    #.......................................................

    if config:
        config_path = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR/configs/floodaiv2_config.yaml')
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

    if user_loss != 'focal':
        focal_alpha = None
        focal_gamma = None
    if user_loss != 'bce_dice':
        bce_weight = None

    #####       WAND INITIALISEATION + CONFIG       ###########
    wandb_config = {
        "name": run_name,
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "batch_size": bs,
        "user_loss": user_loss,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "bce_weight": bce_weight,
        "max_epoch": max_epoch,
    }
    wandb_logger = wandb_initialization(job_type, repo_root, project, dataset_name, run_name,train_list, val_list, test_list, wandb_config, WandB_online)
    config = wandb.config
    logger.info(f"---Current config: {wandb.config}")
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
    logger.info(f"---config.name: {config.name}")
    if is_sweep_run():
        logger.info(" IN SWEEP MODE <<<")
    #........................................................

    ckpt = next(ckpt_folder.rglob("*.ckpt"), None)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_folder}")

    #########    TRAIN / TEST - CREATE DATA LOADERS    #########

    if  train:
        logger.info(" Creating data loaders")
        train_dl = create_subset(mode, train_list, dataset_path, 'train', subset_fraction, bs, num_workers, persistent_workers, input_is_linear)
        val_dl = create_subset(mode, val_list, dataset_path, 'val', subset_fraction, bs, num_workers, persistent_workers, input_is_linear)

    if test:
        test_dl = create_subset(mode, test_list, dataset_path, 'test', subset_fraction, bs, num_workers, persistent_workers, input_is_linear)

        
    ########     INITIALISE THE MODEL     #########
    model = UnetModel(encoder_name='resnet34', in_channels=in_channels, classes=1, pretrained=PRETRAINED).to(device)

    # CREATE THE EXTRACTED FOLDER
    extracted = predict_input / f'{image_code}_extracted'
    save_tiles_path = predict_input /  f'{image_code}_tiles'
    if save_tiles_path.exists():
        # logger.info(f" Deleting existing tiles folder: {save_tiles_path}")
        # delete the folder and create a new one
        shutil.rmtree(save_tiles_path)
        save_tiles_path.mkdir(exist_ok=True, parents=True)

#  #+++++++++++++++++ INFERENCE - SETUP ++++++++++
    if inference:
        MAKE_TIFS = 0
        MAKE_DATAARRAY = 1
        MAKE_TILES = 1
        sensor = 'sensor'
        date = 'date'
        # TODO GET ABOVE VALUES FROM SOMEWHERE EG FILENAME
        stitched_image = output_folder / f'{sensor}_{image_code}_{date}_{tile_size}_{threshold}{output_filename}_WATER_AI.tif'
        logger.info(f'output filename: {stitched_image.name}')
        if stitched_image.exists():
            logger.info(f"---overwriting existing file! : {stitched_image}")

    logger.info(f' MAKE_TIFS = {MAKE_TIFS}, MAKE_DATAARRAY = {MAKE_DATAARRAY}, MAKE_TILES = {MAKE_TILES}   ')
    if MAKE_TIFS:
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

    if MAKE_TILES:
        # TILE DATACUBE
        tiles, metadata = tile_datacube_rxr_inf(cube, save_tiles_path, tile_size=tile_size, stride=stride, norm_func=norm_func, stats=stats, percent_non_flood=0, inference=True) 
    # logger.info(f"{len(tiles)} tiles saved to {save_tiles_path}")
    # logger.info(f"{len(metadata)} metadata saved to {save_tiles_path}")
    # metadata = Path(save_tiles_path) / 'tile_metadata.json'
        
    # LOAD THE CHECKPOINT
    checkpoint = torch.load(ckpt)

    # EXTRACT THE MODEL STATE DICT
    cleaned_state_dict = clean_checkpoint_keys(checkpoint["state_dict"])

    # LOAD THE MODEL STATE DICT
    model.load_state_dict(cleaned_state_dict)

    bs = 1,
    shuffle = False,
    input_is_linear = True

    # CREATE THE INFERENCE DATALOADER
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

    if train or test:
        ########.     CHOOE LOSS FUNCTION     #########
        loss_fn = loss_chooser(user_loss, config.focal_alpha, config.focal_gamma, config.bce_weight)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,  # Stop if no improvement for 3 consecutive epochs
            mode="min",
    )
        ###########    SETUP TRAINING LOOP    #########
        ckpt_dir = repo_root / "checkpoints" / "ckpt_training"
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