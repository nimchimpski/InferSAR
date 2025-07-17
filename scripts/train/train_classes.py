import torchvision.models as models

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
import csv
import logging
from typing import Optional, Tuple, Union, List

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
# from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc

import torch
import torch.nn as nn
import rasterio
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import wandb
import io
import random
import logging
from scripts.train.train_helpers import is_sweep_run
from scripts.train.train_functions import plot_auc_pr 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodDataset_from_multiband(Dataset):
    def __init__(self, tile_list, tile_root, stage='train', inputs=None):
        with open(tile_list, 'r') as _in:
            sample_list = _in.readlines()
        self.sample_list = [t[:-1] for t in sample_list]
        
        if stage == 'train':
            self.tile_root = Path(tile_root, 'train')
        elif stage == 'test':
            self.tile_root = Path(tile_root, 'test')
        elif stage == 'val':
            self.tile_root = Path(tile_root, 'val')
        self.inputs = inputs

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.sample_list)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        # logger.info(f'+++++++++++++++++++ get item')

        filename = self.sample_list[idx]
        if  filename.endswith('.json'):
            raise ValueError(f"Unexpected filename: {filename}")
        file_path = Path(self.tile_root, filename)
        # Use rasterio to inspect layer descriptions and dynamically select layers
        try:
            with rasterio.open(file_path) as src:
                layer_descriptions = src.descriptions  # Get layer descriptions
                # logger.info(f"Layer Descriptions: {layer_descriptions}")  # Debugging

                # Ensure descriptions are present
                if not layer_descriptions:
                    raise ValueError(f"No layer descriptions found in {file_path}")

                # Dynamically select layers based on their descriptions
                hh_index = layer_descriptions.index('hh')  # Index of 'hh' layer
                mask_index = layer_descriptions.index('mask')  # Index of 'mask' layer

                # Read only the required layers
                hh = src.read(hh_index + 1)  # Rasterio uses 1-based indexing
                mask = src.read(mask_index + 1)
        except Exception as e:
            logger.info(f"Error reading file {file_path}: {e}")
            raise
        # tile = tiff.imread(Path(self.tile_root, filename))

        # logger.info tile info and shape
        # logger.info(f"---Tile shape b4 permute: {tile.shape}")

        # Transpose to (C, H, W)
        # tile = torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1)  # Shape: (2, 256, 256)

        # logger.info(f"---Tile shape after permute: {tile.shape}")

        # Ensure the tile has 2 channels
        # assert tile.shape[0] == 2, f"Unexpected number of channels: {tile.shape[0]}"    
        # Select channels based on `inputs` list position
        # input_idx = list(range(len(self.inputs)))
        # model_input = tile[input_idx, :, : ]  # auto select the channels
        # model_input = tile[:1, :, : ].clone()  # auto select the channels
        model_input = torch.tensor(hh, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # logger.info(f"---model_input shape: {model_input.shape}")  # Should logger.info torch.Size([batch_size, 2, 256, 256])  

        # model_input = model_input.cuda() #???

        # EXTRACT MASK TO BINARY
        # logger.info('---mask index:', self.inputs.index('mask'))
        # mask = tile[ self.inputs.index('mask'),:,: ]
        # mask = tile[ 1,:,: ].clone()
        # mask = mask.unsqueeze(0)  # Add a channel dimension
        # CONVERT TO TENSOR
        mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)  # Add a channel dimension
        mask = (mask > 0.5).float()

        # logger.info(f"---mask shape: {mask.shape}")  # Should logger.info torch.Size([batch_size, 1, 256, 256])

        assert mask.shape == (1, 256, 256), f"Unexpected mask shape: {mask.shape}"

        # Debugging: Check unique values in the mask
        # logger.info("---Unique values in mask:", torch.unique(mask))

        # Combine HH and MASK into a single input tensor
        # input_tensor = torch.stack([model_input, mask], dim=0)  # Shape: (2, 256, 256)
        return [model_input, mask]
        # return model_input.float(), val_mask.float()

class FloodDataset(Dataset):
    def __init__(self, tile_list, tile_root, stage='train', inputs=None):
        with open(tile_list, 'r') as _in:
            sample_list = _in.readlines()
        self.sample_list = [t[:-1] for t in sample_list]
        
        if stage == 'train':
            self.tile_root = Path(tile_root, 'train')
        elif stage == 'test':
            self.tile_root = Path(tile_root, 'test')
        elif stage == 'val':
            self.tile_root = Path(tile_root, 'val')
        self.inputs = inputs

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.sample_list)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        # logger.info(f'+++++++++++++++++++ get item')

        filename = self.sample_list[idx]
        if  filename.endswith('.json'):
            raise ValueError(f"Unexpected filename: {filename}")
        file_path = Path(self.tile_root, filename)
        # Use rasterio to inspect layer descriptions and dynamically select layers
        try:
            with rasterio.open(file_path) as src:
                layer_descriptions = src.descriptions  # Get layer descriptions
                # logger.info(f"Layer Descriptions: {layer_descriptions}")  # Debugging

                # Ensure descriptions are present
                if not layer_descriptions:
                    raise ValueError(f"No layer descriptions found in {file_path}")

                # Dynamically select layers based on their descriptions
                hh_index = layer_descriptions.index('hh')  # Index of 'hh' layer
                mask_index = layer_descriptions.index('mask')  # Index of 'mask' layer

                # Read only the required layers
                hh = src.read(hh_index + 1)  # Rasterio uses 1-based indexing
                mask = src.read(mask_index + 1)
        except Exception as e:
            logger.info(f"Error reading file {file_path}: {e}")
            raise
        # tile = tiff.imread(Path(self.tile_root, filename))

        # logger.info tile info and shape
        # logger.info(f"---Tile shape b4 permute: {tile.shape}")

        # Transpose to (C, H, W)
        # tile = torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1)  # Shape: (2, 256, 256)

        # logger.info(f"---Tile shape after permute: {tile.shape}")

        # Ensure the tile has 2 channels
        # assert tile.shape[0] == 2, f"Unexpected number of channels: {tile.shape[0]}"    
        # Select channels based on `inputs` list position
        # input_idx = list(range(len(self.inputs)))
        # model_input = tile[input_idx, :, : ]  # auto select the channels
        # model_input = tile[:1, :, : ].clone()  # auto select the channels
        model_input = torch.tensor(hh, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # logger.info(f"---model_input shape: {model_input.shape}")  # Should logger.info torch.Size([batch_size, 2, 256, 256])  


        # CONVERT TO TENSOR
        mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)  # Add a channel dimension
        mask = (mask > 0.5).float()

        # logger.info(f"---mask shape: {mask.shape}")  # Should logger.info torch.Size([batch_size, 1, 256, 256])

        assert mask.shape == (1, 256, 256), f"Unexpected mask shape: {mask.shape}"

        # Debugging: Check unique values in the mask
        # logger.info("---Unique values in mask:", torch.unique(mask))

        # Combine HH and MASK into a single input tensor
        # input_tensor = torch.stack([model_input, mask], dim=0)  # Shape: (2, 256, 256)
        return [model_input, mask]
        # return model_input.float(), val_mask.float()


class Sen1Dataset(Dataset):
    def __init__(self, job_type: str, input_folder: Path, csv_path: Path,image_code: str,  input_is_linear: bool, db_min: float = -30.0, db_max: float = 0.0, ):
        """
        csv_path: Path to one of the split CSVs (train.csv / val.csv / test.csv/ pred.csv)
        """
        self.job_type = job_type
        self.input_folder = input_folder
        self.image_code= image_code
        self.db_min = db_min
        self.db_max = db_max
        self.input_is_linear = input_is_linear
        valid_thresh = 0.8  # Minimum fraction of valid pixels to keep a tile

        self.img_paths  = []
        self.mask_paths = []
        orig_imgs = []
        orig_masks = []
        
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            # assume the columns are named exactly "image" and "mask"
            #  MAKE LISTS OF 
            next(reader, None)  # skip header row
            for row in reader:
                img_rel = row[0]
                orig_imgs.append(self.input_folder / f'{image_code}_tiles' / img_rel)
            
            if job_type in  ('train', 'val'):
                mask_rel = row[1]
                orig_masks.append(self.input_folder / f'{image_code}_tiles' / mask_rel)
                for img_pth, mask_pth in zip(orig_imgs, orig_masks):
                    with rasterio.open(img_pth) as src:
                        valid = src.dataset_mask().astype(bool)
                        # dataset_mask()==1 where data is valid, 0 where nodata or outside the swath
                        # if np.sum(valid) >= valid_thresh:
                        #     self.img_paths.append(img_pth)
                        #     self.mask_paths.append(mask_pth)
                        

                        assert len(self.img_paths) == len(self.mask_paths) 
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(
                f"Split file {csv_path} has {len(orig_imgs)} images "
                f"but {len(orig_masks)} masks."
            )
        if len(orig_imgs) != len(orig_masks):
            logger.info(f"some images have no valid pixels, skipping them")


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1) Load VV & VH
        img_pth = self.img_paths[idx]
        with rasterio.open(self.input_folder/ f'{self.image_code}_tiles' /img_pth) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
            valid = src.dataset_mask().astype(bool)
            # dataset_mask()==1 where data is valid, 0 where nodata or outside the swath
    
        # blank out invalid pixels:
        vv[~valid] = np.nan
        vh[~valid] = np.nan

        #  Clip & normalize
        for arr in (vv, vh):

            #  CHECK IF LINEAR OR DB
            if self.input_is_linear:
                # logging.info(f"Input is linear")
                np.clip(arr, 1e-6, None, out=arr)  # floor zeros
                np.log10(arr, out=arr)
                arr *= 10.0  # convert to dB
            # elif not self.input_is_linear:
            #     logging.info(f"Input is in dB")
            arr = np.nan_to_num(arr,
                               copy=False,           # modify vv in-place
                               nan=self.db_min,
                               posinf=self.db_max,
                               neginf=self.db_min)
            #  global clip
            np.clip(arr, self.db_min, self.db_max, out=arr)
            #  scale to [0,1]
            arr -= self.db_min
            arr /= (self.db_max - self.db_min)

        img_tensor = np.stack([vv, vh], axis=0)  # shape [2,H,W]

        if self.job_type in('train', 'val'):
            msk_pth = self.input_folder/ f'{self.image_code}_masks' / self.mask_paths[idx]
            with rasterio.open(msk_pth) as src:
                raw = src.read(1).astype(np.int64)
                valid_mask = torch.from_numpy(raw != -1).unsqueeze(0)  # valid pixels are 1, invalid are 0
                mask = np.where(raw ==  -1, 255, raw)
                mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # Add a channel dimension
            return (img_tensor, mask, valid_mask, img_pth.name)

        valid_mask = torch.from_numpy((raw != -1).astype(np.float32))
        return (img_tensor, valid_mask.unsqueeze(0), img_pth.name) 
        
import csv
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class Sen1Dataset(Dataset):
    """
    A single Dataset for train/val/infer, driven by a CSV.

    CSV format:
      train/val -> rows of [image_filename, mask_filename]
      infer     -> rows of [image_filename, <ignored>]

    __getitem__ returns:
      train/val: (img [2,H,W], mask [1,H,W], valid [1,H,W], fname)
      infer:     (img [2,H,W], valid [1,H,W], fname)
    """
    def __init__(
        self,
        job_type:       str,        # "train","val" or "infer"
        input_folder:   Path,       # root of your S1Hand/… folders
        csv_path:       Path,
        image_code:     str,        # e.g. "myevent"
        input_is_linear: bool,
        db_min:         float = -30.0,
        db_max:         float =   0.0,
    ):
        assert job_type in ("train","val","inference")
        self.job_type        = job_type
        self.input_folder    = input_folder
        self.image_code      = image_code
        self.input_is_linear = input_is_linear
        self.db_min          = db_min
        self.db_max          = db_max

        # will hold full Paths to files
        self.img_paths:  List[Path] = []
        self.mask_paths: List[Path] = []
        self.fnames:     List[str]  = []

        tile_dir = input_folder / f"{image_code}_tiles"
        mask_dir = input_folder / f"{image_code}_masks"

        # 1) parse the CSV
        with csv_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header row
            for row in reader:
                img_name = row[0]
                self.img_paths.append(tile_dir / img_name)
                self.fnames.append(img_name)

                # only for train/val do we need the second column
                if job_type in ("train","val"):
                    mask_name = row[1]
                    self.mask_paths.append(mask_dir / mask_name)

        # sanity check
        if job_type in ("train","val"):
            assert len(self.img_paths) == len(self.mask_paths), (
                f"csv has {len(self.img_paths)} images but "
                f"{len(self.mask_paths)} masks"
            )


    def __len__(self) -> int:
        return len(self.img_paths)


    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        Tuple[torch.Tensor, torch.Tensor, str]
    ]:
        # --- load VV & VH + valid mask from the tile file ---
        img_path = self.img_paths[idx]
        logger.info(f"---Loading image from {img_path}")

        with rasterio.open(img_path) as src:
            vv_arr  = src.read(1).astype(np.float32)
            vh_arr  = src.read(2).astype(np.float32)
            valid_np = src.dataset_mask().astype(bool)

        # blank invalid pixels
        vv_arr[~valid_np] = np.nan
        vh_arr[~valid_np] = np.nan

        # --- log→clip→minmax→[0,1] normalize each band ---
        for arr in (vv_arr, vh_arr):
            if self.input_is_linear:
                np.clip(arr, 1e-6, None, out=arr)
                np.log10(arr, out=arr)
                arr *= 10.0
            np.nan_to_num(
                arr,
                copy=False,
                nan=self.db_min,
                posinf=self.db_max,
                neginf=self.db_min,
            )
            np.clip(arr, self.db_min, self.db_max, out=arr)
            arr -= self.db_min
            arr /= (self.db_max - self.db_min)

        # stack & convert to tensor
        img_tensor = torch.from_numpy(np.stack([vv_arr, vh_arr], axis=0))

        # --- now branch by job_type ---
        if self.job_type in ("train","val"):
            # load the mask, build mask & valid_mask from raw values
            mask_path = self.mask_paths[idx]
            with rasterio.open(mask_path) as src:
                raw = src.read(1).astype(np.int64)

            # valid_mask = 1 for pixels ≠ –1
            valid_mask = torch.from_numpy((raw != -1).astype(np.float32))
            # flood mask = 1 for pixels == 1
            mask       = torch.from_numpy((raw == 1).astype(np.float32))

            # add channel dims
            return (
                img_tensor,
                mask.unsqueeze(0),
                valid_mask.unsqueeze(0),
                self.fnames[idx]
            )

        # inference mode: no mask file, just return valid mask
        valid_mask = torch.from_numpy(valid_np.astype(np.float32))
        return (
            img_tensor,
            valid_mask.unsqueeze(0),
            self.fnames[idx]
        )


class Segmentation_training_loop(pl.LightningModule):

    def __init__(self, model, loss_fn, save_path, user_loss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.save_hyperparameters(ignore = ['model', 'loss_fn', 'save_path'])   
        # Container to store validation results
        self.validation_outputs = []
        self.dynamic_weights = False
        self.user_loss = user_loss
        self.test_images = []

    def forward(self, x):
        # logger.info(f"---Input device in forward: {x.device}")
        try:
            # for name, param in self.model.named_parameters():
            #     logger.info(f"Parameter {name} is on device: {param.device}")  # Debug each parameter

            x = self.model(x)  # Pass through the model
            # logger.info(f"---Output device in forward: {x.device}")
        except Exception as e:
            logger.info(f"Error during forward pass: {e}")
            raise
        return x

    def training_step(self, batch, batch_idx):
        # logger.info(f'+++++++++++++++++++   training step') 
        job_type = 'train'

        images, masks, valids, fnames = batch
        # DEBUGGING
        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.info(f"TRAIN STEP - Batch {batch_idx} - Input contains NaN or Inf")
            logger.info(f"Mean: {images.mean()}, Std: {images.std()}, Min: {images.min()}, Max: {images.max()}")
            raise ValueError(f"Input contains NaN or Inf at batch {batch_idx}")
        images, masks, valids = images.to(self.device), masks.to(self.device), valids.to(self.device)
        logits = self(images)
        loss_per_pixel = self.loss_fn(logits, masks)  
        loss_per_pixel = (loss_per_pixel * valids.float()).sum() / valids.sum()  # Apply valid mask to loss
        # ONLY APPLIES DYNAMIC WEIGHTS TO BCE LOSS
        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        # logger.info(f'---used dynamic weights = {dynamic_weights}')
        assert logits.device == masks.device

        lr = self._get_current_lr()
        # self.log('lr', lr,  on_step=True, on_epoch=True, prog_bar=True, logger=True)

        _, _, _, _= self.metrics_maker(logits, masks, valids, job_type, loss, self.user_loss, lr)
        return loss

    def validation_step(self, batch, batch_idx):
        # logger.info(f'+++++++++++++    validation step')
        job_type = 'val'
        images, masks, valids, fnames = batch

        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.info(f"VAl STEP - Batch {batch_idx} - Input contains NaN or Inf")
            logger.info(f"Mean: {images.mean()}, Std: {images.std()}, Min: {images.min()}, Max: {images.max()}")
            raise ValueError(f"Input contains NaN or Inf at batch {batch_idx}")

        images, masks, valids  = images.to(self.device), masks.to(self.device), valids.to(self.device)
        logits = self(images)

        # logger.info(f"---Validation Step {batch_idx}: logits shape={logits.shape}, masks shape={masks.shape}")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.info(f"---Batch {batch_idx} - Logits contain NaN or Inf")
            logger.info(f"---Logits Stats - Mean: {logits.mean()}, Std: {logits.std()}, Min: {logits.min()}, Max: {logits.max()}")
            raise ValueError(f"Logits contain NaN or Inf at batch {batch_idx}")
        
        self.validation_outputs.append({'logits': logits, 'masks': masks})  # Store outputs
        loss_per_pixel = self.loss_fn(logits, masks)
        loss_per_pixel = (loss_per_pixel * valids).sum() / valids.sum()  # Apply valid mask to loss

        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        assert logits.device == masks.device
        # Check if this is the last batch and save visualization
        val_dataloader = self.trainer.val_dataloaders
        total_batches = len(val_dataloader) 
        # logger.info(f"---Total batches: {total_batches}")

        images = images.squeeze(1)  # Remove the channel dimension if it's 1

        #   DEBUGGING
        # logger.info(f"---Validation Step {batch_idx}: images shape={images.shape}, logits shape={logits.shape}, masks shape={masks.shape}")
        if self.current_epoch == self.trainer.max_epochs - 1 and batch_idx < 3:
        # This is the last epoch
            # logger.info(f'---used dynamic weights = {dynamic_weights}')
            if not is_sweep_run():
                self.log_combined_visualization(images, logits, masks, valids, fnames, self.user_loss)

            # Save images if this is the best-performing model
            # if loss == self.trainer.checkpoint_callback.best_model_score and batch_idx < 2:
            #     self.log_combined_visualization(images, logits, masks, self.user_loss)

        ioumean, precisionmean , recall, f1mean  = self.metrics_maker(logits, masks, valids, job_type,  loss, self.user_loss )

        return {"loss": loss, "precision": precisionmean,"recall": recall, "iou": ioumean,  "f1": f1mean, 'logits': logits, 'labels': masks}   

    def test_step(self, batch, batch_idx):
        # logger.info(f'+++++++++++++    test step')
        job_type = 'test'
        images, masks, valids, fnames = batch

        self.test_images.append(images.cpu())

        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.info(f"TEST STEP -Batch {batch_idx} - Input contains NaN or Inf")
            raise ValueError(f"Input contains NaN or Inf at batch {batch_idx}")
        # logger.info(f"Validation Image Stats - Batch {batch_idx}")
        # logger.info(f"Mean: {images.mean()}, Std: {images.std()}, Min: {images.min()}, Max: {images.max()}")

        images, masks, valids = images.to(self.device), masks.to(self.device), valids.to(self.device)
        logits = self(images)

        # Debug tensor stats
        # logger.info(f"---Batch {batch_idx}: images.min={images.min()}, images.max={images.max()}")
        # logger.info(f"---Batch {batch_idx}: logits.min={logits.min()}, logits.max={logits.max()}")
        # logger.info(f"---Batch {batch_idx}: masks.min={masks.min()}, masks.max={masks.max()}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.info(f"---Batch {batch_idx} - Logits contain NaN or Inf")
            logger.info(f"---Logits Stats - Mean: {logits.mean()}, Std: {logits.std()}, Min: {logits.min()}, Max: {logits.max()}")
            raise ValueError(f"Logits contain NaN or Inf at batch {batch_idx}")

        self.test_outputs.append({'logits': logits, 'masks': masks})  # Store outputs
        loss_per_pixel = self.loss_fn(logits, masks)
        loss_per_pixel = (loss_per_pixel * valids).sum() / valids.sum()  # Apply valid mask to loss

        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        assert logits.device == masks.device

        # logger.info(f"---weighted_loss device: {weighted_loss.device}")
        # preds = (torch.sigmoid(logits) > 0.5).int() # BCE, 
        # Determine if this is the last batch
        test_dataloader = self.trainer.test_dataloaders # First DataLoader
        total_batches = len(test_dataloader)
        # logger.info(f"---Total batches: {total_batches}")
        # if batch_idx == 1:
        #     # logger.info('---batch_idx:', batch_idx)
        #     # logger.info(f"---Saving test outputs for batch {batch_idx}")
        self.log_combined_visualization(images, logits, masks, valids, self.user_loss)
        #     # JUST logger.info ON FIRST BATCH
        #     if batch_idx == 1:
        #         logger.info(f'---used dynamic weights = {dynamic_weights}')


        
            # CALCULATE METRICS
        ioumean, precisionmean, recallmean, f1mean = self.metrics_maker(logits, masks, job_type, loss, self.user_loss)

        return { "loss": loss,  "precision": precisionmean, "recall": recallmean, "iou": ioumean, "f1": f1mean, 'logits': logits, 'labels': masks}
    
    def _get_current_lr(self):
        # logger.info(f'+++++++++++++    get current lr')
        lr = [x["lr"] for x in self.optimizers().param_groups]
        return lr[0]
    
    def compute_dynamic_weights(self, mask):
        
        # logger.info(f'+++++++++++++    compute dynamic weights')
        assert torch.unique(mask).tolist() in [[0], [1], [0, 1]], f"Unexpected mask values: {torch.unique(mask)}"

        flood_pixels = (mask == 1).sum().float()
        non_flood_pixels = (mask == 0).sum().float()
        total_pixels = flood_pixels + non_flood_pixels

        if total_pixels > 0:
            flood_weight = non_flood_pixels / total_pixels
            non_flood_weight = flood_pixels / total_pixels
        else:
            flood_weight = 1.0
            non_flood_weight = 1.0

        weights = torch.ones_like(mask).float()
        weights[mask == 0] = non_flood_weight
        weights[mask == 1] = flood_weight

        # weights = weights.to('cuda')

        return weights
    
    def dynamic_weight_chooser(self, masks, loss, user_loss):
        # logger.info(f'+++++++++++++    dynamic weight chooser')
        # logger.info(f'---loss_fn: {self.loss_fn}')
        if user_loss in ['smp_bce']:

            # logger.info(f'*********computing dynamic weights')
            weights = self.compute_dynamic_weights(masks)
            # weights = weights.to('cuda')
            assert masks.device == weights.device
            dynamic_bool = True
            return  (loss * weights).mean(), dynamic_bool
        else:
            # logger.info(f'---no dynamic weights')
            dynamic_bool = False
            return loss.mean(), dynamic_bool
        

    def configure_optimizers(self):
        logger.info(f'+++++++++++++    configure optimizers')
        params = [x for x in self.model.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5) # gama= coefficient. bigger value means faster decay
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Return as a list of dictionaries with `scheduler` and `interval` specified
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",  # or "step" if you want to step every batch
                "frequency": 1
            }
        }
    
    def log_combined_visualization(self, images, logits, masks, valids, fnames, loss_name):
        """
        Visualizes input images, predictions, and ground truth masks side by side for a batch.
        """

        preds = (torch.sigmoid(logits) > 0.1).int()

        # logger.info(f'+++++++++++++    log combined visualization')
        # logger.info(f'---images shape: {images.shape[0]}') 
        assert images.ndim == 4, f"Expected images with 4 dimensions (B, C, H, W), got {images.shape}"
        assert preds.ndim == 4, f"Expected preds with 4 dimensions (B, C, H, W), got {preds.shape}"
        assert masks.ndim == 4, f"Expected masks with 4 dimensions (B, C, H, W), got {masks.shape}"

        plt.rcParams['axes.titlesize'] = 35
        max_samples = 20  # Maximum number of samples to visualize
        examples = []
        for i in range(min(images.shape[0], max_samples)):  # Loop through each sample in the batch
            # logger.info(f"---images.shape {images.shape}")
            # logger.info(f"---Sample {i}")
            cmap_cyan = ListedColormap(['black', 'cyan'])           
            # CONVERT TENSORS TO NUMPY
            image = images[i, 0].cpu().numpy()
            fname = fnames[i]
            valid = valids[i].squeeze().cpu().numpy() # Convert valid mask to numpy

            # combined = np.concatenate([image, pred, mask], axis=1)
            # plt.imshow(np.concatenate([image, pred, mask], axis=1), cmap="gray")
            # plt.title(f"Sample {i} | Input | Prediction | Ground Truth")
            # plt.show()

            epsilon = 1e-6
            vmin, vmax =  np.percentile(image, (2, 98))
            img_vis = ((image - vmin) / (vmax - vmin + epsilon)).clip(0.1) * 255  # Normalize input image to [0, 255]
            prob_vis = torch.sigmoid(logits[i]).squeeze().cpu().numpy().squeeze()
            prob_vis = (prob_vis * valid.astype(np.uint8))  # Apply valid mask to probabilities

            pred_vis = (prob_vis * valid.astype(np.uint8)* 255).astype(np.uint8)  # Scale prediction to [0, 255]
            
            mask_np = masks[i].squeeze().cpu().numpy().astype(np.uint8)  # 0,1
            mask_vis = mask_np * 255

            fig, axes = plt.subplots(1, 4, figsize=(16, 5)) # in inches
            axes[0].imshow(img_vis.astype(np.uint8),cmap='OrRd')
            axes[0].set_title('Input')

            # 2) Probability heat-map
            axes[1].imshow(prob_vis, cmap='inferno', vmin = 0, vmax = 1)
            axes[1].set_title('Prob')

            # 3) Thresholded prediction (white=flood)
            axes[2].imshow(pred_vis, cmap= cmap_cyan)
            axes[2].set_title('Pred >0.5')

            # 4) Ground truth mask (white=flood)
            axes[3].imshow(mask_vis, cmap='gray')
            axes[3].set_title('Label')

            # fig.subplots_adjust(bottom=0.2) # in %
  
            for ax in axes:
                ax.axis('off') # Turn off axis
            # plt.tight_layout()  
            examples.append(wandb.Image(fig, caption= f'{i} {fname}'))

        # Log to WandB and add title
        self.logger.experiment.log({"examples": examples} )  

    

    def on_validation_epoch_start(self):
        self.validation_outputs = []  # Reset the list for the new epoch

    def on_validation_epoch_end(self):
        """
        Compute AUC-PR only during the final validation epoch. takes the 
        """
        # if is_sweep_run():
        #     logger.info(f'---in sweep mode so skipping auc-pr calculation')
        #     return

        # Check if this is the final epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            logger.info(f"---Calculating AUC-PR for the final validation epoch: {self.current_epoch}")

            # Ensure validation_outputs has been populated
            if not self.validation_outputs:
                raise ValueError("---Validation outputs are empty. Check your validation_step implementation.")

            # Aggregate outputs from all validation batches
            all_logits = torch.cat([output['logits'] for output in self.validation_outputs], dim=0)
            all_labels = torch.cat([output['masks'] for output in self.validation_outputs], dim=0)

            if all_logits.numel() == 0 or all_labels.numel() == 0:
                raise ValueError("Validation outputs are empty. Ensure validation_step is properly implemented.")

            # Flatten logits and labels
            # logits_np = all_logits.detach().cpu().numpy().flatten()
            logits_np = torch.sigmoid(all_logits).detach().cpu().numpy().flatten()
            labels_np = all_labels.detach().cpu().numpy().flatten()

            # GET RID OF THE 255 INVALID CHANNEL
            labels_np = labels_np != 255
            if labels_np.sum() == 0:
                raise ValueError("Validation batch contains only ignore pixels.")

            # DEBUGGING
            # Check for NaN or Inf values in logits_np and labels_np
            if not np.isfinite(logits_np).all():
                raise ValueError("---logits_np contains NaN or Inf values.")
            if not np.isfinite(labels_np).all():
                raise ValueError("---labels_np contains NaN or Inf values.")
            # logger.info(f"---Logits: Min={logits_np.min()}, Max={logits_np.max()}, Mean={logits_np.mean()}")
            # logger.info(f"---Labels: Unique={np.unique(labels_np)}, Counts={np.bincount(labels_np.astype(int))}")

            # COUNT UNIQUE CLASSES
            unique_classes, class_counts = np.unique(labels_np, return_counts=True)
            # logger.info(f"---Unique classes: {unique_classes}, Counts: {class_counts}")
            # Raise an error if there's only one class
            if len(unique_classes) < 2:
                raise ValueError("---Precision-Recall curve requires at least two classes in the ground truth.")
            
            try:
                precision, recall, thresholds = precision_recall_curve(labels_np, logits_np)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
                best_index = f1_scores.argmax()
                best_threshold = thresholds[best_index]
                logger.info(f"---Best Threshold: {best_threshold}, F1-Score: {f1_scores[best_index]}")
                aucpr_plot = plot_auc_pr(recall, precision, thresholds, best_index, best_threshold)
                # aucpr_plot.show()
                self.logger.experiment.log({"Precision-Recall Curve": wandb.Image(aucpr_plot)})
                auc_pr = auc(recall, precision)

            except ValueError as e:
                logger.info(f"---AUC-PR calculation failed: {e}")
                auc_pr = 0.0  # Default value for invalid AUC-PR

            # Log the final AUC-PR
            self.log('val_auc_pr', auc_pr, prog_bar=True, logger=True)
        else:
        #     logger.info(f"---Skipping AUC-PR calculation for epoch: {self.current_epoch}")
            self.validation_outputs = []

    def on_test_epoch_start(self):
        self.test_outputs = []

    
    def on_test_epoch_end(self):
        # Ensure validation_outputs has been populated
        logger.info(f'+++++++++++++    on test epoch end')
        if not self.test_outputs:
            raise ValueError("---test outputs are empty. Check your test_step implementation.")
        # Aggregate outputs
        all_logits = torch.cat([output['logits'] for output in self.test_outputs], dim=0)
        all_masks = torch.cat([output['masks'] for output in self.test_outputs], dim=0)

        # Convert to NumPy arrays
        logits_np = torch.sigmoid(all_logits).cpu().numpy().flatten()
        masks_np = all_masks.cpu().numpy().flatten()

            # Debugging statistics
        logger.info(f"Logits Stats - Min: {logits_np.min()}, Max: {logits_np.max()}, Mean: {logits_np.mean()}")
        logger.info(f"Labels Unique: {np.unique(masks_np, return_counts=True)}")

        # Handle edge cases (e.g., single-class masks)
        if len(np.unique(masks_np)) < 2:
            logger.info("---Skipping AUC-PR calculation due to insufficient class variability.")
            return

        # Compute Precision-Recall and AUC
        precision, recall, thresholds = precision_recall_curve(masks_np, logits_np, pos_label=1)
        auc_pr = auc(recall, precision)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
        best_index = f1_scores.argmax()
        best_threshold = thresholds[best_index]
        logger.info(f"---Best Threshold: {best_threshold}, F1-Score: {f1_scores[best_index]}")
        aucpr_plot = plot_auc_pr(recall, precision, thresholds, best_index, best_threshold)
        self.logger.experiment.log({
        "Precision-Recall Curve": wandb.Image(aucpr_plot),
        "Best Threshold": best_threshold, 
        "Best F1-Score": f1_scores[best_index]})


        # Log AUC-PR for the test set
        self.log('auc_pr_test', auc_pr, prog_bar=True, logger=True)

    def metrics_maker(self, logits, masks, valid, job_type, loss, user_loss, lr=None):
  
        mthresh = 0.5
        probs = torch.sigmoid(logits) 

        preds = (probs > mthresh).int()
        # logger.info(f'---metric threshod={mthresh}')
        if valid.sum() == 0:
        # skip batch that contains only ignore pixels
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        preds = preds[valid].unsqueeze(1)  # Apply the mask to predictions
        masks = masks[valid].long().unsqueeze(1)  # Apply the mask to ground truth

        # Compute metrics
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn)
        precision = smp.metrics.precision(tp, fp, fn, tn)
        recall = smp.metrics.recall(tp, fp, fn, tn)
        f1 = smp.metrics.f1_score(tp, fp, fn, tn)

        # Averages
        ioumean = iou.mean()
        precisionmean = precision.mean()
        recallmean = recall.mean()
        f1mean = f1.mean()

        assert loss is not None, f"Loss is None for {job_type} job" 

        # Logging
        if job_type == 'train':
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train_lr', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif job_type == 'val':
            self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        if not job_type == 'train':
            self.log(f'iou_{job_type}', ioumean, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'precision_{job_type}', precisionmean, prog_bar=True, on_step=False, on_epoch=True )
            self.log(f'recall_{job_type}', recallmean, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'f1_{job_type}', f1mean, prog_bar=True, on_step=False, on_epoch=True )
            self.log(f'thresh_{job_type}', mthresh)
            # Precision-Recall Curve Logging (Binary Classification)
            # wandb_pr = wandb.plot.pr_curve(masks, probs, title=f"Precision-Recall Curve {job_type}")
            # self.log({"pr": wandb_pr})
     

    
        return ioumean, precisionmean, recallmean, f1mean
    


'''
def log_combined_visualization_plt(self, preds, mask):
        logger.info(f'+++++++++++++    log combined visualization plt')
        logger.info(f"Global Step: {self.global_step}")
        # Convert tensors to numpy
        preds = preds.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        # Show input image - stacked in reverse order!
        ax.imshow(mask, cmap="Blues", alpha=1)  # Overlay ground truth
        ax.imshow(preds, cmap="Reds", alpha=0.5)  # Overlay predictions
        ax.axis("off")

        legend_elements = [
            Line2D([0], [0], color="red", lw=4, label="Predictions"),
            Line2D([0], [0], color="blue", lw=4, label="Ground Truth")
        ]

        ax.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=True)

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
    
        # Convert buffer to PIL image for WandB
        pil_image = Image.open(buf)

        pil_image.save(self.save_path / f"debug_visualization_{self.global_step}.png")  # Save the image locally
        # logger.info("Visualization saved locally as debug_visualization.png")

        # Log to WandB
        # Log to WandB with a unique key
        self.logger.experiment.log({
            f"Combined Visualization Step {self.global_step}": wandb.Image(
                pil_image,
                caption=f"Step {self.global_step} | Input with Prediction and Ground Truth Overlay"
            )
        })
    
        # Close buffer
        buf.close()
'''
# MODELS

class UnetModel(nn.Module):
    def __init__(self,encoder_name='resnet34', in_channels=2, classes=1, pretrained=True):
        super().__init__()
        self.model= smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels, 
            classes=classes,
            activation=None)
        
        # Fix the first convolutional layer
        self.model.encoder.conv1 = nn.Conv2d(
            in_channels=in_channels,     # Match your input channel count
            out_channels=64,   # Keep the same number of filters
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )

                # Add dropout to the decoder
        self.model.decoder.dropout = nn.Dropout(p=0.5)

        self.model.decoder.final_conv = nn.Conv2d(
        in_channels=1,  # Use the appropriate number of input channels from the decoder
        out_channels=1,  # Single output channel for binary segmentation
        kernel_size=1
)

        # if pretrained:
        #     checkpoint_dict= torch.load(f'/kaggle/working/Fold={index}_Model=mobilenet_v2.ckpt')['state_dict']
        #     self.model.load_state_dict(load_weights(checkpoint_dict))
    def forward(self,x):
        x= self.model(x)
        return x
 
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, classes=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, classes, kernel_size=3, padding=1)  # Output with 2 channels

        # Optional: Use an upsampling layer to match the output size with the ground truth
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        
        # Optional: Use upsampling if necessary to match the size
        x = self.upsample(x)
        return x

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=1, pretrained=True):
        super().__init__()
        # Load pretrained ResNet
        self.backbone = models.__dict__[encoder_name](pretrained=pretrained)
        
        # Modify the first convolution layer to accept the HH SAR band (in_channels=1)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer to output a single logit for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # Single logit output

    def forward(self, x):
        return self.backbone(x)

# ------------------- LOSS FUNCTION -------------------

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # ****ADDED****
        # Check if the input tensor `one_hot_gt` is in the correct format
        # If it's not, convert it to float32
        if one_hot_gt.dtype != torch.float32:
            one_hot_gt = one_hot_gt.float()

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt


        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
        
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        logger.info(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss
   
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# COMBOS
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * focal + (1 - self.alpha) * dice
    
class BoundaryDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        return self.alpha * dice + (1 - self.alpha) * boundary
    
class SurfaceDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.surface_loss = SurfaceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        surface = self.surface_loss(logits, targets)
        return self.alpha * dice + (1 - self.alpha) * surface

