import logging
import shutil
from pathlib import Path
import numpy as np
import torch
import rasterio
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict


logger = logging.getLogger(__name__)

def create_weight_matrix(tile_size, overlap_size):
    """Generate a weight matrix using cosine decay for blending."""
    weight = np.ones((tile_size, tile_size), dtype=np.float32)

    # Cosine weights for overlap regions
    edge_weight = 0.5 * (1 + np.cos(np.linspace(-np.pi, 0, overlap_size)))
    weight[:overlap_size, :] *= edge_weight[:, None]  # Top edge
    weight[-overlap_size:, :] *= edge_weight[::-1][:, None]  # Bottom edge
    weight[:, :overlap_size] *= edge_weight[None, :]  # Left edge
    weight[:, -overlap_size:] *= edge_weight[::-1][None, :]  # Right edge

    return weight


def make_prediction_tiles(tile_folder, metadata, model, device, threshold, ):
    logger.info(f'---ORIGINAL PREDICTIONS FUNCTION')
    predictions_folder = Path(tile_folder).parent / f'{tile_folder.stem}_predictions'
    if predictions_folder.exists():
        logger.info(f"--- Deleting existing predictions folder: {predictions_folder}")
        # delete the folder and create a new one
        shutil.rmtree(predictions_folder)
    predictions_folder.mkdir(exist_ok=True)

    for tile_info in tqdm(metadata, desc="Making predictions"):
        tile_path = tile_folder /  tile_info["tile_name"]
        pred_path = predictions_folder / tile_info["tile_name"]

        with rasterio.open(tile_path) as src:
            tile = src.read(1).astype(np.float32)  # Read the first band
            profile = src.profile   
            nodata_mask = src.read_masks(1) == 0  # True where no-data

        # Prepare tile for model
        tile_tensor = torch.tensor(tile).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims

        # Perform inference
        with torch.no_grad():
            pred = model(tile_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()  
            pred = (pred > threshold).astype(np.float32)  
            pred[nodata_mask] = 0  # Mask out no-data areas

        # Save prediction as GeoTIFF
        profile.update(dtype=rasterio.float32)
        with rasterio.open(pred_path, "w", **profile) as dst:
            dst.write(pred.astype(np.float32), 1)

    return predictions_folder

def make_prediction_tiles_new(tile_folder, metadata, model, device, threshold, stride):
    predictions_folder = Path(tile_folder).parent / f'{tile_folder.stem}_predictions'
    if predictions_folder.exists():
        logger.info(f"--- Deleting existing predictions folder: {predictions_folder}")
        shutil.rmtree(predictions_folder)
    predictions_folder.mkdir(exist_ok=True)

    # DETERMINE THE OVERALL OUTPUT SHAPE
    tile_size = 256
    stride = tile_size
    overlap = tile_size - stride

    # CREATE A WEIGHT MATRIX FOR BLENDING
    weight_matrix = create_weight_matrix(tile_size, overlap)

    # DETERMINE THE OVERALL OUTPUT DIMENSIONS
    max_x = max([tile_info['x_start'] for tile_info in metadata]) + tile_size
    max_y = max([tile_info['y_start'] for tile_info in metadata]) + tile_size
    global_shape = (max_x, max_y)

    # INITIALIZE ARRAYS FOR MERGING PREDICTIONS
    global_prediction = np.zeros(global_shape, dtype=np.float32)
    global_weight_sum = np.zeros(global_shape, dtype=np.float32)

    #.................
    # GET A TILE FORM THE METADATA
    for tile_info in tqdm(metadata, desc="Making predictions"):
        tile_path = tile_folder / tile_info["tile_name"]
        x, y = tile_info['x_start'], tile_info['y_start']

        # OPEN THE TILE
        with rasterio.open(tile_path) as src:
            tile = src.read(1).astype(np.float32)  # Read the first band
            profile = src.profile   
            nodata_mask = src.read_masks(1) == 0  # True where no-data

        # PREPARE TILE FOR MODEL
        tile_tensor = torch.tensor(tile).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims

        # PERFORM INFERENCE
        with torch.no_grad():
            pred = model(tile_tensor)
            # pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # Convert logits to probabilities
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # Convert logits to probabilities
            pred[nodata_mask] = 0  # Mask out no-data areas

        # ADD WEIGHTED PREDICTION TO GLOBAL ARRAYS
        global_prediction[x:x+tile_size, y:y+tile_size] += pred * weight_matrix
        global_weight_sum[x:x+tile_size, y:y+tile_size] += weight_matrix

    # NORMALIZE GLOBAL PREDICTIONS BY WEIGHT SUM
    global_weight_sum[global_weight_sum == 0] = 1  # Prevent division by zero
    final_prediction = global_prediction / global_weight_sum
    final_prediction = (final_prediction > threshold).astype(np.float32)
      # Convert probabilities to binary mask


    # SAVE FINAL MERGED PREDICTION AS GEOTIFF
    profile.update(dtype=rasterio.float32, height=global_shape[0], width=global_shape[1])
    merged_path = predictions_folder / "merged_prediction.tif"
    with rasterio.open(merged_path, "w", **profile) as dst:
        dst.write(final_prediction.astype(np.float32), 1)

    return predictions_folder


def stitch_tiles(metadata: list, prediction_tiles, save_path, image):
    # GET CRS AND TRANSFORM
    with rasterio.open(image) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        logger.info(f'src shape: {src.shape}')
    
        # INITIALIZE THE STITCHED IMAGE AND COUNT
        # give stitched_image the same crs, transform and shape as the source image
        stitched_image = np.zeros((height, width))
        # logger.info("stitched_image dtype:", stitched_image.dtype)
        logger.info(f"stitched_image shape: { stitched_image.shape}")
        #logger.info unique values in the stitched image
        # logger.info(f'unique values in empty stitched image: {np.unique(stitched_image)}')

    for tile_info in tqdm(metadata, desc="Stitching tiles"):
        tile_name = tile_info["tile_name"]
        # Extract position info from metadata
        x_start, x_end = tile_info["x_start"], tile_info["x_end"]
        y_start, y_end = tile_info["y_start"], tile_info["y_end"]

        # Find the corresponding prediction tile
        tile = prediction_tiles / tile_name

        # Load the tile
        with rasterio.open(tile) as src:
            tile = src.read(1).astype(np.float32)
            # Debugging: logger.info tile info and shapes
            # logger.info(f"Tile shape: {tile.shape}")
        # logger.info(f" Tile info: {tile_info}")

        # Extract the relevant slice from the stitched image
        stitched_slice = stitched_image[y_start:y_end, x_start:x_end]
        if (stitched_slice.shape[0] == 0) or (stitched_slice.shape[0] == 1):
            continue
        
        # Validate dimensions
        if stitched_slice.shape != tile.shape:
            if (stitched_slice.shape[0] == 0) or (stitched_slice.shape[1] == 0):
                continue
            logger.info(f"---Mismatch: Stitched slice shape: {stitched_slice.shape}, ---Tile shape: {tile.shape}")
            slice_height, slice_width = stitched_slice.shape
            tile = tile[:slice_height, :slice_width]  # Crop tile to match slice
            # Debugging: logger.info the new tile shape
            logger.info(f"New tile shape: {tile.shape}")


        # Add the tile to the corresponding position in the stitched image
        stitched_image[y_start:y_end, x_start:x_end] += tile
        # logger.info STITCHED IMAGE SIZE
        # logger.info(f"Stitched image shape: {stitched_image.shape}")
    logger.info(f'---crs: {crs}')
    # Save the stitched image as tif, as save_path
    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(stitched_image, 1)
    # with rasterio.open(save_path) as src:
    #     logger.info("No-data value:", src.nodata)
        
    return stitched_image


def clean_checkpoint_keys(state_dict):
    """Fix the keys in the checkpoint by removing extra prefixes."""
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.")
        elif key.startswith("model."):
            new_key = key.replace("model.", "")
        else:
            new_key = key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict

# we need a incference_list.csv for the dataloader to work.
# create csv from json tiling metadata file. Extract the first vlue (tile name ) of each item. 
# create a second row in the csv with dummy vales for masks.

def create_inference_csv(metadata):
    """    
    Args:
        metadata (list): List of dictionaries containing tile information.
    """

    # Extract tile names and create dummy mask values
    tile_names = [tile_info["tile_name"] for tile_info in metadata]
    dummy_masks = ["dummy_mask" for _ in tile_names]  # Dummy values for masks

    # Create a DataFrame
    df = pd.DataFrame({
        "image": tile_names,
        "mask": dummy_masks
    })
    return df

def write_df_to_csv(df, csv_path):
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"---Inference CSV created at {csv_path}")