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

def make_prediction_tiles(tile_folder, metadata, model, device, threshold, stride):
    '''

    '''

    logger.info(f'NEW PREDICTIONS FUNCTION')
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


def stitch_tiles_old(metadata: list, prediction_tiles_path, save_path, image):
    '''
    Stitch prediction tiles back into a single image based on metadata.
    Arg image must be one of the original images used for tiling, to get crs and transform.
    '''
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
    metadata_tile_list = metadata
    logger.info(f'---metadata_tile_list: {type(metadata_tile_list)}')

    for tile in tqdm(metadata_tile_list, desc="Stitching tiles"):
        tile_name = tile["tile_name"]
        # Extract position info from metadata
        x_start, x_end = tile["x_start"], tile["x_end"]
        y_start, y_end = tile["y_start"], tile["y_end"]

        # Find the corresponding prediction tile
        tile = prediction_tiles_path / tile_name

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


def stitch_tiles(metadata, prediction_tiles, save_path, image, tile_size, stride, threshold=0.5, blend="cosine"):
    """
    Reconstructs a full-resolution prediction image from overlapping tiles.
    
    This function takes individual prediction tiles (which may overlap) and stitches them 
    back together into a single georeferenced image. When tiles overlap, it uses weighted 
    blending to ensure smooth transitions between tiles.
    
    Args:
        metadata: List of dicts, each containing:
                  - x_start, y_start: Top-left corner of tile in full image coords (pixels)
                  - x_end, y_end: Bottom-right corner of tile in full image coords (pixels)
                  - tile_name: Filename of the prediction tile GeoTIFF
        prediction_tiles: Path to folder containing per-tile prediction GeoTIFFs (float 0–1)
        save_path: Where to save the final stitched GeoTIFF
        image: Path to original image (used to get CRS, transform, and output dimensions)
        tile_size: int, edge length of each tile (pixels)
        stride: int, step size used when sliding window during tiling
        threshold: float (0-1), binarization threshold to convert probabilities to binary predictions
        blend: "cosine" or "avg"; blending method for overlap regions (only used when overlap > 0)
               - "cosine": Smooth cosine-weighted blend (reduces tile boundary artifacts)
               - "avg": Simple averaging (uniform weights)
    
    Returns:
        merged: 2D numpy array (H, W) of binary predictions (0 or 1)
    """
    logger.info(f'++++STITCH TILES FUNCTION')
    
    # Calculate overlap between tiles: if stride < tile_size, tiles will overlap
    overlap = tile_size - stride

    # Get dimensions and geospatial profile from the original image
    with rasterio.open(image) as src:
        H, W = src.height, src.width  # Full image dimensions
        out_profile = src.profile.copy()  # Copy CRS, transform, etc.
        out_profile.update(dtype=rasterio.float32, count=1)  # Single-band float output

    # Case 1: No overlap (stride == tile_size)
    # In this case, tiles are adjacent with no blending needed
    if overlap <= 0:
        logger.info(f' NO OVERLAP')
        # Initialize empty canvas for the full image
        stitched_image = np.zeros((H, W), dtype=np.float32)
        
        # Place each tile directly into its position
        for t in metadata:
            ys, ye = t["y_start"], t["y_end"]
            xs, xe = t["x_start"], t["x_end"]
            tile_path = prediction_tiles / t["tile_name"]
            
            # Load the prediction tile
            with rasterio.open(tile_path) as tsrc:
                tile = tsrc.read(1).astype(np.float32)  # Shape: (tile_h, tile_w)
            
            # Add tile to canvas (simple addition since there's no overlap)
            stitched_image[ys:ye, xs:xe] += tile
        
        # Apply threshold to binarize probabilities
        merged = (stitched_image > threshold).astype(np.float32)
    
    # Case 2: Tiles overlap (stride < tile_size)
    # Need weighted blending to avoid sharp edges at tile boundaries
    else:
        logger.info(f' OVERLAP: {overlap}')
        
        def _cosine_ramp(n):
            """
            Creates a smooth cosine ramp from 0 to 1 over n samples.
            This produces a smooth transition that reduces tile boundary artifacts.
            
            Formula: 0.5 * (1 + cos(theta)) where theta goes from -π to 0
            Result: Values smoothly transition from 0 → 1
            """
            return 0.5 * (1.0 + np.cos(np.linspace(-np.pi, 0.0, n, dtype=np.float32)))

        def _make_weight(h, w, ov):
            """
            Creates a 2D weight matrix for blending overlapping tiles.
            
            The weight is 1.0 in the tile center, and tapers down to 0 at the edges 
            using a cosine ramp over the overlap region. This ensures smooth blending
            where tiles overlap.
            
            Args:
                h, w: Height and width of the tile
                ov: Overlap size (in pixels)
            
            Returns:
                2D weight array of shape (h, w) with values in [0, 1]
            """
            # Start with uniform weights of 1.0
            wx = np.ones((w,), dtype=np.float32)  # Weight in x-direction
            wy = np.ones((h,), dtype=np.float32)  # Weight in y-direction
            
            # Limit overlap to tile dimensions
            ox = min(ov, w)
            oy = min(ov, h)
            
            # Apply cosine ramp to left and right edges (x-direction)
            if ox > 0:
                r = _cosine_ramp(ox)  # Ramp from 0 to 1
                wx[:ox] *= r           # Left edge: ramp up from 0 to 1
                wx[-ox:] *= r[::-1]    # Right edge: ramp down from 1 to 0
            
            # Apply cosine ramp to top and bottom edges (y-direction)
            if oy > 0:
                r = _cosine_ramp(oy)
                wy[:oy] *= r           # Top edge: ramp up
                wy[-oy:] *= r[::-1]    # Bottom edge: ramp down
            
            # Combine x and y weights into 2D weight matrix
            # Outer product: wy (vertical) × wx (horizontal)
            return wy[:, None] * wx[None, :]

        # Initialize accumulators for weighted blending
        global_pred = np.zeros((H, W), dtype=np.float32)  # Accumulates weighted predictions
        global_wsum = np.zeros((H, W), dtype=np.float32)  # Accumulates sum of weights

        # Process each tile
        for t in metadata:
            # Get tile position in full image coordinates
            ys, ye = t["y_start"], t["y_end"]
            xs, xe = t["x_start"], t["x_end"]
            tile_path = prediction_tiles / t["tile_name"]
            
            # Load the prediction tile (values in [0, 1])
            with rasterio.open(tile_path) as tsrc:
                tile = tsrc.read(1).astype(np.float32)

            h, w = tile.shape
            
            # Create weight matrix for this tile
            if blend == "avg":
                # Uniform weights (simple averaging in overlap regions)
                weight = np.ones((h, w), dtype=np.float32)
            else:  # "cosine"
                # Cosine-weighted (smooth tapering at edges)
                weight = _make_weight(h, w, overlap)

            # Add weighted tile to global accumulator
            # In overlap regions, multiple tiles will contribute with their respective weights
            global_pred[ys:ye, xs:xe] += tile * weight
            
            # Accumulate the weights themselves
            # This tells us the total weight at each pixel (for normalization)
            global_wsum[ys:ye, xs:xe] += weight

        # Normalize by dividing by total weight at each pixel
        # This gives the weighted average in overlap regions
        np.putmask(global_wsum, global_wsum == 0, 1.0)  # Avoid division by zero
        merged = global_pred / global_wsum
        
        # Apply threshold to binarize the blended probabilities
        merged = (merged > threshold).astype(np.float32)

    # Save the stitched result as a GeoTIFF
    with rasterio.open(save_path, "w", **out_profile) as dst:
        dst.write(merged, 1)

    return merged


def stitch_tiles_raw(metadata, tiles_path, save_path, image, tile_size, stride):
    """
    Reconstructs a full-resolution raw image from overlapping tiles.
    
    This function is similar to stitch_tiles() but operates on raw image data (not predictions).
    It reconstructs the original SAR/optical imagery by stitching tiles back together with 
    smooth blending in overlap regions. Unlike stitch_tiles(), this does NOT apply any 
    thresholding or binarization - it preserves the original pixel values and data type.
    
    Use case: Quality checking or visualizing the tiled input data to verify that 
    the tiling/stitching process doesn't introduce artifacts.
    
    Args:
        metadata: List of dicts, each containing:
                  - x_start, y_start: Top-left corner of tile in full image coords (pixels)
                  - x_end, y_end: Bottom-right corner of tile in full image coords (pixels)
                  - tile_name: Filename of the raw tile GeoTIFF
        tiles_path: Path to folder containing raw tile GeoTIFFs (not predictions)
        save_path: Path where the stitched full-resolution image will be saved
        image: Path to original image (used to get CRS, transform, dimensions, and dtype)
        tile_size: int, edge length of each tile (pixels)
        stride: int, step size used when sliding window during tiling
    
    Returns:
        stitched_image: 2D numpy array (H, W) with reconstructed image data in original dtype
    """
    logger.info(f'++++STITCH TILES RAW (NO PREDICTIONS)')
    
    # Calculate overlap between tiles: if stride < tile_size, tiles will overlap
    overlap = tile_size - stride

    # Get dimensions, geospatial profile, and original data type from the source image
    with rasterio.open(image) as src:
        H, W = src.height, src.width  # Full image dimensions
        out_profile = src.profile.copy()  # Copy CRS, transform, etc.
        # Preserve original format for raw image stitching
        nb = src.count  # Number of bands in original image
        orig_dtype = src.dtypes[0]  # Original data type (uint8, uint16, float32, etc.)

    # Case 1: No overlap (stride == tile_size)
    # Tiles are adjacent with no blending needed - simple placement
    if overlap <= 0:
        logger.info(f'NO OVERLAP - simple reconstruction')
        # Initialize empty canvas for the full image
        stitched_image = np.zeros((H, W), dtype=np.float32)
        
        # Place each tile directly into its position
        for t in metadata:
            ys, ye = t["y_start"], t["y_end"]
            xs, xe = t["x_start"], t["x_end"]
            tile_path = tiles_path / t["tile_name"]
            
            # Load the raw tile data
            with rasterio.open(tile_path) as tsrc:
                tile = tsrc.read(1).astype(np.float32)  # Shape: (tile_h, tile_w)
            
            # Direct assignment (no addition since tiles don't overlap)
            stitched_image[ys:ye, xs:xe] = tile
    
    # Case 2: Tiles overlap (stride < tile_size)
    # Need weighted blending to avoid sharp edges at tile boundaries
    else:
        logger.info(f'OVERLAP: {overlap} - using cosine blending')
        
        def _cosine_ramp(n):
            """
            Creates a smooth cosine ramp from 0 to 1 over n samples.
            This produces a smooth transition that reduces tile boundary artifacts.
            
            Formula: 0.5 * (1 + cos(theta)) where theta goes from -π to 0
            Result: Values smoothly transition from 0 → 1
            """
            return 0.5 * (1.0 + np.cos(np.linspace(-np.pi, 0.0, n, dtype=np.float32)))

        def _make_weight(h, w, ov):
            """
            Creates a 2D weight matrix for blending overlapping tiles.
            
            The weight is 1.0 in the tile center, and tapers down to 0 at the edges 
            using a cosine ramp over the overlap region. This ensures smooth blending
            where tiles overlap.
            
            Args:
                h, w: Height and width of the tile
                ov: Overlap size (in pixels)
            
            Returns:
                2D weight array of shape (h, w) with values in [0, 1]
            """
            # Start with uniform weights of 1.0
            wx = np.ones((w,), dtype=np.float32)  # Weight in x-direction
            wy = np.ones((h,), dtype=np.float32)  # Weight in y-direction
            
            # Limit overlap to tile dimensions
            ox = min(ov, w)
            oy = min(ov, h)
            
            # Apply cosine ramp to left and right edges (x-direction)
            if ox > 0:
                r = _cosine_ramp(ox)  # Ramp from 0 to 1
                wx[:ox] *= r           # Left edge: ramp up from 0 to 1
                wx[-ox:] *= r[::-1]    # Right edge: ramp down from 1 to 0
            
            # Apply cosine ramp to top and bottom edges (y-direction)
            if oy > 0:
                r = _cosine_ramp(oy)
                wy[:oy] *= r           # Top edge: ramp up
                wy[-oy:] *= r[::-1]    # Bottom edge: ramp down
            
            # Combine x and y weights into 2D weight matrix
            # Outer product: wy (vertical) × wx (horizontal)
            return wy[:, None] * wx[None, :]

        # Initialize accumulators for weighted blending
        global_data = np.zeros((H, W), dtype=np.float32)  # Accumulates weighted pixel values
        global_wsum = np.zeros((H, W), dtype=np.float32)  # Accumulates sum of weights

        # Process each tile
        for t in metadata:
            # Get tile position in full image coordinates
            ys, ye = t["y_start"], t["y_end"]
            xs, xe = t["x_start"], t["x_end"]
            tile_path = tiles_path / t["tile_name"]
            
            # Load the raw tile data
            with rasterio.open(tile_path) as tsrc:
                tile = tsrc.read(1).astype(np.float32)

            h, w = tile.shape
            
            # Create cosine-weighted blend matrix for this tile
            # (Always uses cosine for raw data - no "avg" option here)
            weight = _make_weight(h, w, overlap)

            # Add weighted tile to global accumulator
            # In overlap regions, multiple tiles will contribute with their respective weights
            global_data[ys:ye, xs:xe] += tile * weight
            
            # Accumulate the weights themselves
            # This tells us the total weight at each pixel (for normalization)
            global_wsum[ys:ye, xs:xe] += weight

        # Normalize by dividing by total weight at each pixel
        # This gives the weighted average in overlap regions
        np.putmask(global_wsum, global_wsum == 0, 1.0)  # Avoid division by zero
        stitched_image = global_data / global_wsum

    # Convert back to original data type to preserve image format
    # This is critical - raw data may be uint8 (0-255), uint16 (0-65535), etc.
    if orig_dtype == 'uint8':
        stitched_image = np.clip(stitched_image, 0, 255).astype(np.uint8)
    elif orig_dtype == 'uint16':
        stitched_image = np.clip(stitched_image, 0, 65535).astype(np.uint16)
    else:
        # For float32, float64, or other types, just cast
        stitched_image = stitched_image.astype(orig_dtype)

    # Update output profile with original band count and data type
    out_profile['count'] = nb
    out_profile['dtype'] = orig_dtype
    
    # Save the stitched result as a GeoTIFF
    # Note: Writes the same stitched_image to all bands (assumes single-band processing)
    with rasterio.open(save_path, "w", **out_profile) as dst:
        for band_idx in range(1, nb + 1):
            dst.write(stitched_image, band_idx)

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
    logger.info(f'\n\n++++++++ IN CREATE INFERENCE CSV\n')
    # Extract tile names and create dummy mask values
    tile_names = [tile_info["tile_name"] for tile_info in metadata]
    logger.info(f'---tile names: {tile_names}')
    dummy_masks = ["dummy_mask" for _ in tile_names]  # Dummy values for masks

    # Create a DataFrame
    df = pd.DataFrame({
        "image": tile_names,
        "mask": dummy_masks
    })
    logger.info(f'\n\n+++++++ END FUNCTION\n')
    return df

def write_df_to_csv(df, csv_path):
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"---Inference CSV created at {csv_path}")