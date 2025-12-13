import os
import rasterio
import rioxarray as rxr # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from tqdm import tqdm
from pathlib import Path
import netCDF4 as nc
from osgeo import gdal
import shutil
import re
import json
# MODULES
# from check_int16_exceedance import check_int16_exceedance
import subprocess
import rasterio
import numpy as np
import logging
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.process.process_helpers import  nan_check, print_dataarray_info


logger = logging.getLogger(__name__)

def fill_nodata_with_zero(input_file):
    logger.info('+++in fill_nodata_with_zero fn')
    #logger.info('+++in fill_nodata_with_zero fn')
    input_str = str(input_file)
    # Open the input raster file
    dataset = gdal.Open(input_str, gdal.GA_Update)
    band = dataset.GetRasterBand(1)

    # Get the nodata value (if it exists)
    nodata_value = band.GetNoDataValue()

    # Read the raster as a NumPy array
    data = band.ReadAsArray()

    # Replace nodata values with 0
    if nodata_value is not None:
        #logger.info('---replacing nans with 0 in ',input_file.name)
        data[data == nodata_value] = 0
    
    # Write the modified array back to the raster
    band.WriteArray(data)
    
    # Flush and close the file
    band.FlushCache()
    dataset = None  # Close the file

def check_layers(layers, layer_names):
    '''
    checks the layers and logger.infos out some info
    '''
    logger.info('\n+++in check_layers fn+++++++++++++++++++++++++')
    # Assuming you have a list of Dask arrays, each representing a layer
    
    for i, layer in enumerate(layers):
        logger.info(f'---layer name = {layer_names[i]}')
        #logger.info(f"---Layer {i+1}:")

        # logger.info the shape of the layer
        #logger.info(f"---Shape: {layer.shape}")

        # logger.info the data type of the layer
        #logger.info(f"---Data Type: {layer.dtype}")

        # Assuming the array has x and y coordinates in the `.coords` attribute (like in xarray)
        # You can access and logger.info the coordinates if it is an xarray DataArray or a similar structure
        #if hasattr(layer, 'coords'):  # If using Dask with xarray-like data
        #    logger.info(f"---X Coordinates: {layer.coords['x']}")
        #    logger.info(f"---Y Coordinates: {layer.coords['y']}")
        #else:
        #    logger.info("---No coordinate information available.")

        # Check for NaN or Inf values in the layer
        nan_check(layer)
        check_int16_range(layer)

# SEPERATE SAR LAYERS
def create_vv_and_vh_tifs(file):
    '''
    will delete the original image after creating the vv and vh tifs
    '''
    logger.info('+++in create_vv_and_vh_tifs fn')

    # logger.info(f'---looking at= {file.name}')
    if 'img.tif' in file.name:
        #logger.info(f'---found image file= {file.name}')
        # Open the multi-band TIFF
        with rasterio.open(file) as target:
            # Read the vv (first band) and vh (second band)
            vv_band = target.read(1)  # Band 1 (vv)
            vh_band = target.read(2)  # Band 2 (vh)
            # Define metadata for saving new files
            meta = target.meta
            # Update meta to reflect the single band output
            meta.update(count=1)
            # Save the vv band as a separate TIFF
            vv_newname = file.name.rsplit('_', 1)[0]+'_vv.tif'
            with rasterio.open(file.parent / vv_newname, 'w', **meta) as destination:
                destination.write(vv_band, 1)  # Write band 1 (vv)
            # Save the vh band as a separate TIFF
            vh_newname = file.name.rsplit('_', 1)[0]+'_vh.tif'
            with rasterio.open(file.parent / vh_newname, 'w', **meta) as destination:
                destination.write(vh_band, 1)  # Write band 2 (vh)
        logger.info('---DELETING ORIGINAL IMAGE FILE')
        file.unlink()  # Delete the original image file  
    else:
        logger.info(f'---NO IMAGE FOUND !!!!!= {file.name}')

            # delete original image using unlink() method
    #logger.info('---finished create_vv_and_vh_tifs fn')

# CREAT ANALYSIS EXTENT
def create_extent_from_mask(mask_path, output_raster, no_data_value=None):
    """
    this assumes the mask is a binary mask where valid data is 1 and no-data is 0, probably for the surrounding area"""
    # Load the mask file
    logger.info('+++in create_extent_from_mask fn')
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs
        logger.info(f'---src.nodata= {src.nodata}')

        # Identify no-data value
        if no_data_value is None:
            no_data_value = src.nodata
        if no_data_value is None:
            logger.info("No no-data value found in metadata or provided.")
            return
            # create a binary mask with the entire image as 1
            
        # Create a binary mask (1 for valid data, 0 for no-data)
        binary_mask = (mask != no_data_value).astype(np.uint8)

    # Save the binary mask as a GeoTIFF
    with rasterio.open(
        output_raster,
        "w",
        driver="GTiff",
        height=binary_mask.shape[0],
        width=binary_mask.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(binary_mask, 1)

    logger.info(f"extent saved to {output_raster}")


def create_valid_mask(img_path, mask_path, mask_code, dest_folder, inference=False):  
    """
    Creates a valid mask from img and mask files.
    The valid mask is saved as 'valid_mask.tif'.
    The original mask is rewritten with IGNORE_VAL where invalid.
    """
    logger.info('+++in create_valid_mask fn')
    
    # Constants
    PAD_VAL      = 0          # or 1
    ATOL         = 1e-5
    IGNORE_VAL   = 255

    # --- read the bands ---
    with rasterio.open(img_path) as img_src, \
         rasterio.open(mask_path) as msk_src:

        img   = img_src.read(1)
        mask = msk_src.read(1)
        prof = img_src.profile        # same for all bands after reprojection
        logger.info(f'---img shape= {img.shape}')
        logger.info(f'---mask shape= {mask.shape}')
        logger.info(f'---img dtype= {img.dtype}')
    # --- build valid mask ---
    valid = ~(np.isclose(img, PAD_VAL, atol=ATOL) &
              np.isclose(mask, PAD_VAL, atol=ATOL))

    # --- save valid mask (Byte, 0/1) ---
    vmask_prof = prof.copy()
    vmask_prof.update(count=1, dtype=rasterio.uint8, nodata=0)
    with rasterio.open(dest_folder / "valid_mask.tif", "w", **vmask_prof) as dst:
        dst.write(valid.astype(np.uint8), 1)

    # --- rewrite mask with IGNORE ---
    mask_clean = np.where(valid, mask, IGNORE_VAL)
    mask_prof  = prof.copy()
    mask_prof.update(dtype=rasterio.uint8, nodata=IGNORE_VAL)
    with rasterio.open(dest_folder / "mask_no_padding.tif", "w", **mask_prof) as dst:
        dst.write(mask_clean.astype(np.uint8), 1)

        # --- optional crop to valid bbox ---
    rows, cols = np.where(valid)
    row0, row1 = rows.min(), rows.max()+1
    col0, col1 = cols.min(), cols.max()+1
    window = ((row0, row1), (col0, col1))

    cropped_image = dest_folder / f'{mask_code}_cropped_image.tif'
    
# NORMALIZING
def compute_image_min_max(image, band_to_read=1):
    with rasterio.open(image) as src:
        # Read the data as a NumPy array
        data = src.read(band_to_read)  # Read the first band
        # Update global min and max
        min = int(data.min())
        max = int(data.max())
        logger.info(f"---{image.name}: Min: {data.min()}, Max: {data.max()}")
    return min, max

def calculate_and_normalize_slope(input_dem, mask_code):
    """
    Calculate slope from a DEM using GDAL and normalize it between 0 and 1.
    """

    # Step 1: Calculate slope using GDAL's gdaldem
    temp_slope = "temp_slope.tif"  # Temporary slope file
    gdal_command = [
        "gdaldem", "slope",
        input_dem,         # Input DEM
        temp_slope,         # Output raw slope file
        "-compute_edges"
    ]

    try:
        subprocess.run(gdal_command, check=True)
        logger.info(f"Raw slope raster created: {temp_slope}")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error calculating slope: {e}")
        return

    # Step 2: Normalize slope using rasterio
    with rasterio.open(temp_slope) as src:
        slope = src.read(1)  # Read the slope data
        slope_min, slope_max = slope.min(), slope.max()
        logger.info(f"Min slope: {slope_min}, Max slope: {slope_max}")

        # Normalize the slope to the range [0, 1]
        slope_norm_data = (slope - slope_min) / (slope_max - slope_min)

        # Prepare metadata for output file
        meta = src.meta.copy()
        meta.update(dtype='float32')

        normalized_slope = input_dem.parent / f"{mask_code}_slope_norm.tif"

        # Save the normalized slope
        with rasterio.open(normalized_slope, 'w', **meta) as dst:
            dst.write(slope_norm_data.astype(np.float32), 1)

    # Cleanup temporary raw slope file
    Path(temp_slope).unlink()

    logger.info(f"Normalized slope raster saved to: {normalized_slope}")
    return normalized_slope

# CHANGE DATA TYPE
def make_float32_inf(input_tif, output_file):
    '''
    converts the tif to float32
    '''
    # logger.info('+++in make_float32 inf')
    with rasterio.open(input_tif) as src:
        data = src.read()
        # logger.info(f"---Original shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == 'float32':
            logger.info(f'---{input_tif.name} already float32')
            meta = src.meta.copy()
            meta['count'] = 1
        else:
            # logger.info(f'---{input_tif.name} converting to float32')
            # Update the metadata
            meta = src.meta.copy()
            meta.update(dtype='float32')
            # set num bands to 1
            meta['count'] = 1
            # Write the new file
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(data.astype('float32'))
            return output_file

def make_float32(input_tif, file_name):
    '''
    converts the tif to float32
    '''
    input_tif = Path(input_tif)
    file_name = Path(file_name)
    logger.info('+++in make_float32 fn')
    with rasterio.open(input_tif) as src:
        data = src.read()
        logger.info(f"---Original shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == 'float32':
            logger.info(f'---{input_tif.name} already float32')
            src.close()
            input_tif.rename(file_name)
            logger.info(f'---renamed {input_tif.name} to {file_name}')
            return file_name

        else:
            logger.info(f'---{input_tif.name} converting to float32')
            # Update the metadata
            meta = src.meta.copy()
            meta.update(dtype='float32')
            # set num bands to 1
            meta['count'] = 1
            # Write the new file
            with rasterio.open(file_name, 'w', **meta) as dst:
                dst.write(data.astype('float32'))        
        return file_name

def make_float32_inmem(input_tif):

    # Open the input TIFF file
    with rasterio.open(input_tif) as src:
        # Read the data from the input file
        data = src.read()
        meta = src.meta.copy()

        # Check if data is already float32
        if meta['dtype'] == 'float32':
            logger.info('---Data already in float32 format.')
            return src  # Return the original dataset if already float32

        # Convert data to float32
        converted_data = data.astype('float32')

        # Update metadata to reflect new dtype
        meta.update(dtype='float32')

        # Create a new in-memory file with updated metadata and float32 data
        with MemoryFile() as memfile:
            with memfile.open(**meta) as mem:
                mem.write(converted_data)
                logger.info('---Converted to float32 and written to memory.')
                return memfile.open()


    return output_file

def xxx():

        # MATCH THE DEM TO THE SAR IMAGE
        # final_dem = extract_folder / f'{mask_code}_aligned_dem.tif'
        # match_dem_to_mask(image, dem, final_dem)
        # # logger.info(f'>>>final_dem={final_dem.name}')

        # # CHECK THE NEW DEM
        # with rasterio.open(final_dem) as dem_src:
        #     with rasterio.open(image) as img_src:
        #         image_bounds = img_src.bounds
        #         image_crs = img_src.crs
        #         logger.info(f'>>>image crs={image_crs}')
        #         logger.info(f'>>>dem crs={dem_src.crs}')  
        #         img_width = img_src.width
        #         img_height = img_src.height
        #         img_transform = img_src.transform
        #         logger.info(f'>>>bounds match={dem_src.bounds == image_bounds}')
        #         logger.info(f'>>>crs match={dem_src.crs == image_crs}')
        #         logger.info(f'>>>width match={dem_src.width == img_width}')
        #         logger.info(f'>>>height match={dem_src.height == img_height}')
        #         logger.info(f'>>>transform match={dem_src.transform == img_transform}')
        #         logger.info(f'>>>count match={dem_src.count == img_src.count}')

        # normalized_slope = calculate_and_normalize_slope(final_dem, mask_code)

        # # CHECK THE SLOPE
        # with rasterio.open(extract_folder / f'{mask_code}_slope_norm.tif') as src:
        #     logger.info(f'>>>slope min={src.read().min()}')
        #     logger.info(f'>>>slope max={src.read().max()}')
        #     data = src.read()
        #     nonans = nan_check(data)
        #     logger.info(f'>>>nonans?={nonans}')
        pass

# REPROJECTING
def reproject_layers_to_4326_TSX( src_path, dst_path):
    logger.info('+++in reproject_layers_to_4326_TSX fn')

    with rasterio.open(src_path) as src:
        # logger.info(f'---src_path= {src_path.name}')
        # logger.info(f'---dst_path= {dst_path.name}')
        # logger.info(f'---src_path crs = {src.crs}')
        
        transform, width, height = calculate_default_transform(src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': 'EPSG:4326', 'transform': transform, 'width': width, 'height': height, 'dtype': 'float32'})
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                band=src.read(i)
                band[band > 1] = 1
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)
            # logger.info(f'---reprojected {src_path.name} to {dst_path.name} with {dst.crs}')

def reproject_to_4326_gdal(input_path, output_path, resampleAlg):
    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)
    # Open the input raster
    src_ds = gdal.Open(input_path)
    if not src_ds:
        logger.info(f"Failed to open {input_path}")
        return
    
    # Ensure the input raster has a spatial reference system
    src_srs = src_ds.GetProjection()
    if not src_srs:
        raise ValueError(f"Input raster {input_path} has no CRS.")

    target_srs = 'EPSG:4326'

    # Use GDAL's warp function to reproject
    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,  # Target CRS
        resampleAlg=resampleAlg,  # Resampling method (nearest neighbor for categorical data)
        format="GTiff",
    )  
    gdal.Warp(output_path, src_ds, options=warp_options)
    # logger.info(f"---Reprojected raster saved to: {output_path}")           

    return output_path   

def reproject_to_4326_fixpx_gdal(input_path, output_path, resampleAlg, px_size):
    logger.info('+++in reproject_to_4326_fixpx_gdal fn')
    # logger.info(f'---resampleAlg= {resampleAlg}')
    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)
    # Open the input raster
    src_ds = gdal.Open(input_path)
    if not src_ds:
        logger.info(f"Failed to open {input_path}")
        return

    target_srs = 'EPSG:4326'

    # Use GDAL's warp function to reproject
    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,  # Target CRS
        xRes=px_size,
        yRes=px_size,
        resampleAlg=resampleAlg,  # Resampling method (nearest neighbor for categorical data)
    )  
    gdal.Warp(output_path, src_ds, options=warp_options)
    # logger.info(f"---Reprojected raster saved to: {output_path}")           

    return output_path   

# RESAMPLING
def match_resolutions_with_check(event):
    """
    Match the resolution and dimensions of the target raster to the reference raster
    only if they differ.
    """
    #logger.info('+++++++in match_resolutions_with_check fn')

    # Find the reference file (vv.tif)
    reference_path = None
    for file in event.iterdir():
        if 'vv.tif' in file.name:
            reference_path = file
            break
    
    if not reference_path:
        logger.info('--- No reference vv.tif file found.')
        return
    
    # Open the reference layer to get its resolution
    with rxr.open_rasterio(reference_path) as reference_layer:
        reference_resolution = reference_layer.rio.resolution()
        #logger.info(f'--- Reference file {reference_path.name} resolution: {reference_resolution}')
    
    # Loop through other files to check and reproject if necessary
    for file in event.iterdir():
        if file.is_file(): 
            patterns = ['vv.tif', 'vh.tif', '_s2_', '.json','.nc']
            if not any(i in file.name for i in patterns) and 'epsg4326' in file.name:

                #logger.info(f'--- analysing this file = {file.name}')
                # Open the target layer to compare resolutions
                with rxr.open_rasterio(file) as target_layer:
                    target_resolution = target_layer.rio.resolution()
                    #logger.info(f'--- Target file {file.name} resolution: {target_resolution}')

                    # Compare resolutions with a small tolerance
                    if abs(reference_resolution[0] - target_resolution[0]) < 1e-10 and \
                       abs(reference_resolution[1] - target_resolution[1]) < 1e-10:
                        #logger.info(f"--- Skipping {file.name} (resolution already matches)")
                        continue

                    # Reproject target file to match reference
                    reprojected_layer = target_layer.rio.reproject_match(reference_layer)
                # Save reprojected file (once target file is closed)
                reprojected_layer.rio.to_raster(file)
                #logger.info(f'--- Resampled raster saved to: {file.name}')

def resample_tiff(src_image, dst_image, target_res):
    logger.info(f'+++resample_tiff::::target res= {target_res}')
    with rasterio.open(src_image, 'r') as src:
        # Read metadata and calculate new dimensions
        src_res = src.res  # (pixel width, pixel height in CRS units)
        logger.info(f'---src_res= {src_res}')
        scale_factor_x = src_res[0] / target_res
        scale_factor_y = src_res[1] / target_res

        new_width = round(src.width * scale_factor_x)
        new_height = round(src.height * scale_factor_y)
        logger.info(f'--- New Dimensions: width={new_width}, height={new_height}')
        new_transform = src.transform * src.transform.scale(scale_factor_x, scale_factor_x)
        logger.info(f'--- New Transform: {new_transform}')

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
        })

        # Resample and write to the new file
        with rasterio.open(dst_image, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):  # Loop through each band
                resampled_data = src.read(
                    i,
                    out_shape=(new_height, new_width),
                    resampling=Resampling.bilinear
                )
                dst.write(resampled_data, i)
    logger.info(f"Resampled image saved to {dst_image}")

def resample_tiff_gdal(src_image, dst_image, target_res):
    """
    Simplified resampling of a GeoTIFF to the specified resolution using GDAL.

    Parameters:
        src_image (str): Path to the source GeoTIFF.
        dst_image (str): Path to save the resampled GeoTIFF.
        target_res (float): Target resolution in CRS units (e.g., meters per pixel).
    """
        # Ensure inputs are strings
    if isinstance(src_image, Path):
        src_image = str(src_image)
    if isinstance(dst_image, Path):
        dst_image = str(dst_image)
    logger.info(f'+++ Resampling {src_image} to target resolution: {target_res} m/pixel')

    # Open the source dataset
    src_ds = gdal.Open(src_image)
    if not src_ds:
        raise FileNotFoundError(f"Cannot open source file: {src_image}")

    # Use GDAL's Warp function to resample
    gdal.Warp(
        dst_image,
        src_ds,
        xRes=target_res,
        yRes=target_res,
        resampleAlg=gdal.GRA_Bilinear,  # Use bilinear resampling
        outputType=gdal.GDT_Float32,   # Save as 32-bit float
    )

    logger.info(f'+++ Resampled image saved to: {dst_image}')

# CREATING DATACUBES

def make_layerdict(extracted):
    '''
    iterates through the files in the extacted folder and returns a dict of the datas
    '''
    logger.info(f'+++in make_layerdict fn for {extracted.name}')
    if not extracted.exists():
        raise FileNotFoundError(extracted)
    if not extracted.is_dir():
        logger.warning(f"{extracted} is not a directory; using its parent.")
        extracted = extracted.parent
    
    files = list(extracted.iterdir())
    if not files:
        logger.warning(f"{extracted} is empty.")
        return {}
    
    datas = {}

    for file in extracted.iterdir():
        logger.info(f'---file {file}')
        if '_vv' in file.name.lower():
            datas[file.name] = 'vv'
            # logger.info(f'---+image file found {file}')
        elif '_vh' in file.name.lower():
            datas[file.name] = 'vh'
            # logger.info(f'---+dem file found {file}')
        elif '4326_slope.tif' in file.name.lower():
            datas[file.name] = 'slope'   
            # logger.info(f'---+slope file found {file}')
        elif 'final_mask.tif' in file.name.lower():
            datas[file.name] = 'mask'
            # logger.info(f'---+mask file found {file}')
        elif 'final_extent.tif' in file.name.lower():
            datas[file.name] = 'extent'
            # logger.info(f'---+valid file found {file}')

    logger.info(f'---datas {datas}')
    return datas

    logger.info("\n+++ Datacube Health Check Completed +++")



    
    logger.info("\n+++ Datacube Health Check Completed +++")

def make_das_from_layerdict( layerdict, folder):
    dataarrays = []
    layer_names = []
    for tif_file, band_name in layerdict.items():
        if not 'aux.xml' in tif_file:
            logger.info(f'---tif_file= {tif_file}')
            logger.info(f'---band_name= {band_name}')
            filepath = folder / tif_file
            # logger.info(f'---**************filepath = {filepath.name}')
            tiffda = rxr.open_rasterio(filepath)
            nan_check(tiffda)
            # logger.info(f'---{band_name}= {tiffda}')   
            # check num uniqq values
            # logger.info(f"---Unique data: {np.unique(tiffda.data)}")
            # logger.info("----unique values:", np.unique(tiffda.values))
            dataarrays.append(tiffda)
            layer_names.append(band_name)
#   
    return dataarrays, layer_names
# def set_tif_dtype_to_float32(tif_file):
    
def check_int16_range(dataarray):
    # TAKES A DATAARRAY NOT A DATASET
    #logger.info("+++in small int16 range check fn+++")
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if (dataarray < int16_min).any() or (dataarray > int16_max).any():
        logger.info(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        # Calculate actual min and max values in the array
        actual_min = dataarray.min().item()
        actual_max = dataarray.max().item()
        
        logger.info(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        logger.info(f"---Minimum value found: {actual_min}")
        logger.info(f"---Maximum value found: {actual_max}")
        return False
    
    # else:
    #     logger.info(f"---no exceedances int16.")

    # Optional: Replace NaN and Inf values if necessary
    # dataarray = dataarray.fillna(0)  # Replace NaN with 0 or another appropriate value
    # dataarray = dataarray.where(~np.isinf(dataarray), 0)  # Replace Inf with 0 or appropriate value

def nan_check(nparray):
    if np.isnan(nparray).any():
        logger.info("----Warning: NaN values found in the data.")
        return False
    else:
        logger.info("----NO NANS FOUND")
        return True

def create_event_datacube_TSX(extracted_path, mask_code, VERSION="v1"):
    '''
    An xarray dataset is created for the event folder and saved as a .nc file.
    '''
    logger.info(f'+++++++++++ IN CREAT EVENT DATACUBE TSX {extracted_path.name}+++++++++++++++++')
    # FIND THE EXTRACTED FOLDER
    # extracted_path = list(event.rglob(f'*{mask_code}_extracted'))[0]

    logger.info(f'---extracted-folder = {extracted_path}')
    logger.info(f'---mask code= {mask_code}')
    layerdict = make_layerdict_TSX(extracted_path)

    logger.info(f'---making das from layerdict= {layerdict}')
    dataarrays, layer_names = make_das_from_layerdict( layerdict, extracted_path)

    # logger.info(f'---CHECKING DATAARRAY LIST')
    # check_dataarray_list(dataarrays, layer_names)

    logger.info(f'---CREATING CONCATERNATED DATASET')
    da = xr.concat(dataarrays, dim='layer').astype('float32')   
    da = da.assign_coords(layer=layer_names)

    # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
    if 'band' in da.dims and da.sizes['band'] == 1:
        logger.info('---Squeezing out the "band" dimension')
        da = da.squeeze('band') 

    # logger.info_dataarray_info(da)

    #######   CHUNKING ############
    # da = da.chunk({'x': 256, 'y': 256, 'layer': 1})
    # logger.info('---Rechunked datacube')  

    #######   SAVING ############
    output_path = extracted_path / f"{mask_code}.nc"
    da.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')
    
    logger.info(f'>>>>>>>>>>>  ds saved for= {extracted_path.name} bye bye >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

def create_event_datacube_copernicus(extracted_path, image_code, VERSION="v1"):
    '''
    Creates NetCDF datacube from extracted GeoTIFFs.
    
    Used for: TRAIN and TEST modes only
    INFERENCE: Skips this - tiles directly from normalized TIFFs, with MAKE_TIFFS=True
    
    An xarray dataset is created for the extracted_path folder and saved as a .nc file.
    '''
    logger.info(f'+++++++++++ IN CREAT EVENT  DATACUBE COPERNCUS in {extracted_path.name}+++++++++++++++++')
    # FIND THE EXTRACTED FOLDER
    logger.info(f'---image code= {image_code}')
    logger.info(f'---extracted folder = {extracted_path}')
    layerdict = make_layerdict(extracted_path)

    logger.info(f'---making das from layerdict= {layerdict}')

    dataarrays, layer_names = make_das_from_layerdict( layerdict, extracted_path)

    logger.info(f'---CHECKING DATAARRAY LIST')
    # check_dataarray_list(dataarrays, layer_names)
    logger.info(f'---dataarrays list = {dataarrays}') 
    logger.info(f'---CREATING CONCATERNATED DATASET')
    da = xr.concat(dataarrays, dim='layer').astype('float32')   
    da = da.assign_coords(layer=layer_names)

    # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
    if 'band' in da.dims and da.sizes['band'] == 1:
        # logger.info('---Squeezing out the "band" dimension')
        da = da.squeeze('band') 

    # logger.info_dataarray_info(da)

    #######   CHUNKING ############
    # da = da.chunk({'x': 256, 'y': 256, 'layer': 1})
    # logger.info('---Rechunked datacube')  

    #######   SAVING ############
    output_path = extracted_path / f"{image_code}.nc"
    da.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')
    
    logger.info(f'##################  ds saved in = {extracted_path.name} bye bye #################\n')

# WORK ON DEM
def match_dem_to_mask(sar_image, dem, output_path):
    """
    Matches the DEM to the SAR image grid by enforcing exact alignment of transform, CRS, and dimensions.
    """
    logger.info('+++in match_dem_to_sar fn')
    

    output_path.unlink(missing_ok=True)  # Deletes the file if it exists
    # Open the SAR image to extract its grid and CRS
    with rasterio.open(sar_image) as sar:
        sar_transform = sar.transform
        sar_crs = sar.crs
        logger.info(f"---SAR CRS: {sar_crs}")
        sar_width = sar.width
        sar_height = sar.height

    # Open the DEM to reproject and align it
    with rasterio.open(dem) as dem_src:
        logger.info(f"---DEM CRS: {dem_src.crs}")
        dem_meta = dem_src.meta.copy()
        # Update DEM metadata to match SAR grid
        dem_meta.update({
            'crs': sar_crs,
            'transform': sar_transform,
            'width': sar_width,
            'height': sar_height,
            'dtype': "float32"
        })
        with rasterio.open(output_path, 'w', **dem_meta) as dst:
            logger.info(f'---output_path= {output_path.name}')
            # Reproject each band of the DEM
            for i in range(1, dem_src.count + 1):
                reproject(
                    source=rasterio.band(dem_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=sar_transform,
                    dst_crs=sar_crs,
                    resampling=Resampling.nearest  # Nearest neighbor for discrete data like DEM
                )

    logger.info(f"Reprojected and aligned DEM saved to: {output_path}")

def create_slope_from_dem(target_file, dst_file):
    cmd = [
    "gdaldem", "slope", f"{target_file}", f"{dst_file}", "-compute_edges",
    "-p"
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Slope calculation completed.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error: {e}")

# CLIPPING AND ALIGNMENT
def clip_image_to_mask_gdal(input_raster, mask_raster, output_raster):

    # ensure everything is a plain string for GDAL
    input_raster  = str(input_raster)
    mask_raster   = str(mask_raster)
    output_raster = str(output_raster)

    # Open the mask to extract its bounding box
    mask_ds = gdal.Open(mask_raster)
    if mask_ds is None:
        raise FileNotFoundError(f"Mask file not found: {mask_raster}")
    mask_transform = mask_ds.GetGeoTransform()
    mask_proj = mask_ds.GetProjection()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize

    logger.info(f"---Mask dimensions: width={mask_width}, height={mask_height}")

    # Configure warp options to match the mask's resolution and extent
    options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(mask_transform[0], 
                      mask_transform[3] + mask_transform[5] * mask_height, 
                      mask_transform[0] + mask_transform[1] * mask_width, 
                      mask_transform[3]),
        xRes=mask_transform[1],  # Pixel size in X
        yRes=abs(mask_transform[5]),  # Pixel size in Y
        dstSRS=mask_proj,  # Match CRS
        resampleAlg="nearest",  # Nearest neighbor for categorical data
        outputBoundsSRS=mask_proj  # Ensure alignment in mask CRS
    )
    gdal.Warp(output_raster, input_raster, options=options)

    logger.info(f"Clipped raster saved to: {output_raster}")
    mask_ds = None  # Close the mask dataset
    # Delete the original SAR image
    Path(input_raster).unlink()


    logger.info(f"Clipped raster saved to: {output_raster}")

def clean_mask(mask_path, output_path):
        with rasterio.open(mask_path) as src:
            data = src.read(1)
            logger.info(f">>> Original mask stats: min={data.min()}, max={data.max()}, unique={np.unique(data)}")

            meta = src.meta.copy()
            # remove numbers greater than 1
            data[data > 1] = 0
            logger.info(f"--- Modified mask unique values: {np.unique(data)}")

            assert (data.min() == 0 and data.max() == 1) and len(np.unique(data)) == 2
            meta.update(dtype='uint8')  # Ensure uint8 format

            # Write the cleaned mask
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data.astype('uint8'), 1)  # Ensure uint8 format

        logger.info(f"Cleaned and aligned mask saved to: {output_path}")

def align_image_to_mask(sar_image, mask, aligned_image):
    logger.info('+++in align_image_to_mask fn')

    # Open the mask to get CRS, transform, and dimensions
    with rasterio.open(mask) as mask_src:
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform
        mask_width = mask_src.width
        mask_height = mask_src.height
        mask_res = (abs(mask_transform[0]), abs(mask_transform[4]))  # Ensure positive resolution
        logger.info('---mask ok')

    # Open the SAR image to calculate alignment
    with rasterio.open(sar_image) as sar_src:
        sar_meta = sar_src.meta.copy()
        transform, width, height = calculate_default_transform(
            sar_src.crs, mask_crs, sar_src.width, sar_src.height, *sar_src.bounds, resolution=mask_res
        )
        sar_meta.update({
            'crs': mask_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        logger.info('---sar ok')

        # Write the aligned SAR image
        with rasterio.open(aligned_image, 'w', **sar_meta) as aligned_dst:
            for i in range(1, sar_src.count + 1):
                reproject(
                    source=rasterio.band(sar_src, i),
                    destination=rasterio.band(aligned_dst, i),
                    src_transform=sar_src.transform,
                    src_crs=sar_src.crs,
                    dst_transform=transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.nearest
                )

    logger.info(f"---Aligned SAR image saved to: {aligned_image}")

#  TILING
def tile_geotiff_directly(vv_image: Path, vh_image: Path, output_path: Path, 
                         tile_size: int = 512, stride: int = 512) -> tuple:
    """
    Tile VV and VH GeoTIFFs directly without creating datacube.
    More efficient for inference.
    
    Returns:
        tiles: List of tile paths
        metadata: Dict with tile info for stitching
    """
    import rasterio
    from rasterio.windows import Window
    
    logger.info('\n\n++++++++++++++ IN TILE_GEOTIFF_DIRECTLY FN\n')

    tiles = []
    metadata = []
    
    # Open both images
    with rasterio.open(vv_image) as vv_src, rasterio.open(vh_image) as vh_src:
        # Verify they have same dimensions
        assert vv_src.shape == vh_src.shape, "VV and VH must have same dimensions"
        
        height, width = vv_src.shape
        profile = vv_src.profile.copy()
        profile.update(count=2, dtype='float32')  # 2 channels (VV, VH)
        logger.info(f'image dimensions: width={width}, height={height}')
        
        tile_idx = 0
        for y in range(0, height, stride):  # Changed: iterate through full height
            for x in range(0, width, stride):  # Changed: iterate through full width
                # Calculate actual tile bounds (handles edge tiles correctly)
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                actual_width = x_end - x
                actual_height = y_end - y
                
                # Create window with actual dimensions
                window = Window(x, y, actual_width, actual_height)
                
                # Read VV and VH for this window
                vv_data = vv_src.read(1, window=window)
                vh_data = vh_src.read(1, window=window)
                
                # Stack into 2-channel tile
                tile_data = np.stack([vv_data, vh_data], axis=0)
                
                # Create tile filename (using y, x for consistency with xarray version)
                tile_name = f"tile_{tile_idx:04d}_{y}_{x}.tif"
                tile_path = output_path / tile_name
                
                # Get transform for this window
                tile_transform = vv_src.window_transform(window)
                profile.update(transform=tile_transform, 
                              height=actual_height, 
                              width=actual_width)
                
                # Write tile
                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(tile_data)
                
                tiles.append(tile_path)
                
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)

                metadata.append({
                    'tile_name': tile_name,
                    'x_start': x,
                    'y_start': y,
                    'x_end': x_end,
                    'y_end': y_end,
                    'width': x_end - x,      # Direct calculation
                    'height': y_end - y      # Direct calculation
                })
                
                tile_idx += 1
        
        # Add global metadata for stitching
        # metadata['original_width'] = width
        # metadata['original_height'] = height
        # metadata['transform'] = list(vv_src.transform)
        # metadata['crs'] = str(vv_src.crs)
    
    return tiles, metadata

# probably not needed
def process_terraSARx_data(data_root):
    '''
    makes a 'datacube_files' folder in each event folder and copies the necessary files to it
    '''
    logger.info('+++in process_terraSARx_data fn')
    #image = list(Path('.').rglob("IMAGE_HH_*"))
    #logger.info('---image= ',image)

    target_filename = "DEM_MAP.tif"

    for event in data_root.iterdir():
        if event.is_dir() and any(event.iterdir()):
            logger.info(f"******* {event.name}   PREPARING TIFS ********")
            datacube_files_path = event / 'datacube_files'
            if datacube_files_path.exists() :
                shutil.rmtree(datacube_files_path)  # Delete the directory and all its contents

    
            datacube_files_path.mkdir(parents=True, exist_ok=True)
            pattern = re.compile(f'^{re.escape(target_filename)}$')

            # STEP THROUGH FILENAMES WE WANT 
            filename_parts = ['DEM_MAP', 'IMAGE_HH']
            for i in filename_parts:
                # logger.info(f'---looking for files starting with {i}')
                pattern = re.compile(f'^{re.escape(i)}')  

                # COPY THEM TO THE EVENT DATA CUBE FOLDER
                for file_path in Path(event).rglob("*"):
                    if file_path.suffixes == ['.tif'] and pattern.match(file_path.name):
                    # if file_path.suffix != '.aux.xml'and file_path.name == target_filename:
                    # if True:    
                        target = datacube_files_path / file_path.name
                        # if Path(target).exists():
                        #     logger.info('---file already exists') 
                        #     continue
                        shutil.copy(file_path, target)
            

