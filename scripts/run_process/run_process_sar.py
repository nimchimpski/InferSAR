from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
import time
import os
import click
import logging, inspect
from rasterio.warp import calculate_default_transform, reproject, Resampling

from scripts.process.process_tiffs import create_event_datacube_TSX, clean_mask,  reproject_to_4326_fixpx_gdal, make_float32,  create_extent_from_mask,  clip_image_to_mask_gdal, create_valid_mask
from scripts.process.process_dataarrays import tile_datacube_rxr, compute_dataset_minmax
from scripts.process.process_helpers import  print_tiff_info_TSX, write_minmax_to_json, read_minmax_from_json, compute_dataset_minmax

start=time.time()

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

@click.command()
@click.option('--test', is_flag=True, help='loading from test folder', show_default=False)

def main(test=None):

    ############################################################################
    # data_src = Path(r"Y:\1NEW_DATA\1data\2interim\ALL TSX PROCESSING")
    repo_root = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR')
    events = repo_root / 'data' / '2interim' / 'events_extracted'
    data_src = Path('/Volumes/Lacie storage 6TB/SAR')

    logger.info(f'>>>data_src= {data_src}')

    if not test:
        logger.debug('>>>FULL DATASET MODE<<<')
        dataset =  data_src / 'SAR_process_INPUT' 
    else:
        logger.debug('>>>TEST MODE<<<')
        dataset =  data_src / 'SAR_process_TEST'
    
    make_tifs = 0
    make_datacubes = 0
    get_minmax = 0
    make_norm_tiles = 0
    norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
    minmax_path = repo_root / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'
    percent_non_flood = 1
    ############################################################################
    logger.info(f'\n>>>make_tifs= {make_tifs==1} \n>>>make_datacubes= {make_datacubes==1} \n>>>get minmax= {get_minmax==1} \n>>>make_tiles= {make_norm_tiles==1}')
    
    # ITERATE OVER THE DATASET
    # for folder in dataset.iterdir(): # ITER ARCHIVE AND CURRENT
    
    if make_tifs or make_datacubes:
        items = [d for d in dataset.iterdir() if d.is_dir()]
        # logger.debug(f'################### FOLDER={folder.name}  ###################')
        for event in tqdm(items, desc='processing events') : # ITER EVENT
            if event.is_dir():
                logger.info(f'################### EVENT={event.name}  ###################')
                # list event contents
                logger.debug(f'>>>event contents not including ds_store: {[x.name for x in event.iterdir() if x.name != ".DS_Store"]}') 
                orig_mask = list(event.rglob('*mask.tif'))[0]
                # GET REGION CODE FROM FOLDER
                mask_code = "_".join(orig_mask.parent.name.split('_')[:3])
                logger.debug(f'>>>mask_code= {mask_code}')

                # COPY  THE MASK, IMAGE, AND DEM TO THE EXTRACTED FOLDER
                logger.debug('\n>>>>>>>>>>>>>>>> making tiffs >>>>>>>>>>>>>>>>>')
                if make_tifs:
                    extract_folder = events / f'{mask_code}_extracted'
                    if extract_folder.exists():
                        shutil.rmtree(extract_folder)
                    extract_folder.mkdir(exist_ok=True)

                    # COPY THE MASK
                    ex_mask = extract_folder / f'{mask_code}_mask.tif'
                    shutil.copy(orig_mask, ex_mask)
                    logger.debug(f'>>>mask={ex_mask.name}')

                    ex_extent = extract_folder / f'{mask_code}_extent.tif'
                    create_extent_from_mask(ex_mask, ex_extent)

                    # copy the poly
                    # poly = list(event.rglob('*POLY*.kml'))[0]
                    # ex_poly = extract_folder / f'{mask_code}_poly.tif'
                    # shutil.copy(poly, ex_poly)
                    # COPY THE SAR IMAGE
                    image = list(event.rglob('*image*.tif') )[0]

                    logger.debug(f'>>>image={image.name}')
                    ex_image = extract_folder / f'{mask_code}_image.tif'
                    shutil.copy(image, ex_image)

                    # COPY THE DEM
                    # dem = list(event.rglob('*srtm*.tif'))[0]
                    # ex_dem = extract_folder / f'{mask_code}_dem.tif'
                    # logger.debug(f'>>>dem={ex_dem.name}')
                    # shutil.copy(dem, ex_dem)

                    #############################################

                    # logger.debug_tiff_info_TSX(ex_image, ex_mask)

                    #*****************************************************
                    # TODO ALL INPUTS SHOULD BE RESAMPLED TO EX. 2.5M/PX, TO MATCH THE RESAMPLING OF ALL INFERENCE INPUT IMAGES
                    #******************************************************

                    # REPROJECT THE TIFS TO EPSG:4326
                    logger.info('\n>>>>>>>>>>>>>>>> reproj all tifs to 4326 >>>>>>>>>>>>>>>>>')
                    reproj_image = extract_folder / f'{mask_code}_4326_image.tif'
                    # reproj_dem = extract_folder / f'{mask_code}_4326_dem.tif'
                    # reproj_slope = extract_folder / f'{mask_code}_4326_slope.tif'
                    reproj_mask = extract_folder / f'{mask_code}_4326_mask.tif'
                    # reproj_extent = extract_folder / f'{mask_code}_4326_extent.tif'

                    # orig_images = [ ex_image,  ex_mask]
                    # rep_images = [reproj_image,   reproj_mask]

                    # for i,j in zip( orig_images, rep_images):
                    #     logger.debug(f'---i={i.name} j={j.name}')
                    #     # check if mask or extent 
                    #     if 'extent' in i.name or 'mask' in i.name:
                    #         resampleAlg = 'near'
                    #     elif 'image' in i.name:
                    #         resampleAlg = 'bilinear'
                    #     logger.debug(f'---resampleAlg= {resampleAlg}')
                    #     reproject_to_4326_fixpx_gdal(i, j, resampleAlg, px_size=0.0001)

                    # if the image, mask are alreadt 4326 we ignore the above and just change the names
                    logger.debug('>>>tifs were already 4326, just changing names')
                    ex_image.rename(reproj_image)
                    ex_mask.rename(reproj_mask)

                    print_tiff_info_TSX(reproj_image)
                    print_tiff_info_TSX(reproj_mask)

                    # CLEAN THE MASK
                    logger.info('\n>>>>>>>>>>>>>>>> clean mask >>>>>>>>>>>>>>>>>')
                    cleaned_mask = extract_folder / f'{mask_code}_cleaned_mask.tif'
                    clean_mask(reproj_mask, cleaned_mask)
                    # delete reproj_mask
                    reproj_mask.unlink()  
                    # logger.debug('>>>TIFF CHECK 2')
                    # logger.debug_tiff_info_TSX(image=reproj_image, mask=cleaned_mask)

                    # CLIP THE IMAGE TO THE MASK
                    logger.info('\n>>>>>>>>>>>>>>>> clip image to mask >>>>>>>>>>>>>>>>>')
                    clipped_image = extract_folder / f'{mask_code}_clipped_image.tif'
                    logger.debug(f'>>>reproj_image= {reproj_image.name}')
                    logger.debug(f'>>>cleaned_mask= {cleaned_mask.name}')
                    clip_image_to_mask_gdal(reproj_image, cleaned_mask, clipped_image)
                    # logger.debug_tiff_info_TSX(clipped_image, cleaned_mask)
                     # MAKE VALID MASK
                    logger.info('\n>>>>>>>>>>>>>>>> make valid mask >>>>>>>>>>>>>>>>>')
                    
                    create_valid_mask(clipped_image, cleaned_mask, mask_code, extract_folder, inference=False)

                    mask_no_padding = extract_folder / 'mask_no_padding.tif'
                    # get values
                    # logger.debug_tiff_info_TSX(mask_no_padding)
                                    
                    # MAKE FLOAT32
                    logger.info('\n>>>>>>>>>>>>>>>> make image float32 >>>>>>>>>>>>>>>>>')
                    file_name = extract_folder / f'{mask_code}_final_image.tif'
                    final_image = make_float32(clipped_image, file_name)
                    # logger.debug_tiff_info_TSX(final_image, mask=cleaned_mask)
                    clipped_image.unlink()
                    
                    logger.info('\n>>>>>>>>>>>>>>>> make mask float32 >>>>>>>>>>>>>>>>>')
                    mask_w_nodata = extract_folder / 'mask_no_padding.tif'
                    final_mask = extract_folder / f'{mask_code}_final_mask.tif'
                    make_float32(mask_no_padding, final_mask)
                    cleaned_mask.unlink()

                    # logger.debug('\n>>>>>>>>>>>>>>>> make extent float32 >>>>>>>>>>>>>>>>>')
                    # final_extent = extract_folder / f'{mask_code}_final_extent.tif'
                    # make_float32(reproj_extent, final_extent)

                    print_tiff_info_TSX(final_image) 
                    print_tiff_info_TSX(final_mask) 
                    mask_w_nodata.unlink()
                    (extract_folder / 'valid_mask.tif').unlink()

                # CREATE AN EVENT DATA CUBE
                if make_datacubes:
                    logger.info('\n>>>>>>>>>>>>>>>> create 1 event datacube >>>>>>>>>>>>>>>>>')
                    # find folder with mask code in 'events'
                    event = events / f'{mask_code}_extracted'
                    create_event_datacube_TSX(event, mask_code)

    # CALCULATE MIN MAX
    if get_minmax:
        if minmax_path.exists():
            logger.debug(f'>>>deleting existing minmax file: {minmax_path}')
            # delete file
            os.remove(minmax_path)

        # else:
        logger.debug(f'>>>minmax file does not exist: getting vals')
        glob_min, glob_max = compute_dataset_minmax(dataset, band_to_read=1)
        logger.debug(f'>>>new gmin= {glob_min} gmax= {glob_max}')
        write_minmax_to_json(glob_min,glob_max, minmax_path)
        # CHECK THE NEW JSON
    minmax_data = read_minmax_from_json(minmax_path)
    logger.debug(f'>>>from json minmax now= {minmax_data}')
    minmax = (minmax_data['min'], minmax_data['max'])
    logger.debug(f'>>>minmax= {minmax}') 

    # MAKE NORMALIZED TILES
    if make_norm_tiles:
        total_num_tiles = 0
        total_saved = 0
        total_has_nans = 0
        total_novalid_layer = 0
        total_novalid_pixels = 0
        total_nomask_pixels = 0
        total_skip_nomask_pixels = 0
        total_failed_norm = 0
        total_num_not_256 = 0
        total_num_px_outside_extent = 0

        cubes = list(events.rglob("*.nc"))   
        logger.debug(f'>>>num cubes= ',len(cubes))
        for cube in tqdm(cubes, desc="### Datacubes tiled"):
            event_code = "_".join(cube.name.split('_')[:3])
            logger.info(f'\n>>>>>>>>>>>> cube >>>>>>>>>>>>>>>= {cube.name}')
            logger.debug(">>>event_code=", event_code)
            save_tiles_path = repo_root / 'data' / '3processed' / 'sar_tiles'  / f"{event_code}_normalized_tiles_{norm_func}_pcnf{percent_non_flood}"
            if save_tiles_path.exists():
                logger.debug(f"### Deleting existing tiles folder: {save_tiles_path}")
                # delete the folder and create a new one
                shutil.rmtree(save_tiles_path)
            save_tiles_path.mkdir(exist_ok=True, parents=True)


            logger.info(f"\n################### tiling ###################")
            # DO THE TILING AND GET THE STATISTICS
            num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask_pixels, num_skip_nomask_pixels, num_failed_norm , num_not_256, num_px_outside_extent= tile_datacube_rxr(cube, save_tiles_path, tile_size=256, stride=256, norm_func=norm_func, stats=minmax, percent_non_flood=percent_non_flood, inference=False)

            logger.debug(f'<<<  num_tiles=  {num_tiles}')
            logger.debug(f'<<< num_saved= {num_saved}')  
            logger.debug(f'<<< num_has_nans=  {num_has_nans}')
            logger.debug(f'<<< num_novalid_layer=  {num_novalid_layer}')
            logger.debug(f'<<< num_novalid_pixels=  {num_novalid_pixels}')
            logger.debug(f'<<< num_nomask pixels=  {num_nomask_pixels}')
            logger.debug(f'<<< num_failed_norm=  {num_failed_norm}')
            logger.debug(f'<<< num_not_256=  {num_not_256}')
            logger.debug(f'<<< num_px_outside_extent= {num_px_outside_extent}')

            total_num_tiles += num_tiles
            total_saved += num_saved
            total_has_nans += num_has_nans
            total_novalid_layer += num_novalid_layer
            total_novalid_pixels += num_novalid_pixels
            total_nomask_pixels += num_nomask_pixels
            total_skip_nomask_pixels += num_skip_nomask_pixels
            total_failed_norm += num_failed_norm
            total_num_not_256 += num_not_256     
            total_num_px_outside_extent += num_px_outside_extent     
            # END OF ONE CUBE TILE PROCESSING
        # END OF ALL CUBES TILE PROCESSING  
        logger.debug(f'>>>>total num of tiles: {total_num_tiles}')
        logger.debug(f'>>>>total saved tiles: {total_saved}')
        logger.debug(f'>>>total has  NANs: {total_has_nans}')
        logger.debug(f'>>>total no valid layer : {total_novalid_layer}')
        logger.debug(f'>>>total no valid  pixels : {total_novalid_pixels}')
        logger.debug(f'>>>total no mask pixels : {total_nomask_pixels}')
        logger.debug(f'>>>num failed normalization : {total_failed_norm}')
        logger.debug(f'>>>num not 256: {total_num_not_256}')
        logger.debug(f'>>>num px outside extent: {total_num_px_outside_extent}')
        logger.debug(f'>>>all tiles tally: {total_num_tiles == total_saved + total_has_nans + total_novalid_layer + total_novalid_pixels + total_skip_nomask_pixels + total_failed_norm + total_num_not_256 + total_num_px_outside_extent}')

    end = time.time()
    # time taken in minutes to 2 decimal places
    logger.debug(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == "__main__":
    main()





