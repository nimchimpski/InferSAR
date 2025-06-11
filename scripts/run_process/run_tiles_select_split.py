from pathlib import Path
import sys
import os
from tqdm import tqdm
import click
import shutil
import signal
import logging
from scripts.process.process_dataarrays  import  make_train_folders, get_incremental_filename, select_tiles_and_split
from scripts.process.process_helpers import handle_interrupt

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--test",is_flag=True)
def main(test=None):
    '''
    with MAKEFOLDER=False this can be used to asses how many tiles will be filtered out by the analysis and mask threshold before actually running the split.
    '''
    signal.signal(signal.SIGINT, handle_interrupt)

    repo_dir = Path(__file__).resolve().parent.parent.parent
    logger.debug(f'>>>repo_dir= {repo_dir}')
    src_base = repo_dir / 'data' / '3processed' / 'sar_tiles' / 'NORM_TILES_FOR_SELECT_AND_SPLIT_INPUT'
    dataset_name = None
    ############################################
    MAKEFOLDER = True
    analysis_threshold=1
    mask_threshold=0.3
    percent_under_thresh=0.25 # 0.01 = 1% 

    dst_base = repo_dir / 'data' / '4final' / 'train_INPUT'
    if test:
        click.echo("TEST DESTINATION")
        dst_base = repo_dir / 'data' / '4final' / 'test_INPUT'
    if not dst_base.exists():
        raise FileNotFoundError(f"Destination folder {dst_base} does not exist.")
    logger.debug(f"Destination folder: {dst_base}")
  

    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    if test:
        train_ratio=0.001
        val_ratio=0.001
        test_ratio=0.998
    ########################################

    total = 0
    rejected = 0
    tot_missing_extent = 0
    tot_missing_mask = 0
    tot_under_thresh = 0
    low_selection = []

    # GET EVENT FOLDER NAME
    folders_to_process = list(f for f  in iter(src_base.iterdir()) if f.is_dir())
    # folder_to_process = folders_to_process[0]
    # logger.debug(f'>>>folder_to_process= {folder_to_process.name}')
    if len(folders_to_process) == 0:
        logger.debug(">>>No event folder found.")
        
    elif len(folders_to_process) > 1:
        logger.debug(">>>Multiple event folders found.")
        
    # else:
    #     src_tiles = folders_to_process[0]
    #     logger.debug(f'>>>src_tiles_name= {src_tiles.name}')
    
    # parts = src_tiles.name.split('_')[:3]
    # logger.debug(f'>>>newname= {parts}')
    # newname = '_'.join(parts)
    # logger.debug(f'>>>newname= {newname}')


    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(src_base.rglob('*normalized*'))
    logger.debug(f'>>>len recursive_list= {len(recursive_list)}')
    if not recursive_list:
        logger.debug(">>>No normalized folders found.")
        return
    
    # FILTER AND SPLIT
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        # GET NUMBER OF FILES IN FOLDER

        dest_dir = get_incremental_filename(dst_base, f'{folder.name}_mt{mask_threshold}_pcu{percent_under_thresh}')

        logger.debug(f'>>>source dir = {src_base}')
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f'>>>dest dir: {dest_dir}')   
        if not dest_dir.exists():
            logger.debug(f"Failed to create {dest_dir}")
            return
        logger.debug('>>>mask threshold:', mask_threshold)
        logger.debug('>>>analysis threshold:', analysis_threshold)
        make_train_folders(dest_dir)

        logger.debug(f"\n>>>>>>>>>>>>>>>>>>> AT FOLDER {folder.name}>>>>>>>>>>>>>>>")
        foldertotal, folder_selected, folderrejected, folder_missing_extent, folder_missing_mask, folder_under_thresh = select_tiles_and_split(folder, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, percent_under_thresh, MAKEFOLDER)
        logger.debug(f'>>>folder total= {foldertotal}')
        logger.debug(f'>>>folder selected= {folder_selected}')
        logger.debug(f'>>>folder rejected= {folderrejected}')
        logger.debug(f'>>>folder under threshold= {folder_under_thresh}')
        if folder_selected < 10:
            low_selection.append(folder.name)

        total += foldertotal
        rejected += folderrejected   
        tot_missing_mask += folder_missing_mask
        tot_missing_extent += folder_missing_extent 
        tot_under_thresh += folder_under_thresh    
 
        logger.debug(f">>>subtotal tiles: {total}")
        logger.debug(f">>>subtotal Rejected tiles: {rejected}")
        # logger.debug(f">>>subtotal missing extent: {tot_missing_extent}")
        # logger.debug(f">>>subtotal missing mask: {tot_missing_mask}")
        logger.debug(f">>>subtotal under threshold: {tot_under_thresh}")
        
    logger.debug('>>>>>>>>>>>>>>>> DONE >>>>>>>>>>>>>>>>>')
    with open(dest_dir / "train.txt", "r") as traintxt,  open(dest_dir / "val.txt", "r") as valtxt,  open(dest_dir / "test.txt", "r") as testtxt:
        traintxtlen = sum(1 for _ in traintxt)
        logger.debug('>>>len traint.txt= ', traintxtlen)
        valtxtlen = sum(1 for _ in valtxt)
        logger.debug('>>>len val.txt= ', valtxtlen)
        testtxtlen = sum(1 for _ in testtxt)
        logger.debug('>>>len test.txt= ', testtxtlen) 

    # logger.debug(">>>total txt len = ", traintxtlen + valtxtlen + testtxtlen)
    trainsize = len(list((dest_dir / 'train').iterdir()))
    valsize = len(list((dest_dir / 'val').iterdir()))
    testsize = len(list((dest_dir / 'test').iterdir()))
    selected_tiles = trainsize + valsize + testsize
    logger.debug(f">>>Total all original tiles: {total}")
    logger.debug(f'>>>total t+v+t tiles: {selected_tiles}')
    logger.debug(f">>>Rejected tiles: {rejected}")
    logger.debug(f'>>>total under threshold included: {tot_under_thresh}')
    logger.debug(f'>>>total irrelevant files: {total - rejected - selected_tiles}')
    logger.debug(">>>tiles and texts match= ", (trainsize + valsize + testsize)== traintxtlen + valtxtlen + testtxtlen)
    
    
    # logger.debug(f">>>Total missing extent: {tot_missing_extent}")
    # logger.debug(f">>>Total missing mask: {tot_missing_mask}")
    logger.debug(f'>>>list of low selection folders: {low_selection}') 
    if MAKEFOLDER:
        logger.debug(f"Saved split data to {dest_dir}")
    else:
        logger.debug("NO TILES MADE - test run")

if __name__ == "__main__":
    main()