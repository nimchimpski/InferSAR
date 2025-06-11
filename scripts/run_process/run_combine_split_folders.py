from pathlib import Path
import shutil
import logging
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def combine_txt_files(txt_file1, txt_file2, output_file):
    """
    Combine two .txt files into one, removing duplicates if necessary.
    """
    logger.info('+++in combine_txt_files+++')

    # Read entries from both files
    with open(txt_file1, "r") as f1, open(txt_file2, "r") as f2:
        entries1 = f1.readlines()
        entries2 = f2.readlines()
    
    # Combine and remove duplicates
    combined_entries = list(set(entries1 + entries2))
    
    # Save to the new file
    with open(output_file, "w") as out:
        out.writelines(sorted(combined_entries))  # Sort for consistency

def combine_datasets(dataset1_path, dataset2_path, output_path):
    """
    Combine train/val/test splits and their corresponding .txt files,
    ensuring the original files are preserved.
    """
    logger.info('+++in combine_datasets+++')
    logger.debug(f"---Combining datasets from {dataset1_path} and {dataset2_path} into {output_path}")
    splits = ["train", "val", "test"]
    
    for split in splits:
        logger.debug(f"---Processing split: {split}")
        # Paths to the .txt files
        txt_file1 = dataset1_path / f"{split}.txt"
        txt_file2 = dataset2_path / f"{split}.txt"
        output_txt = output_path / f"{split}.txt"
        
        # Combine .txt files
        combine_txt_files(txt_file1, txt_file2, output_txt)
        logger.debug(f"---Combined {split}.txt saved to {output_txt}")
        
        # Combine image/tiles directories
        split_dir1 = Path(dataset1_path) / split
        split_dir2 = Path(dataset2_path) / split
        output_split_dir = Path(output_path) / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files from both datasets
        items1 = list(split_dir1.glob("*"))
        items2 = list(split_dir2.glob("*"))
        for file in tqdm(items1, desc=f"Copying {split} files"):
            dest_file = output_split_dir / file.name
            if not dest_file.exists():
                shutil.copy(file, dest_file)  # Copy file to the destination
            else:
                # Handle duplicates by renaming
                new_name = f"{file.stem}_copy{file.suffix}"
                shutil.copy(file, output_split_dir / new_name)
        
        for file in tqdm(items2, desc=f"Copying {split} files"):
            dest_file = output_split_dir / file.name
            if not dest_file.exists():
                shutil.copy(file, dest_file)  # Copy file to the destination
            else:
                # Handle duplicates by renaming
                new_name = f"{file.stem}_copy{file.suffix}"
                shutil.copy(file, output_split_dir / new_name)
        logger.debug(f"---Combined {split} directory saved to {output_split_dir}")

def main():
    ##################################
    combine_folder = Path("/Users/alexwebb/laptop_coding/floodai/UNOSAT_FloodAI_v2/data/4final/to_combine")
    output_path = Path("/Users/alexwebb/laptop_coding/floodai/UNOSAT_FloodAI_v2/data/4final/train_input")
    if not (combine_folder.exists()) or not (combine_folder.is_dir()):
        logger.debug(f" folder {combine_folder} does not exist.")
        return
    ##################################
    # get folders in the combine folder
    combine_folders = [f for f in combine_folder.iterdir() if f.is_dir()]
    logger.debug(f">>>Combine folders: {combine_folders}")

    combine_datasets(
    dataset1_path = combine_folders[0] ,
    dataset2_path = combine_folders[1],
    output_path = output_path
    )

if __name__ == "__main__":
    main()