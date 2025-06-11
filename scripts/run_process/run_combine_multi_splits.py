from pathlib import Path
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def combine_datasets(
    input_dirs,            # list of str or Path: folders to merge
    output_dir,            # str or Path: where to put the combined dataset
    splits=("train","val","test"),  # names of subfolders & list-files
    copy_files=True        # if False, will try to create symlinks instead
):
    """
    Merge multiple dataset folders into one.

    Each input_dir is expected to have:
        input_dir/<split>/           # a folder of files for that split
        input_dir/<split>.txt        # a text file listing items in that split

    The function will create under output_dir:
        output_dir/<split>/          # with all files copied/linked
        output_dir/<split>.txt       # merged, deduped list of all lines

    Returns None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare merged lists
    merged = {split: set() for split in splits}

    for split in splits:
        (output_dir / split).mkdir(exist_ok=True)

    for in_dir in map(Path, input_dirs):
        for split in splits:
            src_folder = in_dir / split
            if src_folder.is_dir():
                for src_file in src_folder.iterdir():
                    if not src_file.is_file():
                        continue
                    dest = output_dir / split / src_file.name
                    if not dest.exists():
                        if copy_files:
                            shutil.copy2(src_file, dest)
                        else:
                            # create a relative symlink
                            rel = src_file.relative_to(output_dir / split)
                            dest.symlink_to(rel)

            # merge the accompanying .txt list
            list_file = in_dir / f"{split}.txt"
            if list_file.is_file():
                for line in list_file.read_text().splitlines():
                    line = line.strip()
                    if line:
                        merged[split].add(line)

    # write out the merged lists
    for split, items in merged.items():
        out_list = output_dir / f"{split}.txt"
        with open(out_list, "w") as f:
            for item in sorted(items):
                f.write(f"{item}\n")

def main():
    root = Path("/Users/alexwebb/laptop_coding/floodai/UNOSAT_FloodAI_v2/data/4final")
    src = root / 'to_combine'
    input_dirs = [p for p in src.iterdir()]
    
    output_dir = root / 'train_input'

    combine_datasets(input_dirs, output_dir)
if __name__ == "__main__":
    main()

    print("Datasets combined successfully.")
