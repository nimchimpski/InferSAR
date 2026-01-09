import logging
import sys
from pathlib import Path
'''
writes the vals to .json.
skips any tiles that are all nans.
Consider adding headroom of 1-2 units to max to avoid clipping to the .json file
'''
# Add project directory to Python path for imports
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from scripts.process.process_helpers import compute_traintiles_minmax, write_minmax_to_json

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    """
    Compute the min and max values for each band in a dataset.
    """
    # Define the dataset path
    dataset_path = project_path / 'data' / '4final' / 'dataset' / 'S1Hand'  # Replace with your actual dataset path

    # Compute min and max values
    globmin, globmax = compute_traintiles_minmax(dataset_path)
    print(f"Raw Global Min: {globmin}, Global Max: {globmax}")
    globmin=globmin - 1
    globmax=globmax + 1

    # Print the results
    print(f"Global Min-1: {globmin}")
    print(f"Global Max+1: {globmax}")

    output_path= project_path / 'configs' / 'global_minmax_INPUT' / 'global_minmax.json'
    write_minmax_to_json(int(globmin), int(globmax), output_path)


if __name__ == "__main__":
    main()