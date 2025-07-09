import logging
from pathlib import Path
from scripts.process.process_helpers import compute_traintiles_minmax

logging.basicConfig(
    level=logging.INFO,                            # DEBUG, INFO,[ WARNING,] ERROR, CRITICAL
    format=" %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

print('hello')
def main():
    """
    Compute the min and max values for each band in a dataset.
    """
    # Define the dataset path
    dataset_path = Path("/Users/alexwebb/laptop_coding/floodai/UNOSAT_FloodAI_v2/data/4final/to_combine/test_tiles_minmax" ) # Replace with your actual dataset path

    # Compute min and max values
    globmin, globmax = compute_traintiles_minmax(dataset_path)

    # Print the results
    logger.info(f"Global Min: {globmin}")
    logger.info(f"Global Max: {globmax}")

if __name__ == "__main__":
    main()