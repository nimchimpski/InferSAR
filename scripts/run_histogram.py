import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("Running histogram script...")
input_pth = Path('/Users/alexwebb/laptop_coding/floodai/InferSAR/data/4final/dataset/S1Hand/Bolivia_23014_S1Hand.tif')

with rasterio.open(input_pth) as src:
    vv = src.read(1)
    print(f"Unique values: {np.unique(vv)[:20]}")  # First 20 unique values
    plt.hist(vv.flatten(), bins=100)
    plt.show()