from pathlib import Path
import rasterio
import numpy as np
from rasterio.enums import Resampling

PAD_VAL = 1          # value used for padding in every band

def build_valid_mask(image_path: Path, out_path: Path):
    with rasterio.open(image_path) as src:
        # Read all bands as a 3-D array: shape (bands, rows, cols)
        arr = src.read(out_dtype="float32", resampling=Resampling.nearest)

        # valid = True where *any* band differs from PAD_VAL
        valid = np.any(arr != PAD_VAL, axis=0).astype(np.uint8)

        profile = src.profile
        profile.update(count=1, dtype="uint8", nodata=0)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(valid, 1)

def is_full_padding(img, msk, pad_val=1, var_eps=1e-4):
    """Return True if the whole tile is useless padding."""
    return (
        np.all(img == pad_val) and      # every image pixel = pad_val
        np.all(msk == pad_val) and      # every mask pixel = pad_val
        np.var(img)  < var_eps          # virtually zero variance
    )
    
# example
build_valid_mask(
    Path("scene_vv_vh_dem.tif"),
    Path("scene_valid_mask.tif")
)

src = PATH('/Users/alexwebb/Library/Mobile Documents/com~apple~CloudDocs/Documents/coding/floodai/UNOSAT_FloodAI_v2/data/2interim/TSX_TILES')