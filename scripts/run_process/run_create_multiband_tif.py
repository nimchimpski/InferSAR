from pathlib import Path
import rasterio
import numpy as np

dataset = Path('/path/to/dataset')  
# images = *S1Hand.tif, masks = *LabelHAnd.tif
images = list(dataset.glob('*S1Hand.tif'))
masks = list(dataset.glob('*LabelHand.tif'))

if len(images) != len(masks):
    raise ValueError(f"Number of images ({len(images)}) does not match number of masks ({len(masks)})")

for image, mask in zip(images, masks):
    # Create a new filename for the multiband TIFF
    new_filename = image.stem + '_multiband.tif'
    new_filepath = image.parent / new_filename
    
    # Open the image and mask files
    with rasterio.open(image) as img_src, rasterio.open(mask) as mask_src:
        # Read the data from both files
        img_data = img_src.read()
        mask_data = mask_src.read(1)  # Read the mask as a single band
        
        # Stack the image and mask data along a new dimension
        multiband_data = np.stack((img_data, mask_data), axis=0)
        
        # Update metadata for the new file
        profile = img_src.profile.copy()
        profile.update(count=multiband_data.shape[0], dtype=rasterio.float32)
        
        # Write the multiband TIFF
        with rasterio.open(new_filepath, 'w', **profile) as dst:
            dst.write(multiband_data.astype(np.float32))
    
    print(f"Created multiband TIFF: {new_filepath}")