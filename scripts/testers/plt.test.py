import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


img1_pth = '/Users/alexwebb/laptop_coding/floodai/INFERSAR/data/4final/train_input/LabelHand/Bolivia_23014_LabelHand.tif.tif'  # Replace with your image path
img2_pth = '/Users/alexwebb/laptop_coding/floodai/INFERSAR/data/4final/train_input/S1Hand/Bolivia_23014_S1Hand.tif'  # Replace with your mask path


def main(self, img1_pth, img2_pth):
    cmap_cyan = ListedColormap(['black', 'cyan'])           
    fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # in inches
    axes[0].imshow(img1_pth.astype(np.uint8),cmap='OrRd')
    # axes[0].set_title('SAR Input (OrRd)')
    # 2) Probability heat-map
    # axes[1].imshow(prob_vis, cmap='inferno')
    # # axes[1].set_title('P(flood)')
    # # 3) Thresholded prediction (white=flood)
    # axes[2].imshow(pred_vis, cmap= cmap_cyan)
    # # axes[2].set_title('Pred >0.5')
    # # 4) Ground truth mask (white=flood)
    # axes[3].imshow(mask_vis, cmap='gray')
    # # axes[3].set_title('GT Mask')
    fig.subplots_adjust(bottom=0.2) # in %
    labels = ['SAR Input', 'P(flood)', 'Pred >0.5', 'GT Mask']

    for ax in axes:
        ax.axis('off') # Turn off axis
    plt.tight_layout()  
    examples.append(wandb.Image(fig, caption=f"Sample {i}"))