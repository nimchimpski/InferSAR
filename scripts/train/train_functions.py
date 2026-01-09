import torch
import numpy as np
import wandb
import sys
import signal
import random
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F

from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset , Dataset, DataLoader
from torch import Tensor, einsum
import segmentation_models_pytorch as smp
from scripts.train.train_helpers import nsd

logger = logging.getLogger(__name__)

def create_subset(mode, file_list, dataset_pth, stage,  subset_fraction , bs, num_workers, persistent_workers, input_is_linear):
    from scripts.train.train_classes import Sen1Dataset

    dataset = Sen1Dataset(mode, file_list, dataset_pth, input_is_linear )   
    subset_indices = random.sample(range(len(dataset)), int(subset_fraction * len(dataset)))
    subset = Subset(dataset, subset_indices)
    dl = DataLoader(subset, batch_size=bs, num_workers=num_workers, persistent_workers= persistent_workers,  shuffle = (stage == 'train'))
    return dl


# FOR INFERENCE / COMPARISON FN
def calculate_metrics(logits, masks, metric_threshold):
    """
    Calculate TP, FP, FN, TN, and related metrics for a batch of predictions.
    """
    # Initialize accumulators
    tps, fps, fns, tns = [], [], [], []
    nsds = []

    for logit, mask in zip(logits, masks):
        # metric predictions
        tp, fp, fn, tn = smp.metrics.get_stats(
            logit, mask.long(), mode='binary', threshold=metric_threshold
        )
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)

        # Compute Normalized Spatial Difference (NSD)
        nsd_value = nsd(
            logit[0].cpu().numpy() > metric_threshold,
            mask[0].cpu().numpy().astype(bool),
        )
        nsds.append(nsd_value)

    # Aggregate metrics
    tps = torch.vstack(tps).sum()
    fps = torch.vstack(fps).sum()
    fns = torch.vstack(fns).sum()
    tns = torch.vstack(tns).sum()

    return {
        "tps": tps,
        "fps": fps,
        "fns": fns,
        "tns": tns,
        "nsd_avg": np.mean(nsds)
    }
        

def handle_interrupt(signal, frame):
    '''
    usage: signal.signal(signal.SIGINT, handle_interrupt)
    '''
    logger.info("Interrupt received! Cleaning up...")
    # Add any necessary cleanup code here (e.g., saving model checkpoints)
    sys.exit(0)


def loss_chooser(loss_name, alpha=0.25, gamma=2.0, bce_weight=0.5, device=None):
    '''
    Returns a loss function for binary flood segmentation.
    
    Available loss functions:
        - "torch_bce": Standard PyTorch Binary Cross-Entropy with logits
        - "smp_bce": Segmentation Models BCE with ignore_index=255 support
        - "bce_dice": Weighted combination of BCE + Dice (recommended)
        - "focal": Focal Loss for hard example mining
    
    Loss function design considerations:
        - CLASS IMBALANCE: Addressed by pos_weight in BCE, naturally handled by Dice
        - HIGH RECALL: Dice loss emphasizes true positives (overlap between pred and target)
        - BOUNDARY ACCURACY: Both Dice and Focal Loss help with precise boundaries
        - IGNORE INVALID PIXELS: All losses support ignore_index=255 for no-data regions
    
    Parameters:
        loss_name (str): Name of the loss function to use
        alpha (float): Class weighting for Focal Loss (default 0.25)
        gamma (float): Focusing parameter for Focal Loss - higher = more focus on hard examples (default 2.0)
        bce_weight (float): Weight for BCE component in bce_dice combo (default 0.5)
                           E.g., 0.35 = 35% BCE + 65% Dice
        device (torch.device): Device to place loss function on (for pos_weight tensor)
    Returns:
        callable: Loss function that takes (predictions, targets) and returns scalar loss
    '''

    def bce_dice_valid(
        logits: torch.Tensor,          # [B,1,H,W] raw outputs
        labels: torch.Tensor,          # [B,1,H,W] 0/1 flood mask
        valid_mask: torch.Tensor,      # [B,1,H,W] 1 = valid pixel, 0 = ignore
        eps: float = 1e-6,
        ) -> torch.Tensor:
        """Weighted BCE + Dice loss that ignores pixels where valid_mask == 0."""

        # ----- BCE -----
        bce_map = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )                                           # [B,1,H,W]
        bce = (bce_map * valid_mask).sum() / (valid_mask.sum() + eps)

        # ----- Dice -----
        prob   = torch.sigmoid(logits)              # probabilities [0,1]
        prob   = prob   * valid_mask               # mask invalid
        label  = labels * valid_mask

        prob_f  = prob.view(prob.size(0), -1)       # flatten [B,N]
        label_f = label.view(label.size(0), -1)

        intersection = (prob_f * label_f).sum(1)
        union        = prob_f.sum(1) + label_f.sum(1)
        dice_loss    = 1 - (2 * intersection + eps) / (union + eps)
        dice         = dice_loss.mean()

        # ----- Combine -----
        return bce_weight * bce + (1.0 - bce_weight) * dice

    if loss_name == "torch_bce":
        torch_bce = torch.nn.BCEWithLogitsLoss()
        if device is not None:
            torch_bce = torch_bce.to(device)
        return torch_bce        
    if loss_name == "smp_bce":
        # Create pos_weight on the correct device
        pos_weight = torch.tensor([8.0])
        if device is not None:
            pos_weight = pos_weight.to(device)
        smp_bce =  smp.losses.SoftBCEWithLogitsLoss(ignore_index=255, reduction='mean', pos_weight=pos_weight)  # ignore_index=255 is used to ignore pixels where the mask is not valid (e.g., no data)
        if device is not None:
            smp_bce = smp_bce.to(device)
        return smp_bce
    if loss_name == "focal": # no weighting
        logger.info(f'---alpha: {alpha}, gamma: {gamma}---')  
        focal = smp.losses.FocalLoss(mode='binary', alpha=alpha, gamma=gamma)
        # Adjust alpha if one class dominates or struggles.
        # Adjust gamma to fine-tune focus on hard examples
        if device is not None:
            focal = focal.to(device)
        return focal
    if loss_name == "bce_dice":
        # Create pos_weight on the correct device
        pos_weight = torch.tensor([8.0])
        if device is not None:
            pos_weight = pos_weight.to(device)
        smp_bce = smp.losses.SoftBCEWithLogitsLoss(ignore_index=255, reduction='mean', pos_weight=pos_weight)
        if device is not None:
            smp_bce = smp_bce.to(device)
        
        dice = smp.losses.DiceLoss(mode='binary', from_logits=True, ignore_index=255)  # from_logits=True means the input is raw logits, not probabilities
        if device is not None:
            dice = dice.to(device)
        
        def bce_dice(preds, targets):
            preds_prob = torch.sigmoid(preds)  # Convert logits to probabilities for Dice Loss
            return bce_weight * smp_bce(preds, targets) + (1 - bce_weight) * dice(preds_prob, targets)
        logger.info(f'---loss chooser returning bce_dice with weight: {bce_weight}---')
        return bce_dice

    # THESE 3 NEED PREDS > PROBS
    # if loss_name == "dice":
    #     return dice
    # elif loss_name == "tversky": # no weighting
    #     return smp.losses.TverskyLoss()
    # elif loss_name == "jakard":
    #     return smp.losses.JaccardLoss() # penalize fp and fn. use with bce
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    

def wandb_initialization(job_type, repo_path, project, dataset_name, run_name, train_list, val_list, test_list, wandb_config, wandb_online):
    """
    Initialize W&B and return a WandbLogger for PyTorch Lightning.
    Handles dataset artifacts for 'train', 'test', and 'reproduce' jobs.
    """

    if  wandb_online:
        mode = 'online'
    else:
        mode = 'offline'
        logger.info(f"--- WandB is: {mode} ---")
    # Update parameters based on job type
    if job_type == "reproduce":
        artifact_dataset_name = f'unosat_emergencymapping-United Nations Satellite Centre/{project}/{dataset_name}/{dataset_name}'
    elif job_type == "debug":
        mode = 'disabled'


    # Initialize W&B run
    run = wandb.init(
        project=project,
        job_type=job_type,
        config=wandb_config,
        mode=mode,
        dir=repo_path / "results",
    )

    if job_type != 'reproduce':
        # Create and log dataset artifact
        data_artifact = wandb.Artifact(
            dataset_name,
            type="dataset",
            description=f"{dataset_name} - 12 events",
            metadata={
                "train_list": str(train_list),
                "val_list": str(val_list),
                "test_list": str(test_list),
            },
        )
        # Add references

        # turn your train_list (str or Path) into a proper file:// URI
        # logger.info(f">>> train_list: {train_list}")
        p = Path(train_list).expanduser().resolve()
        # logger.info(f">>> train_list resolved: {p}")
        train_list_uri = p.as_uri()
        # logger.info(f">>> train_list_uri: {train_list_uri}")
        # data_artifact = mlflow.data.DataArtifact()  # or however you construct it
        data_artifact.add_reference(train_list_uri, name="train_list")
        run.log_artifact(data_artifact, aliases=[dataset_name])

    elif job_type == 'reproduce':
        # Retrieve dataset artifact
        data_artifact = run.use_artifact(artifact_dataset_name)
        metadata_data = data_artifact.metadata
        logger.info(">>> Current Artifact Metadata:", metadata_data)

        # Update dataset paths
        train_list = Path(metadata_data.get('train_list', train_list))
        val_list = Path(metadata_data.get('val_list', val_list))
        test_list = Path(metadata_data.get('test_list', test_list))

        # Warn if any path is missing
        if not train_list.exists():
            logger.info(f"Warning: train_list path {train_list} does not exist.")
        if not val_list.exists():
            logger.info(f"Warning: val_list path {val_list} does not exist.")
        if not test_list.exists():
            logger.info(f"Warning: test_list path {test_list} does not exist.")

    # Return WandbLogger for PyTorch Lightning
    return WandbLogger(experiment=run)

              
def job_type_selector(job_type):

    train, test,  debug = False, False, False

    if job_type == "train":
        train = True
    elif job_type == "test":
        test = True
    elif job_type == "debug":
        debug = True

    return train, test, debug

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


    
# def initialize_wandb(project, job_type, run_name):
    """
    Initializes WandB if not already initialized.
    
    Args:
    - project (str): The name of the WandB project.
    - job_type (str): The type of job (e.g., 'train', 'reproduce').
    - run_name (str): The name of the WandB run.
    
    Returns:
    - wandb.run: The active WandB run.
    """
    # # Check if WandB is already initialized
    # if wandb.run is None:
    #     # Initialize WandB
    #     run = wandb.init(
    #         project=project,
    #         job_type=job_type,
    #         name=run_name
    #     )
    #     return run
    # else:
    #     # If already initialized, return the existing run
    #     return wandb.run
    

def plot_auc_pr(recall, precision, thresholds, best_index, best_threshold):


    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.scatter(recall[best_index], precision[best_index], color='red', label=f"Best Threshold: {best_threshold:.2f}")
    # for i, t in enumerate(thresholds):
    #     if i % 10 == 0:  # Mark every 10th threshold for clarity
    #         plt.annotate(f"{t:.2f}", (recall[i], precision[i]))
    # plt.show()
    return plt



    