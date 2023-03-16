import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import utils as u
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import torch.nn as nn


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, fig_path: str):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    valid_loss = 0
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # compute the Dice score
                dice_score += dice_coeff((F.sigmoid(mask_pred) > 0.5).float(), mask_true, reduce_batch_first=False)
                loss = criterion(mask_pred.squeeze(1), mask_true.float())
                loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    reduce_batch_first=False)
                loss = criterion(mask_pred, mask_true)
                loss += dice_loss(
                    F.softmax(mask_pred, dim=1).float(),
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
            valid_loss += loss.item()
    np_images = image.cpu()[:3]
    np_mask_preds = mask_pred.argmax(dim=1).float().cpu()[:3]
    np_mask_trues = mask_true.float().cpu()[:3]
    u.plot_evaluate(fig_path, np_images, np_mask_preds,
                    np_mask_trues)
    net.train()
    return dice_score / max(num_val_batches, 1), valid_loss / max(num_val_batches, 1)
