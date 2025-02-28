import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from unet import UNetLite
from unet import UNetSmall
from utils.best_checker import BestChecker
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from utils.early_stopper import EarlyStopper
from utils.utils import save_running_csv, plot_running

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        early_stop_patient=10,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        save_epoch_plot: bool = True,
        n_channel=1,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, n_channel=n_channel)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, n_channel=n_channel)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
    # (Initialize logging)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Data path:       {root_data_path}
        Checkpoint path: {dir_checkpoint}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Save epoch plot: {save_epoch_plot}
        Image channel: {n_channel}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    best = Path(dir_checkpoint).joinpath(f"best")
    best.mkdir(parents=True, exist_ok=True)
    early_stopper = EarlyStopper(patience=early_stop_patient, verbose=True, path=best.joinpath('best.pth'),
                                 dataset=dataset)
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        train_dice = 0
        num_train_batches = len(train_loader)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                if device.type == 'mps':
                    amp_device = device.type
                else:
                    amp_device = "cpu"

                with torch.autocast(amp_device, enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        train_dice += dice_coeff((F.sigmoid(masks_pred) > 0.5).float(), true_masks,
                                                 reduce_batch_first=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        train_dice += multiclass_dice_coeff(
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                            F.one_hot(masks_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                            reduce_batch_first=False)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        epoch_dice = train_dice / max(num_train_batches, 1)
        logging.info('Validation')
        fig_path = ""
        epoch_path = dir_checkpoint.joinpath("validation_image")
        if save_epoch_plot:
            epoch_path.mkdir(parents=True, exist_ok=True)
            fig_path = str(epoch_path / f"checkpoint_epoch{epoch}.jpg")
        val_score, val_loss = evaluate(model, val_loader, device, amp, fig_path)

        logging.info('Validation Dice score: {}'.format(val_score))
        train_loss = epoch_loss / max(len(train_loader), 1)
        values = [epoch, train_loss, val_loss, epoch_dice.item(), val_score.item(), learning_rate]
        HEADER = ["Epoch", "Train_loss", "Validation_loss", "Train_dice", "Validation_dice", "Learning_rate"]
        running_path = save_running_csv(dir_checkpoint, HEADER, values, epoch == 1)
        plot_running(running_path)
        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            latest = Path(dir_checkpoint).joinpath(f"latest")
            latest.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict,
                       latest.joinpath('latest.pth'))
            logging.info(f'Checkpoint {epoch} saved!')
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', '-dp', type=str, default="data/split_data", help='root data path')
    parser.add_argument('--check_point_path', '-cpp', type=str, default="checkpoints", help='checkpoint model path')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--model_type', '-mt', type=str, default="NORMAL",
                        help='choose model type: NORMAL or LITE or SMALL')
    parser.add_argument('--save_epoch_plot', '-sep', type=bool, default=True,
                        help='Save plot image after epoch')
    parser.add_argument('--xla', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--early_stop_patient', '-esp', type=int, default=10, help='Early stop patient')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channel', '-cn', type=int, default=3,
                        help='Number of channel image 1 for gray, and 3 for color')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    n_channel = args.channel
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model_type == "NORMAL":
        model = UNet(n_channels=n_channel, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_type == "LITE":
        model = UNetLite(n_channels=n_channel, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNetSmall(n_channels=n_channel, n_classes=args.classes, bilinear=args.bilinear)

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\tmodel size {args.model_type}\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if "mask_values" in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    root_data_path = args.data_path
    dir_img = Path(f'{root_data_path}').joinpath("imgs")
    dir_mask = Path(f'{root_data_path}').joinpath("masks")
    dir_checkpoint = Path(args.check_point_path)
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            early_stop_patient=args.early_stop_patient,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_epoch_plot=args.save_epoch_plot,
            amp=args.amp,
            n_channel=n_channel,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            early_stop_patient=args.early_stop_patient,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_epoch_plot=args.save_epoch_plot,
            amp=args.amp,
            n_channel=n_channel,
        )
