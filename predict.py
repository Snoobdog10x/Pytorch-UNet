import argparse
import logging
import os
from utils.data_loading import BasicDataset, CarvanaDataset
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
from utils import utils as u
from pathlib import Path
from torch.utils.data import DataLoader, random_split


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def evaluate_test(net, device, test_dir, output_viz, img_scale):
    try:
        dataset = CarvanaDataset(f"{test_dir}/imgs", f"{test_dir}/masks", img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(f"{test_dir}/imgs", f"{test_dir}/masks", img_scale)
    num_val_batches = len(dataset)
    loader_args = dict(batch_size=16, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    dice_score = 0
    count = 1
    batch_num = []
    dice_scores = []
    output_path = Path(output_viz)
    output_path.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        for batch in tqdm(test_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # compute the Dice score
                batch_dice = dice_coeff((F.sigmoid(mask_pred) > 0.5).float(), mask_true, reduce_batch_first=False)
                dice_score += batch_dice
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # compute the Dice score, ignoring background
                batch_dice = multiclass_dice_coeff(
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    reduce_batch_first=False)
                dice_score += batch_dice
            batch_num.append(count)
            dice_scores.append(batch_dice)
            count += 1
            np_images = image.cpu()[:3]
            np_mask_preds = mask_pred.argmax(dim=1).float().cpu()[:3]
            np_mask_trues = mask_true.float().cpu()[:3]
            fig_path = f"{output_viz}/test_images"
            Path(fig_path).mkdir(parents=True, exist_ok=True)
            u.plot_evaluate(f"{fig_path}_batch_{count}.jpg", np_images, np_mask_preds,
                            np_mask_trues)
        avg_dice_score = dice_score / max(num_val_batches, 1)
        logging.info(f'avg dice score {avg_dice_score}')
        u.plot_and_save_running(f"{output_viz}", f"Test accuracy {avg_dice_score}",
                                batch_num, batch_dice)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', type=str, default="data", help='root data path')
    parser.add_argument('--output', '-o', type=str, default="data", help='root data path')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_files = args.input
    # out_files = get_output_filenames(args)
    out_files = args.output
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    evaluate_test(net, device, in_files, out_files, args.scale)
    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename)
    #
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
    #
    #     if not args.no_save:
    #         out_filename = out_files[i]
    #         result = mask_to_image(mask, mask_values)
    #         result.save(out_filename)
    #         logging.info(f'Mask saved to {out_filename}')
    #
    #     if args.viz:
    #         logging.info(f'Visualizing results for image {filename}, close to continue...')
    #         plot_img_and_mask(img, mask)
