import argparse
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet
from utils.utils import *
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torch.utils.data import DataLoader


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


def evaluate_test(net, device, test_dir, output_viz, out_threshold=0.5):
    net.eval()
    imgs_dir = os.path.join(test_dir, "imgs")
    mask_dir = os.path.join(test_dir, "masks")
    try:
        dataset = CarvanaDataset(imgs_dir, mask_dir, 1)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(imgs_dir, mask_dir, 1)
    loader_args = dict(batch_size=16, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    num_test_batches = len(test_loader)
    batch_dice_scores = []
    total_dict_score = 0
    loader_bar = tqdm(test_loader, total=num_test_batches, desc='Validation round', unit='batch', leave=False)
    with torch.no_grad():
        for index, batch in enumerate(loader_bar, start=1):
            # get_image_and_reshape
            image, mask_true = batch["image"], batch["mask"]
            # image, mask_true = image.unsqueeze(0), mask_true.unsqueeze(0)
            # convert to cpu, float 32
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)
            if net.n_classes > 1:
                dice_score = multiclass_dice_coeff(
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                    reduce_batch_first=False)
            else:
                dice_score = dice_coeff((F.sigmoid(mask_pred) > 0.5).float(), mask_true, reduce_batch_first=False)
            total_dict_score += dice_score.item()
            batch_dice_scores.append(dice_score.item())
            logging.info(f'\nBatch {index} Dice score: {dice_score.item()}')
            save_running_csv(output_viz, ["Batches", "Dice_score"], [index, dice_score.item()], index == 1)
        logging.info(f'\nAvg Dice score: {total_dict_score / max(num_test_batches, 1)}')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='trained_models/model_1/best.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', type=str, default="data/test", help='root data path')
    parser.add_argument('--output', '-o', type=str, default="test_evaluate", help='root data path')
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
    evaluate_test(net, device, in_files, out_files)
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
