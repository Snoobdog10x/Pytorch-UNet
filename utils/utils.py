import matplotlib.pyplot as plt
import os
from pathlib import Path
from csv import writer


def save_running(path: str, train_loss, validation_loss, valid_dice, epoch):
    Path(path).mkdir(parents=True, exist_ok=True)
    HEADER = ["Epoch", "Train loss", "Validation loss", "Validation dice"]
    List = [epoch, train_loss, validation_loss, valid_dice]
    if epoch == 1:
        with open(f'{path}/running.csv', 'w') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(HEADER)
            writer_object.writerow(List)
            f_object.close()
    else:
        with open(f'{path}/running.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_evaluate(save_checkpoint_path: str, img, pred, mask):
    fig, ax = plt.subplots(3, 3)
    ax[0][0].set_title('Input image')
    ax[0][1].set_title(f'predict')
    ax[0][2].set_title(f'truth')
    for i in range(3):
        ax[i][0].imshow(img[i].permute(1, 2, 0))
        ax[i][1].imshow(pred[i])
        ax[i][2].imshow(mask[i])
    if save_checkpoint_path != "":
        plt.savefig(save_checkpoint_path)
    plt.show()
