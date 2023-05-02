import matplotlib.pyplot as plt
import os
from pathlib import Path
from csv import writer
from pandas import *
import numpy as np


def plot_running(path: str):
    data = read_csv(path)
    epochs = data['Epoch'].tolist()
    np_epochs = np.arange(1, len(epochs) + 1, 1)
    path = path[:-4]
    plot_and_save_running(path, "train vs valid loss", np_epochs,
                          [data["Train_loss"].tolist(), data["Validation_loss"].tolist()], ["train", "valid"])
    plot_and_save_running(path, "train vs valid accuracy", np_epochs,
                          [data["Train_dice"].tolist(), data["Validation_dice"].tolist()]
                          , ["train, valid"])
    plot_and_save_running(path, "Learning rate", np_epochs,
                          [data["Learning_rate"].tolist()])


def plot_and_save_running(path, title: str, np_epoch, data: [], line_label: [] = None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    for i, value in enumerate(data):
        if line_label is not None:
            if line_label[i] is not None:
                ax.plot(np_epoch, value, label=line_label[i])
                continue
        ax.plot(np_epoch, value)
    if line_label is not None:
        ax.legend()
    save_path = f"{path}_{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()


def save_running_csv(path: str, HEADER: [], values: [], is_first: bool):
    assert len(HEADER) == len(values), "same size header and values"
    if is_first:
        with open(f'{path}/running.csv', 'w') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(HEADER)
            writer_object.writerow(values)
            f_object.close()
    else:
        with open(f'{path}/running.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(values)
            f_object.close()
    return f'{path}/running.csv'


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
    img_len = len(img)
    fig, ax = plt.subplots(img_len, 3)
    if img_len > 1:
        ax[0][0].set_title('Input image')
        ax[0][1].set_title(f'predict')
        ax[0][2].set_title(f'truth')
    else:
        ax[0].set_title('Input image')
        ax[1].set_title(f'predict')
        ax[2].set_title(f'truth')
    for i in range(img_len):
        if img_len > 1:
            ax[i][0].imshow(img[i].permute(1, 2, 0))
            ax[i][1].imshow(pred[i])
            ax[i][2].imshow(mask[i])
        else:
            ax[0].imshow(img[i].permute(1, 2, 0))
            ax[1].imshow(pred[i])
            ax[2].imshow(mask[i])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if save_checkpoint_path != "":
        plt.savefig(save_checkpoint_path)
    # plt.close()
    plt.show()
    plt.close()
