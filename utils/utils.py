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
    plot_and_save_running(path, "accuracy", np_epochs,
                          [data["Validation_dice"].tolist()])
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


def save_running_csv(path: str, train_loss, validation_loss, valid_dice, learning_rate, epoch):
    HEADER = ["Epoch", "Train_loss", "Validation_loss", "Validation_dice", "Learning_rate"]
    List = [epoch, train_loss, validation_loss, valid_dice, learning_rate]
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
    fig, ax = plt.subplots(3, 3)
    ax[0][0].set_title('Input image')
    ax[0][1].set_title(f'predict')
    ax[0][2].set_title(f'truth')
    for i in range(3):
        ax[i][0].imshow(img[i].permute(1, 2, 0))
        ax[i][1].imshow(pred[i])
        ax[i][2].imshow(mask[i])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if save_checkpoint_path != "":
        plt.savefig(save_checkpoint_path)
    # plt.show()
