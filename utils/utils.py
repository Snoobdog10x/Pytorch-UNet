import matplotlib.pyplot as plt
import os
from pathlib import Path
from csv import writer
from pandas import *
import numpy as np


def plot_running(path: str):
    data = read_csv(path)
    # converting column data to list
    epochs = data['Epoch'].tolist()
    np_epochs = np.arange(1, len(epochs) + 1, 1)
    train_loss = data['Train_loss'].tolist()
    valid_loss = data['Validation_loss'].tolist()
    valid_dice = data['Validation_dice'].tolist()
    learning_rate = data['Learning_rate'].tolist()
    fig, ax = plt.subplots(2, 2)
    ax[0][0].set_title('Train loss')
    ax[0][1].set_title(f'Valid loss')
    ax[1][0].set_title(f'Accuracy')
    ax[1][1].set_title(f'Learning rate')
    ax[0][0].plot(epochs, train_loss)
    ax[0][0].set_xticks(np_epochs)
    ax[0][1].plot(epochs, valid_loss)
    ax[0][1].set_xticks(np_epochs)
    ax[1][0].plot(epochs, valid_dice)
    ax[1][0].set_xticks(np_epochs)
    ax[1][1].plot(epochs, learning_rate)
    ax[1][1].set_xticks(np_epochs)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f"{path[:-4]}.png")


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
