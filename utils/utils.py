import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img.permute(1, 2, 0))
    ax[1].set_title(f'predict')
    ax[1].imshow(pred)
    ax[2].set_title(f'truth')
    ax[2].imshow(mask)
    if save_checkpoint_path != "":
        plt.savefig(save_checkpoint_path)
    plt.show()
