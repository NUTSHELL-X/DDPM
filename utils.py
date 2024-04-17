import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_images(images, max_num=16, n_col=4, save_path=None):
    if isinstance(images, torch.Tensor):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 2, 3, 1)
        images = images.numpy()
    elif isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)

    image_num = min(images.shape[0], max_num)
    assert n_col >= 1
    n_row = np.ceil(image_num / n_col).astype(np.int8)
    for i in range(image_num):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i])
    plt.show()
    if save_path:
        plt.savefig(save_path)


def save_image_seq(images, save_folder, image_name_prefix):
    # images : list or tuple if torch tensor (c,h,w) or torch tensor (n,c,h,w)
    if isinstance(images, list) or isinstance(images, tuple):
        for i in range(len(images)):
            save_image(
                images[i].detach().cpu().to(torch.float32),
                os.path.join(save_folder, "img_{}_{}.jpg".format(image_name_prefix, i)),
            )
    elif isinstance(images, torch.Tensor):
        for i in range(images.shape[0]):
            save_image(
                images[i].detach().cpu().to(torch.float32),
                os.path.join(save_folder, "img_{}_{}.jpg".format(image_name_prefix, i)),
            )


def print_networks(net):
    """Print the total number of parameters in the network and (if verbose) network architecture"""
    if not isinstance(net, nn.Module):
        print("net type error")
    print("net layers and weights:")
    for name, param in net.named_parameters():
        print("layer: ", name, "param dtype: ", param.dtype)
    print("-----------------------------------------------")
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("[Network] Total number of parameters : %.3f M" % (num_params / 1e6))
    print("-----------------------------------------------")


def save_weights(net, save_path):
    torch.save(net.module.cpu().state_dict(), save_path)


def save_training_params(optimizer, total_epochs, save_path):
    torch.save(
        {
            "opt_state_dict": optimizer.state_dict(),
            "total_epochs": total_epochs,
        },
        save_path,
    )
