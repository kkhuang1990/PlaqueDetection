"""
=========================
Measure region properties
=========================

This example shows how to measure properties of labelled image regions.

"""
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure
from sklearn.preprocessing import label_binarize
import os.path as osp
from utils import gray2mask, mask2rgb
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage

# def bound_weight(mask, sigma=5.0, w0=10.0, n_classes=3, k=2):
#     """ calculate boundary weight of each pixel given GT mask
#             For more information, please refer to the original paper of U-Net
#         :param mask: tensor, [N, C, H, W], encoded mask
#         :param sigma: float, variance of Gaussian pdf
#         :param w0: float, aptitude of Gaussian pdf
#         :param n_classes: int, # of classes
#         :param k: int, top k shortest distances to calculate
#         :return: weights, tensor, [N, H, W], weights for each pixel
#     """
#
#     h, w = mask.size()[2:]
#     conv_filter = torch.ones(1, 1, 3, 3).cuda()
#     y = mask.clone().zero_()
#     for i in range(n_classes):
#         tmp = mask[:, i].unsqueeze(1)
#         y[:,i] = F.conv2d(tmp, conv_filter, padding=2).squeeze(1)[:, 1:-1, 1:-1]
#
#     y = y.long()
#     y[y == 9] = 0
#     bounds = y.sum(1)
#     plt.figure()
#     plt.imshow(bounds[0])
#     plt.show()
#     pixel_cords = torch.meshgrid(torch.arange(h), torch.arange(w)) # 2, H, W
#
#     weights = torch.zeros(len(mask), h, w).cuda()
#     for i, b in enumerate(bounds):
#         b_cords = torch.nonzero(b) # N, 2
#         tmp = pixel_cords.repeat(len(b_cords), 1, 1, 1).permute(2, 3, 0, 1)
#         tmp = (tmp - pixel_cords).norm(dim=-1)
#         tmp = torch.topk(tmp, k, largest=False, dim=-1)[0]
#         weights[i] = w0 * torch.exp(-0.5 * torch.sum(tmp, dim=-1) ** 2 / sigma ** 2) # [H, W]
#
#     return weights


def bound_weight(mask, sigma=5.0, w0=10.0, n_classes=3, k=2):
    """ calculate boundary weight of each pixel given GT mask
        For more information, please refer to the original paper of U-Net
    :param mask: numpy ndarray, [N, H, W], GT mask
    :param sigma: float, variance of Gaussian pdf
    :param w0: float, aptitude of Gaussian pdf
    :param n_classes: int, # of classes
    :param k: int, top k shortest distances to calculate
    :return: weights, numpy ndarray/ pytorch tensor, [N, H, W], weights for each pixel
    """

    h, w = mask.shape[1:]
    weights = np.zeros_like(mask).astype(np.float32)

    for l_inx, label in enumerate(mask):
        label_binary = label_binarize(label.flatten(), classes=range(0, n_classes))
        label_binary = np.reshape(label_binary, (h, w, -1))
        bound_binary = np.zeros_like(label_binary)

        for i in range(n_classes):
            tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
            cdt = np.logical_and(tmp>=1, tmp<=2)
            bound_binary[:, :, i] = cdt

        bound = np.any(bound_binary, axis=2)

        # calculate prior weight for each pixel following rule defined in original paper of U-Net
        bound_cords = np.array(np.where(bound)).transpose()

        weight = np.zeros_like(label).astype(np.float32)
        pixel_cords = np.array(np.meshgrid(range(h), range(w), indexing='ij')) #  [2, H, W]

        for i in range(h):
            for j in range(w):
                dists = np.linalg.norm(bound_cords - pixel_cords[:, i, j], axis=-1)
                dists_topk = np.partition(dists, k)[:k].sum()
                weight[i, j] = w0 * math.exp(-0.5 * dists_topk**2 / sigma**2)

        weights[l_inx] = weight

        if l_inx == 0:
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(mask2rgb(label))
            axes[0].set_title("input image")
            axes[1].imshow(bound, cmap=plt.cm.gray)
            axes[1].set_title("detected boundary")
            axes[2].imshow(weight)
            axes[2].set_title("weight map")
            plt.show()
            plt.savefig("./bound_weight/{}_{}_{}.png".format(w0, sigma, k))

    return weights

if __name__ == "__main__":
    img_dir = "../mask/038.tiff"
    image = io.imread(img_dir)
    image[image == 76] = 255
    image[image == 151] = 255
    image = gray2mask(image)
    image = np.tile(image, (2, 1, 1))
    print(image.shape)
    # image_t = torch.from_numpy(image)
    # encoded_image = torch.zeros(10, 1, 512, 512)
    # encoded_image.scatter_(1, image.unsqueeze(1), 1)

    weights = bound_weight(image)
    print(weights.size())


# image[bound] = 76

# plt.figure()
# plt.imshow(image)
# plt.show()


# mask = gray2mask(image)
#
# struct = ndimage.generate_binary_structure(2, 2)
# edges = ndimage.binary_erosion(image, struct)
#
# print(edges.shape)
#
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image, cmap=plt.cm.gray)
# axes[1].imshow(bound)

# plt.show()

# contours = measure.find_contours(image, level=1.0) # with size [N, 2]
# print(contours)
# print("{} contours pixels found".format(len(contours)))
#
# # Display the image and plot all contours found
# fig, ax = plt.subplots()
#
# ax.imshow(image, interpolation='nearest')
#
# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
#
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

