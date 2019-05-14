# _*_ coding: utf-8 _*_

""" use morphological operations and Snake to obtain single-pixel contour
from prediction results
"""

import matplotlib as mpl
mpl.use('Agg')
import torch
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from multiprocessing import Pool
from utils import lslist2bound
from skimage import img_as_float

import cv2 as cv2

def probmap2bound_slicewise(i, prob_map_b, thres=0.7, ks=9):
    """ obtain boundary from probability map for each slice
    :param prob_map_b: ndarray of size [B, H, W], probability map
    :param thres: float, thres for filtering out pixels with prob lower than given thres
    :param outer_ks: int, kernel size for bound detection
    :return: lses: list of obtained bounds
    """

    n_channel, height, width = prob_map_b.shape
    iter_max = 30
    lses = []

    for bound_inx in range(1, n_channel):  # inner and outer bound
        prob_map_bb = prob_map_b[bound_inx]
        pred_filter = (prob_map_bb >= thres).astype(np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        for inx in range(iter_max):
            image_close = cv2.morphologyEx(pred_filter, cv2.MORPH_CLOSE, kernel_close, iterations=inx+1)
            _, contours, _ = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [contour for contour in contours
                        if len(contour) > 4 and cv2.contourArea(contour) / len(contour) >= 4.0]

            if len(contours) > 0: # find the optimal number of iterations
                break

        if len(contours) > 0:
            mask = np.zeros(image_close.shape[:2], np.uint8)
            ls = cv2.drawContours(mask, contours, -1, 1, -1)

        else: # use Snake to find contours if cv2.findContours doesn't work well
            if bound_inx == 1:
                gimage = inverse_gaussian_gradient(img_as_float(image_close), alpha=100, sigma=5.0)
            else:
                gimage = inverse_gaussian_gradient(img_as_float(image_close), alpha=100, sigma=3.0)
            init = np.zeros(gimage.shape, dtype=np.int8)
            init[5:-5, 5:-5] = 1
            ls = morphological_geodesic_active_contour(gimage, 100, init, smoothing=1,
                                                       balloon=-1, threshold='auto')
        lses.append(ls)

    reg = lslist2bound(lses)

    return reg

def probmap2bound(prob_map, n_workers=32, thres=0.7, kernel_size=9):
    """ calculate constrained boundary from probmap
        this function accepts both ndarray and tensor inputs
    :param prob_map: Tensor/Ndarray of size [B, C, D, H, W], probability map as the Network output
    :return: bounds_cuda: Tensor/Ndarray of size [B, D, H, W], obtained closed contour
    """

    if not isinstance(prob_map, np.ndarray):  # convert tensor into ndarray
        # prob_map = F.softmax(prob_map, 1)
        if prob_map.dim() == 5:  # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:])  # combine first 2 dims

        prob_map_np = prob_map.data.cpu().numpy()  # [B', C, H, W]

    batch_size, n_channel, height, width = prob_map_np.shape
    args = []
    for b in range(batch_size):
        args.append((b, prob_map_np[b], thres, kernel_size))

    pool = Pool(processes=n_workers)
    bounds = pool.starmap(probmap2bound_slicewise, args)
    pool.close()

    bounds = np.stack(bounds).astype(np.uint8)
    bounds_cuda = Variable(torch.from_numpy(bounds).cuda()).long()  # convert ndarray into tensor

    return bounds_cuda