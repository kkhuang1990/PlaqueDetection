import matplotlib as mpl
mpl.use('Agg')

import os
import os.path as osp
from os import listdir
import numpy as np
import random
from skimage import io
from multiprocessing import Pool
from utils import dcm2hu, hu2gray, rgb2gray, rgb2mask, centra_crop, mask2gray

def create_image_mask_pair_cycleGAN(data_dir, des_dir, mode, prob=0.1, patch_size= 256):
    """ create image and mask pairs for training cycleGAN
    Args:
        data_dir: str, from where to read slices
        des_dir: str, to where to save slices
        mode: str, data type: train/val/test
        prob: float, sampling ratio of slices with risk 0
    """
    slice_inx = 0
    with open(osp.join('./config', mode+'.txt'), 'r') as reader:
        samples = [line.strip('\n') for line in reader.readlines()]

    des_image_dir = osp.join(des_dir, 'A', mode)
    des_mask_dir = osp.join(des_dir, 'B', mode)
    if not osp.exists(des_image_dir):
        os.makedirs(des_image_dir)
    if not osp.exists(des_mask_dir):
        os.makedirs(des_mask_dir)

    for sample in samples:
        sample_path = osp.join(data_dir, sample)

        for artery in sorted(listdir(sample_path)):
            artery_path = osp.join(sample_path, artery)
            image_path = osp.join(sample_path, artery, 'image')
            mask_path = osp.join(sample_path, artery, 'mask')

            # extract slice files
            slice_files = sorted([file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
            label_files = sorted([file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

            # extract slice information
            risks = np.loadtxt(osp.join(artery_path, "risk_labels.txt"), dtype=np.uint8)

            for slice_file, label_file, risk in zip(slice_files, label_files, risks):
                slice_path = osp.join(image_path, slice_file)
                label_path = osp.join(mask_path, label_file)
                x = random.uniform(0, 1)
                if (risk == 0 and x < prob) or risk != 0:
                    slice_inx += 1
                    slice, label  = io.imread(slice_path), io.imread(label_path)
                    slice = centra_crop(hu2gray(slice), patch_size=patch_size)
                    label = centra_crop(mask2gray(label), patch_size=patch_size)
                    assert slice.shape == label.shape, "slice size and label size must match with each other"

                    des_slice_path = osp.join(des_image_dir, "{:05d}.tiff".format(slice_inx))
                    des_label_path = osp.join(des_mask_dir, "{:05d}.tiff".format(slice_inx))
                    io.imsave(des_slice_path, slice)
                    io.imsave(des_label_path, label)

    print("{} : {} samples".format(mode, slice_inx))

def create_image_mask_pair_cycleGAN_multi_preocess(method, data_dir, des_dir, prob=0.1, patch_size=256, num_workers=24):
    """ resave data into desired format for faster read
    Args:
        method: function, use which method to resave the data
        data_dir: string, from where to read data
        des_dir: string, to where to save data
        num_workers: int, how many processes in parallel
    """
    args = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        args.append((data_dir, des_dir, mode, prob, patch_size))

    pool = Pool(processes=num_workers)
    print("{} CPUs are used".format(num_workers))
    pool.starmap(method, args)
    pool.close()