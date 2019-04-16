# _*_ coding: utf-8 _*_
""" functions for image processing and other operations """

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import os
import os.path as osp
from os import listdir
import numpy as np
from skimage import io
from multiprocessing import Pool
from functools import reduce
from sklearn.metrics import f1_score
from vision import sample_stack_color, sample_list3
from image.transforms import HU2Gray, RandomCentralCrop, Rescale, Gray2Triple, RandomRotation, RandomFlip
from torchvision import transforms
from utils import hu2lut
import random


def sample_hist_statistic_multi_preocess(method, num_workers=24, step=80):
    """ calculate overlapping between noncal and outline with HU range 0 ~ 50 """

    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    for mode in ['train']:
        with open(osp.join('./configs/config', mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]
            args = [osp.join(data_dir, sample) for sample in samples]

        pool = Pool(processes=num_workers)
        print("{} CPUs are used".format(num_workers))
        results = pool.map(method, args)
        noncal_evals, f1s_samples, file_paths, noncal_maps, outline0050_maps, overlap_maps, labels, slices1, slices2, \
            hu0050_maps, mix_overlap_maps  = [[result[i] for result in results] for i in range(len(results[0]))]
        pool.close()

        f1s_samples = reduce(lambda x, y: x + y, f1s_samples)
        file_paths = reduce(lambda x, y: x + y, file_paths)
        noncal_evals = np.concatenate(noncal_evals, axis=0)
        noncal_maps = np.concatenate(noncal_maps, axis=0)
        outline0050_maps = np.concatenate(outline0050_maps, axis=0)
        overlap_maps = np.concatenate(overlap_maps, axis=0)
        labels = np.concatenate(labels, axis=0)
        slices1 = np.concatenate(slices1, axis=0)
        slices2 = np.concatenate(slices2, axis=0)
        hu0050_maps = np.concatenate(hu0050_maps, axis=0)
        mix_overlap_maps = np.concatenate(mix_overlap_maps, axis=0)

        print(noncal_maps.shape, outline0050_maps.shape, overlap_maps.shape,
              labels.shape, slices1.shape, hu0050_maps.shape)

        save_dir = "./samples_hist/hu_lower-800_224"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # save file paths
        with open(osp.join(save_dir, 'noncal_paths.txt'), 'w') as writer:
            for inx, file_path in enumerate(file_paths):
                writer.write("{} : {}\n".format(inx + 1, file_path))

        # calculate mean noncal map and mean outline_hu0050 map
        noncal_map_ave = np.mean(noncal_maps, axis=0)
        outline0050_map_ave = np.mean(outline0050_maps, axis=0)

        # calculate average F1
        ave_f1 = sum(f1s_samples) / len(f1s_samples)
        print("average f1 score between noncal and outline with HU range 0 ~ 50 for all noncal slices: {}".format(ave_f1))

        plt.figure()
        plt.title("heatmap of noncalcified plaque")
        plt.imshow(noncal_map_ave)
        plt.colorbar()
        plt.savefig(osp.join(save_dir, "noncal_heatmap.jpg"))

        # plot outline 0050 heatmap
        plt.figure()
        plt.title("heatmap of outline with HU range 0 ~ 50")
        plt.imshow(outline0050_map_ave)
        plt.colorbar()
        plt.savefig(osp.join(save_dir, "outline0050_heatmap.jpg"))

        plt.figure()
        plt.hist(f1s_samples, bins=100)
        plt.xlabel("overlapping (measured in F1)")
        plt.ylabel("histogram")
        plt.title("Histogram of overlap between noncal and outline[0~50]: {:.4f}".format(ave_f1))
        plt.savefig(osp.join(save_dir, "hist_overlap.jpg"))


        datas = [{'slice1':slice1, 'slice2':slice2, 'label':label, 'hu0050':hu0050, 'overlap': overlap, 'f1': f1,
                  'mix_overlap':mix_overlap, 'noncal_eval':noncal_eval, 'file_path':file_path}
                 for (slice1, slice2, label, hu0050, overlap, f1, mix_overlap, noncal_eval, file_path)
         in zip(slices1, slices2, labels, hu0050_maps, overlap_maps, f1s_samples, mix_overlap_maps, noncal_evals, file_paths)]

        for i in range(0, len(datas), step):
            end = min(i + step, len(datas))
            data_batch = datas[i:end]
            fig_name = osp.join(save_dir,"{:03d}".format(i+1))

            sample_list3(data_batch, rows=step, cols=6, start_with=0,
                         show_every=1, scale=4, fig_name=fig_name, start_inx=i+1)


def sample_hist_statistic(sample_path):
    """ calculate overlapping between noncal and outline with HU range 0 ~ 50 """
    k= 5
    f1s = []
    file_paths = []  # save paths of slice with non-calcified plaque
    noncal_flag = False

    np.random.seed(42)
    sample = sample_path.split('/')[-1]
    print("Processing ", sample)
    for artery in sorted(listdir(sample_path)):
        mask_path = osp.join(sample_path, artery, 'applicate', 'mask')
        img_path = osp.join(sample_path, artery, 'applicate', 'image')

        # extract label files
        label_files = sorted(
            [file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

        rand_seeds = np.random.uniform(0.0, 1.0, len(label_files))
        for inx, label_file in enumerate(label_files):
            label_path = osp.join(mask_path, label_file)
            slice_path = osp.join(img_path, label_file)

            label = io.imread(label_path)[144:368, 144:368]
            slice = io.imread(slice_path)[144:368, 144:368]

            if rand_seeds[inx] < 0.05:
                # save file path
                file_path = '/'.join([sample, artery, label_file])
                file_paths.append(file_path)

                # calculate noncal evaluations
                n_above50 = np.sum(np.logical_and(label==76, slice>50))
                n_below0 = np.sum(np.logical_and(label==76, slice<0))
                if np.sum(label == 76) != 0:
                    noncal_pxiels_sort = sorted(slice[label == 76].flatten())
                    topk = noncal_pxiels_sort[-k:]
                    buttomk = noncal_pxiels_sort[:k]
                else:
                    topk = [51 for _ in range(k)]
                    buttomk = [-1 for _ in range(k)]
                noncal_eval = np.array([n_above50, n_below0, *topk, *buttomk]).astype(np.int16)

                # hu0050 map
                # mask_hu0050 = np.logical_and(slice <= -800, slice >= -1000)
                mask_hu0050 = (slice <= -800)
                hu0050_map = np.zeros(label.shape, dtype=np.uint8)
                hu0050_map[mask_hu0050] = 150

                slice1 = slice  # only extract HU range [-100, 155]
                slice2 = hu2lut(slice, window=1000, level=700)  # for calcification

                # noncal map
                mask_noncal = (label == 76)
                mask_outline = np.logical_or(label == 76, label == 255)
                mask_outline = np.logical_or(mask_outline, label == 151)
                mask_outline_hu0050 = np.logical_and(mask_outline, mask_hu0050)

                # calculate F1 score
                f1s.append(f1_score(mask_noncal.flatten(), mask_outline_hu0050.flatten()))

                # calculate overlap
                overlap_map = np.zeros(label.shape, dtype=np.uint8)
                overlap_map[mask_noncal] = 76
                overlap_map[mask_outline_hu0050] = 150
                overlap_map[np.logical_and(mask_noncal, mask_outline_hu0050)] = 226  # yellow for overlap

                # combine overlap with GT label
                mix_overlap = label.copy()
                mix_overlap[mask_outline_hu0050] = 150
                mix_overlap[np.logical_and(mask_noncal, mask_outline_hu0050)] = 226  # yellow for overlap

                if not noncal_flag:
                    noncal_evals = noncal_eval[np.newaxis, :]
                    labels = label[np.newaxis, :, :]
                    slices1 = slice1[np.newaxis, :, :]
                    slices2 = slice2[np.newaxis, :, :]
                    hu0050_maps = hu0050_map[np.newaxis, :, :]
                    overlap_maps = overlap_map[np.newaxis, :, :]
                    noncal_maps = mask_noncal[np.newaxis, :, :]
                    outline0050_maps = mask_outline_hu0050[np.newaxis, :, :]
                    mix_overlap_maps = mix_overlap[np.newaxis, :, :]
                    noncal_flag = True
                else:
                    noncal_evals = np.concatenate([noncal_evals, noncal_eval[np.newaxis, :]])
                    labels = np.concatenate([labels, label[np.newaxis, :, :]], axis=0)
                    slices1 = np.concatenate([slices1, slice1[np.newaxis, :, :]], axis=0)
                    slices2 = np.concatenate([slices2, slice2[np.newaxis, :, :]], axis=0)
                    hu0050_maps = np.concatenate([hu0050_maps, hu0050_map[np.newaxis, :, :]], axis=0)
                    noncal_maps = np.concatenate((noncal_maps, mask_noncal[np.newaxis, :, :]), axis=0)
                    outline0050_maps = np.concatenate((outline0050_maps, mask_outline_hu0050[np.newaxis, :, :]), axis=0)
                    overlap_maps = np.concatenate([overlap_maps, overlap_map[np.newaxis, :, :]], axis=0)
                    mix_overlap_maps = np.concatenate([mix_overlap_maps, mix_overlap[np.newaxis, :, :]], axis=0)

    if not noncal_flag:
        noncal_evals = np.empty((0, 2*k+2), dtype=np.int16)
        labels = np.empty((0, *label.shape), dtype=np.uint8)
        slices1 = np.empty((0, *label.shape), dtype=np.uint8)
        slices2 = np.empty((0, *label.shape), dtype=np.uint8)
        noncal_maps = np.empty((0, *label.shape), dtype=np.uint8)
        outline0050_maps = np.empty((0, *label.shape), dtype=np.uint8)
        overlap_maps = np.empty((0, *label.shape), dtype=np.uint8)
        hu0050_maps = np.empty((0, *label.shape), dtype=np.uint8)
        mix_overlap_maps = np.empty((0, *label.shape), dtype=np.uint8)

    print(f1s)
    return noncal_evals, f1s, file_paths, noncal_maps, outline0050_maps, overlap_maps, labels, slices1, slices2, hu0050_maps, mix_overlap_maps


if __name__ == "__main__":
    # plaque_statistic_multi_preocess()
    # hu_statistic_multi_preocess(num_workers=32)
    # overall_statistic_multi_preocess(method=noncal_statistic, num_workers=48)
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180621"
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180518/20180518"
    # data_dir = "/data/ugui0/antonio-t/CPR_20180601/CPR_20180601_copy"
    # # des_dir = "/data/ugui0/antonio-t/CPR_oversample_risk_artery/images"
    # des_dir = "/home/mil/huang/Dataset/CPR_rawdata/images"
    # resave_multi_preocess(dcm2tiff_per_artery_wo_augment, data_dir, des_dir, num_workers=36)
    #
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180621"
    # # risk_statistic(data_dir)
    #
    # data_dir = "/home/mil/huang/Dataset/CPR_rawdata/images"
    # des_dir = "/home/mil/huang/CPR_Segmentation_ver7/pytorch-CycleGAN-and-pix2pix/datasets/CPR_Segmentation_gray_centralcrop"
    # create_image_mask_pair_cycleGAN_multi_preocess(create_image_mask_pair_cycleGAN, data_dir,
    #                                                des_dir, prob=0.1, patch_size=256, num_workers=3)

    sample_hist_statistic_multi_preocess(sample_hist_statistic, num_workers=48)
