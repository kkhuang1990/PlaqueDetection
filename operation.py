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
from image.transforms import Intercept
from torchvision import transforms
from utils import hu2lut
import random

def risk_statistic(data_dir):
    """ count the number of slices for each risk class range from 1 to 4 """
    # data_dir = "/data/ugui0/antonio-t/CPR_20180518/20180518"

    n_slices_risk = {0:0, 1:0, 2:0, 3:0}
    samples = [sample for sample in listdir(data_dir) if sample.startswith('IRB')]

    for sample in samples:
        sample_path = osp.join(data_dir, sample)
        series = [s for s in listdir(sample_path) if not s.startswith('.')]
        series = sorted(series, key=lambda x: int(x[1:]))
        series_path = osp.join(sample_path, series[0])
        exclusions = ['.tiff', 'conf']
        phases = [phase for phase in sorted(listdir(series_path))
                  if not any([phase.endswith(ex) for ex in exclusions]) and not phase.startswith('.')]

        n_slices_per_sample = {0:0, 1:0, 2:0, 3:0}
        for phase in phases:
            # load masks
            mask_dir = osp.join(series_path, phase + 'conf')
            mask_files = [file for file in listdir(mask_dir) if file.startswith('I0') and file.endswith('.tiff')]
            mask_files = sorted(mask_files, key=lambda x: int(x.split('.')[0][2:]))

            # load images
            img_dir = osp.join(series_path, phase)
            img_files = [file for file in listdir(img_dir) if file.startswith('I0')]
            img_files = sorted(img_files, key=lambda x: int(x.split('.')[0][2:]))

            if len(mask_files) > 0 and len(img_files) > 0:
                slice_info_file = osp.join(mask_dir, 'sliceinfo.txt')
                start, end, risk, _, _, _ = get_slice_info(slice_info_file)
                inxs, cnts = np.unique(risk, return_counts=True)
                for i, cnt in zip(inxs, cnts):
                    n_slices_risk[i] += cnt
                    n_slices_per_sample[i] += cnt

        print("{}: {} - {}".format(sample, n_slices_per_sample.keys(), n_slices_per_sample.values()))

    print("Overall: {} - {}".format(n_slices_risk.keys(), n_slices_risk.values()))

########################################################################################
trans = transforms.Compose([RandomCentralCrop(),
                            Intercept(),
                            RandomRotation(),
                            RandomFlip(),
                            Rescale(96)])

def hist_mean_var_statistic(sample_path):
    """ calculate histogram, mean, variance, max, min et al statistic information for each patient sample """
    sample = sample_path.split('/')[-1]
    # num_slices_per_class = np.zeros(5, dtype=np.uint16)
    imgs_samp = []
    for artery in sorted(listdir(sample_path)):
        mask_path = osp.join(sample_path, artery, 'applicate', 'mask')
        img_path = osp.join(sample_path, artery, 'applicate', 'image')

        # extract label files
        files = sorted(
            [file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

        imgs_trans = np.zeros((len(files), 96, 96), dtype=np.float32)
        for f_inx, file in enumerate(files):
            slice = io.imread(osp.join(img_path, file))
            label = io.imread(osp.join(mask_path, file))
            imgs_trans[f_inx], _ = trans((slice, label))

        imgs_samp.append(imgs_trans)

    print("sample: {}".format(sample))
    arteries = sorted(listdir(sample_path))
    fig, axes = plt.subplots(1, len(arteries))
    for a_inx in range(len(arteries)):
        artery = arteries[a_inx]
        imgs_artery = imgs_samp[a_inx]
        print(artery)
        print("# of slices: {}".format(len(imgs_artery)))
        print("Max: {}, Min: {}, Mean: {}, Std: {}".format(imgs_artery.max(), imgs_artery.min(), imgs_artery.mean(), imgs_artery.std()))
        per95 = np.percentile(imgs_artery.flatten(), 95)
        per5 = np.percentile(imgs_artery.flatten(), 5)
        print("Percentile 95: {}, Percentile 5: {}".format(per95, per5))
        axes[a_inx].hist(imgs_artery.flatten(), bins=2000)
        axes[a_inx].set_title("Histogram of {}".format(artery))
        axes[a_inx].set_xlabel("HU value")
        # axes[a_inx].set_ylabel("count")
        x_pos = int(imgs_artery.min())
        # print(x_pos)
        hist, hist_edge = np.histogram(imgs_artery.flatten(), bins=2000)
        y_pos = np.max(hist)
        # print(y_pos)
        axes[a_inx].text(x_pos, int(y_pos*1.0), "# of slices: {:4d}".format(len(imgs_artery)), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.95), "Max: {:4.2f}".format(imgs_artery.max()), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.9), "Min: {:4.2f}".format(imgs_artery.min()), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.85), "Mean: {:4.2f}".format(imgs_artery.mean()), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.8), "Std: {:4.2f}".format(imgs_artery.std()), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.75), "Per95: {:4.2f}".format(per95), color="red")
        axes[a_inx].text(x_pos, int(y_pos*0.7), "Per5: {:4.2f}".format(per5), color="red")

    plt.savefig("./samples_hist/pixel_hist_per_artery/{}.png".format(sample))

    imgs_samp = np.concatenate(imgs_samp, axis=0)
    return (imgs_samp.mean(), imgs_samp.std())

def hist_mean_var_statistic_multi_preocess(num_workers=24):
    """ calculate plaque statistic for given data """

    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    args = []

    for mode in ['train', 'val', 'test']:
        with open(osp.join('./configs/config_35', mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]
            for sample in samples:
                args.append(osp.join(data_dir, sample))

    pool = Pool(processes=num_workers)
    print("{} CPUs are used".format(num_workers))
    results = pool.map(hist_mean_var_statistic, args)
    means, stds = [[result[i] for result in results] for i in range(len(results[0]))]

    plt.figure()
    plt.errorbar(range(1, len(means)+1), means, yerr=stds, fmt='-o')
    for i in range(len(means)):
        plt.text(i+1, means[i]+stds[i]+1, "{:d}".format(int(stds[i])), color="red",
                 horizontalalignment="center", verticalalignment="center")
    plt.ylabel("standard variance")
    plt.xlabel("index of sample")
    plt.title("variance bar for each sample")
    plt.savefig("./samples_hist/pixel_hist_per_sample/varbar.png")

########################################################################################
# plaque statistics
def plaque_statistic_multi_preocess(num_workers=24):
    """ calculate plaque statistic for given data """

    data_dir = "/data/ugui0/antonio-t/CPR_multiview"
    for mode in ['train', 'val', 'test']:
        with open("./num_slices_per_class.txt", "a") as writer:
            writer.write("{}\n".format(mode))
        with open(osp.join('./config', mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]
            args = [osp.join(data_dir, sample) for sample in samples]

        pool = Pool(processes=num_workers)
        print("{} CPUs are used".format(num_workers))
        num_slices_per_class = pool.map(plaque_statistic, args)
        pool.close()

        total_slices_per_class = reduce(lambda x, y: x+y, num_slices_per_class)
        print("total slices for each class: {}".format(total_slices_per_class))
        with open("./num_slices_per_class.txt", "a") as writer:
            writer.write("total slices for each class: {}\n".format(total_slices_per_class))

def plaque_statistic(sample_path):
    """ count how many slices for each class, especially for cal and non-cal """
    sample = sample_path.split('/')[-1]
    num_slices_per_class = np.zeros(5, dtype=np.uint16)
    for artery in sorted(listdir(sample_path)):
        mask_path = osp.join(sample_path, artery, 'applicate', 'mask')

        # extract label files
        label_files = sorted(
            [file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

        for label_file in label_files:
            label_path = osp.join(mask_path, label_file)
            label = io.imread(label_path)

            for i, j in enumerate([0, 29, 255, 151, 76]):
                num_slices_per_class[i] += (np.sum(label == j) != 0)

    print("{} -- # of slices for each class: {}".format(sample, num_slices_per_class))
    with open("./num_slices_per_class.txt", "a") as writer:
        writer.write("{} -- # of slices for each class: {}\n".format(sample, num_slices_per_class))

    return num_slices_per_class


# plaque statistics
def hu_statistic_multi_preocess(num_workers=24):
    """ calculate plaque statistic for given data """

    # data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    data_dir = "/data/ugui0/antonio-t/CPR_multiview"
    thres = 10
    interval = 8
    for mode in ['train']:
        with open(osp.join('./configs/config', mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]
            args = [osp.join(data_dir, sample) for sample in samples]

        pool = Pool(processes=num_workers)
        print("{} CPUs are used".format(num_workers))
        hus_all_samples = pool.map(hu_statistic, args)
        pool.close()

    names = ['background', 'central_part', 'outline', 'cal', 'noncal']

    if osp.exists("./samples_hist/hist_alldata/hist.txt"):
        os.remove("./samples_hist/hist_alldata/hist.txt")

    # joint list is returned
    for name, val in zip(names, zip(*hus_all_samples)):
        datas = reduce(lambda x, y: x + y, val)  # add up all columns

        if name == 'outline':
            ratio = datas[1500:1551].sum()/ datas.sum()
            print("ratio of pixels with HU range 0~50 in outline: {:.4f}".format(ratio))
            tmp = datas
        if name == 'noncal':
            ratio = datas[1500:1551].sum() / datas.sum()
            print("ratio of pixels with HU range 0~50 in noncal: {:.4f}".format(ratio))

        print("{} pixels in {}".format(datas.sum(), name))

        if name == 'outline' or name == 'background':
            data_list = []
            with open("./samples_hist/hist_alldata/hist.txt", 'a') as writer:
                writer.write("{}\n".format(name))

                for inx in range(len(datas)):
                    curr = datas[inx-interval:inx].mean() if inx>interval else datas[:inx+1].mean()
                    prev = datas[inx-1-interval:inx-1].mean() if inx>interval else datas[:inx+1].mean()
                    hu_diff = curr - prev
                    writer.write("{} : {}\n".format(inx - 1500, int(hu_diff)))
                    data_list.append(abs(int(hu_diff)))
            # plot the absolute HU difference
            plt.figure()
            plt.plot(np.arange(-1000, -200), data_list[500:1300], 'g')
            plt.xlabel("HU value")
            plt.ylabel("count difference")
            plt.title("Histogram of count difference for {}".format(name))
            plt.savefig('./samples_hist/hist_alldata/{}_diff.jpg'.format(name))

        min_hu = np.array(np.where(datas >= thres)).min() - 1500
        max_hu = np.array(np.where(datas >= thres)).max() - 1500
        mean_hu = (np.arange(-1500, 2500) * datas).sum() / datas.sum()
        print("min : {} max : {} ave : {}".format(min_hu, max_hu, mean_hu))
        plt.figure()
        plt.plot(np.arange(min_hu, max_hu), datas[min_hu+1500:max_hu+1500], 'g')
        plt.xlabel("HU value")
        plt.ylabel("count")
        plt.title("Histogram of {}".format(name))
        plt.savefig('./samples_hist/hist_alldata/' + name + '.jpg')

    plt.figure()
    plt.xlabel("HU value")
    plt.ylabel("count")
    plt.title("Histogram comparison before/after adding noncal pixels")
    plt.plot(np.arange(-1500, 2500), tmp, 'b', label="before adding noncal")
    datas = datas + tmp
    plt.plot(np.arange(-1500, 2500), datas, 'r', label="after adding noncal")
    plt.legend()
    plt.savefig("./samples_hist/hist_alldata/hist_comparison.jpg")

def hu_statistic(sample_path):
    """ count how many slices for each class, especially for cal and non-cal """
    cnt_list = [np.zeros(4000, dtype=np.int64) for _ in range(5)] # for background, central_part, outline, cal and noncal respectively
    sample = sample_path.split('/')[-1]
    print("Processing ", sample)
    for artery in sorted(listdir(sample_path)):
        mask_path = osp.join(sample_path, artery, 'applicate', 'mask')
        img_path = osp.join(sample_path, artery, 'applicate', 'image')

        # extract label files
        label_files = sorted(
            [file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

        image = np.stack([io.imread(osp.join(img_path, file)) for file in label_files])
        mask = np.stack([io.imread(osp.join(mask_path, file)) for file in label_files])

        # central crop with size 160. if not, codes will be very slow
        image = image[:, 128:384, 128:384]
        mask = mask[:, 128:384, 128:384]

        # if np.sum(label == 76) != 0:
        for i, grayscale in enumerate([0, 29, 255, 151, 76]):
            pixels = image[mask == grayscale].flatten()
            pixels_unq, cnts= np.unique(pixels, return_counts=True)
            for pixel, cnt in zip(pixels_unq, cnts):
                cnt_list[i][pixel+1500] += cnt

    return cnt_list

# plaque statistics
def outline_noncal_overlap_statistic_multi_preocess(num_workers=24, step=100):
    """ calculate overlapping between noncal and outline with HU range 0 ~ 50 """

    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    for mode in ['train']:
        with open(osp.join('./config', mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]
            args = [osp.join(data_dir, sample) for sample in samples]

        pool = Pool(processes=num_workers)
        print("{} CPUs are used".format(num_workers))
        results = pool.map(outline_noncal_overlap_statistic, args)
        f1s_all_samples, noncal_all_maps, outline0050_all_maps, overlap_maps = \
            [[result[i] for result in results] for i in range(len(results[0]))]
        pool.close()

        f1s_all_samples = reduce(lambda x, y: x + y, f1s_all_samples)
        noncal_all_maps = np.concatenate(noncal_all_maps, axis=0)
        outline0050_all_maps = np.concatenate(outline0050_all_maps, axis=0)
        overlap_maps = np.concatenate(overlap_maps, axis=0)
        print(noncal_all_maps.shape, outline0050_all_maps.shape, overlap_maps.shape)
        noncal_map_ave = np.mean(noncal_all_maps, axis=0)
        outline0050_map_ave = np.mean(outline0050_all_maps, axis=0)


        save_dir = "./overlap_map"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(0, len(overlap_maps), step):
            end = min(i + step, len(overlap_maps))
            data = overlap_maps[i:end]
            metrics = f1s_all_samples[i:end]
            fig_name = osp.join(save_dir,"{:03d}".format(i+1))

            sample_stack_color(data, metrics, rows=step//10, cols=10, start_with=0, show_every=1,
                               scale=4, fig_name=fig_name)

        # plot noncal heatmap
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
        plt.hist(f1s_all_samples, bins=100)
        plt.xlabel("overlapping (measured in F1)")
        plt.ylabel("histogram")
        ave_f1 = sum(f1s_all_samples) / len(f1s_all_samples)
        plt.title("Histogram of overlap between noncal and outline[0~50]: {:.4f}".format(ave_f1))
        plt.savefig(osp.join(save_dir, "hist_overlap.jpg"))

        print("average f1 score between noncal and outline with HU range 0 ~ 50 for all noncal slices: {}".format(ave_f1))


def outline_noncal_overlap_statistic(sample_path):
    """ calculate overlapping between noncal and outline with HU range 0 ~ 50 """
    f1s = []
    noncal_flag = False

    sample = sample_path.split('/')[-1]
    print("Processing ", sample)
    for artery in sorted(listdir(sample_path)):
        mask_path = osp.join(sample_path, artery, 'applicate', 'mask')
        img_path = osp.join(sample_path, artery, 'applicate', 'image')

        # extract label files
        label_files = sorted(
            [file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

        for label_file in label_files:
            label_path = osp.join(mask_path, label_file)
            slice_path = osp.join(img_path, label_file)

            label = io.imread(label_path)
            slice = io.imread(slice_path)

            if np.sum(label == 76) != 0:
                overlap_map = np.zeros(label.shape, dtype=np.uint8)
                # noncal map
                mask_noncal = (label == 76)
                noncal_pixels = slice[mask_noncal].flatten()
                # print(noncal_pixels.max(), noncal_pixels.min())
                mask_hu0050 = np.logical_and(slice <= 50, slice >= 0)

                mask_outline = np.logical_or(label == 76, label == 255)
                mask_outline = np.logical_or(mask_outline, label == 151)

                mask_outline_hu0050 = np.logical_and(mask_outline, mask_hu0050)
                # mask_outline_hu0050 = mask_outline

                try:
                    f1s.append(f1_score(mask_noncal.flatten(), mask_outline_hu0050.flatten()))
                except:
                    print(label_path)

                overlap_map[mask_noncal] = 76
                overlap_map[mask_outline_hu0050] = 150
                overlap_map[np.logical_and(mask_noncal, mask_outline_hu0050)] = 226  # yellow for overlap

                if not noncal_flag:
                    overlap_maps = overlap_map[np.newaxis, :, :]
                    noncal_maps = mask_noncal[np.newaxis, :, :]
                    outline0050_maps = mask_outline_hu0050[np.newaxis, :, :]
                    noncal_flag = True
                else:
                    noncal_maps = np.concatenate((noncal_maps, mask_noncal[np.newaxis, :, :]), axis=0)
                    outline0050_maps = np.concatenate((outline0050_maps, mask_outline_hu0050[np.newaxis, :, :]), axis=0)
                    overlap_maps = np.concatenate([overlap_maps, overlap_map[np.newaxis, :, :]], axis=0)


    if not noncal_flag:
        noncal_maps = np.empty((0, *label.shape), dtype=np.uint8)
        outline0050_maps = np.empty((0, *label.shape), dtype=np.uint8)
        overlap_maps = np.empty((0, *label.shape), dtype=np.uint8)

    return f1s, noncal_maps, outline0050_maps, overlap_maps


def get_slice_info(slice_info_path):
    """ extract range of 'good' slices and risk value
        (Be careful with sample of slices which are all effective or all non-effective)
    """
    with open(slice_info_path, "rb") as reader:
        flag = 0
        risk_list, sig_stenosis_list, pos_remodeling_list, napkin_ring_list = [], [], [], []
        lines = reader.readlines()
        start_inx, end_inx = len(lines), len(lines)
        for l_inx, line in enumerate(lines):
            slice_info = line.decode('utf8').strip().split(',')
            is_effective = slice_info[-1]
            risk, sig_stenosis, pos_remodeling, napkin_ring = slice_info[1:5]

            if is_effective == "true":
                risk_list.append(int(risk))
                sig_stenosis_list.append(int(sig_stenosis=="true"))
                pos_remodeling_list.append(int(pos_remodeling=="true"))
                napkin_ring_list.append(int(napkin_ring=="true"))

            if not flag and is_effective == "true":
                start_inx = l_inx
                flag = 1
            if flag and is_effective == "false":
                end_inx = l_inx
                break
        end_inx = min(end_inx, len(lines))
        print("{}: start-{} end-{}".format(slice_info_path.split('/')[-4:], start_inx, end_inx))

        return start_inx, end_inx, np.array(risk_list), np.array(sig_stenosis_list), \
                       np.array(pos_remodeling_list), np.array(napkin_ring_list)
        # except UnboundLocalError:
        #     print("errors happened in {}".format(slice_info_path.split('/')[-4:]))

def overall_statistic_multi_preocess(method, num_workers=24, step=10):
    """ calculate overlapping between noncal and outline with HU range 0 ~ 50 """

    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    for mode in ['train']:
        with open(osp.join('./config', mode + '.txt'), 'r') as reader:
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

        save_dir = "./noncal_map_9"
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


def noncal_statistic(sample_path):
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

            label = io.imread(label_path)
            slice = io.imread(slice_path)

            if np.sum(label == 76) != 0:   # noncal-76, cal-151
                if rand_seeds[inx] < 2.0:
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
                    mask_hu0050 = np.logical_and(slice <= 50, slice >= 0)
                    hu0050_map = np.zeros(label.shape, dtype=np.uint8)
                    hu0050_map[mask_hu0050] = 150

                    slice1 = hu2lut(slice, window=255, level=27.5)  # only extract HU range [-100, 155]
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

    # hist_mean_var_statistic_multi_preocess(num_workers=48)
    # hist_mean_var_statistic_multi_preocess(num_workers=48)
    # hu_statistic_multi_preocess(num_workers=24)
    hist_mean_var_statistic_multi_preocess(num_workers=48)