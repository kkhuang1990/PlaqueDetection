# _*_ coding: utf-8 _*_

""" Load data using hard mining, which means only load data whose segmentation accuracy is lower than the threshold
obtained from the previous epoch.
"""

import matplotlib as mpl
mpl.use('Agg')

import random
import os
import os.path as osp
from os import listdir

import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage import transform
from .transforms import CentralCrop, Gray2Mask, ToTensor, HU2Gray, Rescale, Gray2Binary, HU2GrayMultiStreamToTensor
from .transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, RandomFlip
from .transforms import Gray2Triple, Gray2TripleWithBound
from utils import hu2lut
from torchvision import transforms
from vision import sample_stack
from torch.autograd import Variable

torch.manual_seed(42)  # for shuffle=True

class CPRPlaqueTrainDataset(Dataset):

    def __init__(self, data_dir, metric_prev_epoch=None, phases_prev_epoch=None, transform=None, mode='train',
                 is_hard_mining=False, percentile=85, multi_view=False, only_plaque=False, config='config',
                 bc_learning=None, n_classes=5):
        """ read images from data_dir and save them into a dataloader
        hard mining strategy is used to recursively select 'hard' samples for each epoch
        :param data_dir: string, from where to read image
        :param transform: transform, what transforms to operate on input images
        :param interval: int, interval of sub-volume
        :param slice_stride: int, stride for selecting sub-volume
        :param hard_mining: bool, whether use bad mining or not
        :param metric_prev_epoch: numpy ndarray, metric obtained from the previous epoch
        :param phases_prev_epoch: list, phases of the previous epoch
        :param only_plaque: bool, whether to only load slices containing plaque or not
        :param config: str, data configuration directory
        :param n_classes: int, number of classes for annotation
        """

        super(CPRPlaqueTrainDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.is_hard_mining = is_hard_mining
        self.percentile = percentile
        self.multi_view = multi_view # whether to use multi-view inputs or not
        self.only_plaque = only_plaque
        self.config = config
        self.bc_learning = bc_learning  # whether to use bc_learning or not [None, 'bc', 'bc_plus']
        self.n_classes = n_classes

        # initialize phases for different modes
        if self.mode == 'train':
            self.phases = self.update_phases(metric_prev_epoch, phases_prev_epoch)
        else:
            self.phases = self.get_phases()

    def update_phases(self, metric_prev_epoch, phases_prev_epoch, thres_num_smps=300):
        """ update the phases by mining the hard samples
        :param thres_num_smps: int, when length of phases is blow given threshold, hard mining is not operated any nore
        :return phases: refined phases after mining the hard samples
        """
        if phases_prev_epoch is None:
            if self.only_plaque:
                phases = self.get_phases_plaque_samples()
            else:
                phases = self.get_phases()

        else:
            if self.is_hard_mining and len(phases_prev_epoch) >= thres_num_smps:
                thres = np.percentile(metric_prev_epoch, self.percentile)
                phases = [phase for phase, metric in zip(phases_prev_epoch, metric_prev_epoch) if metric <= thres]
            else:
                phases = phases_prev_epoch

        return phases

    def get_phases(self):
        phases = []
        with open(osp.join("../configs/{}".format(self.config), self.mode+'.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)

            for artery in sorted(listdir(sample_path)):
                artery_path = osp.join(sample_path, artery)
                # since there are less slices along ordinate/abscissa axis, we extract along ordinate axis
                image_path = osp.join(artery_path, 'ordinate', 'image')
                mask_path = osp.join(artery_path, 'ordinate', 'mask')

                # extract slice files
                slice_files = sorted([file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
                label_files = sorted([file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])

                for slice_file, label_file in zip(slice_files, label_files):
                    slice_path = osp.join(image_path, slice_file)
                    label_path = osp.join(mask_path, label_file)

                    phases.append((slice_path, label_path))

        return phases

    # def get_phases_risk_samples(self, ratio=0.1):
    #     """ choose all samples with non-zero risk plus randomly choose 10% samples with zero risk """
    #
    #     phases = []
    #     with open(osp.join("./{}".format(self.config), self.mode+'.txt'), 'r') as reader:
    #         samples = [line.strip('\n') for line in reader.readlines()]
    #
    #     for sample in samples:
    #         sample_path = osp.join(self.data_dir, sample)
    #
    #         for artery in sorted(listdir(sample_path)):
    #             artery_path = osp.join(sample_path, artery)
    #             # since there are less slices along ordinate/abscissa axis, we extract along ordinate axis
    #             image_path = osp.join(artery_path, 'ordinate', 'image')
    #             mask_path = osp.join(artery_path, 'ordinate', 'mask')
    #
    #             # extract slice files
    #             slice_files = sorted([file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
    #             label_files = sorted([file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])
    #             # extract slice information
    #
    #             risks = np.genfromtxt(osp.join(artery_path, "risk_labels.txt")).astype(np.uint8)
    #
    #             for slice_file, label_file, risk in zip(slice_files, label_files, risks):
    #                 slice_path = osp.join(image_path, slice_file)
    #                 label_path = osp.join(mask_path, label_file)
    #                 # for not 'train' mode, all the samples are loaded
    #                 if self.mode != 'train':
    #                     phases.append((slice_path, label_path))
    #                 # for 'train' mode
    #                 else:
    #                     if risk == 0:
    #                         x = random.uniform(0, 1)
    #                         if x < ratio:
    #                             phases.append((slice_path, label_path))
    #                     else:
    #                         phases.append((slice_path, label_path))
    #
    #     return phases

    def get_phases_plaque_samples(self, ratio=0.05):
        """ choose all samples with non-zero risk plus randomly choose 10% samples with zero risk """

        phases = []
        num_noncal, num_cal  = 0, 0
        with open(osp.join("../configs/{}".format(self.config), self.mode+'.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)

            for artery in sorted(listdir(sample_path)):
                artery_path = osp.join(sample_path, artery)
                # since there are less slices along ordinate/abscissa axis, we extract along ordinate axis
                image_path = osp.join(artery_path, 'ordinate', 'image')
                mask_path = osp.join(artery_path, 'ordinate', 'mask')

                # extract slice files
                slice_files = sorted([file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
                label_files = sorted([file for file in listdir(mask_path) if file.endswith('.tiff') and not file.startswith('.')])
                # extract slice information

                noncals = np.genfromtxt(osp.join(artery_path, "non_calcified_plaque_labels.txt")).astype(np.uint8)
                cals = np.genfromtxt(osp.join(artery_path, "calcified_plaque_labels.txt")).astype(np.uint8)

                for slice_file, label_file, noncal, cal in zip(slice_files, label_files, noncals, cals):
                    slice_path = osp.join(image_path, slice_file)
                    label_path = osp.join(mask_path, label_file)
                    # not in 'train' mode, all the samples are loaded
                    if self.mode != 'train':
                        phases.append((slice_path, label_path))
                    # in 'train' mode, load all abnormal slices and only a ratio of normal slices
                    else:
                        if noncal == 0 and cal == 0:
                            x = random.uniform(0, 1)
                            if x < ratio:
                                phases.append((slice_path, label_path))
                        else:
                            phases.append((slice_path, label_path))
                            if noncal != 0:
                                num_noncal += 1
                            if cal != 0:
                                num_cal += 1

        print("{} non-cals {} cals/ {} samples".format(num_noncal, num_cal, len(phases)))
        return phases

    def __len__(self):
        return len(self.phases)

    def __getitem__(self, inx):
        if self.mode=='train' and self.bc_learning is not None: # BC learning
            slice_path1, label_path1 = self.phases[random.randint(0, len(self.phases) - 1)]
            slice1, label1 = self.read_multiview_sample(slice_path1, label_path1)
            slice1, label1 = self.transform((slice1, label1))

            slice_path2, label_path2 = self.phases[random.randint(0, len(self.phases) - 1)]
            slice2, label2 = self.read_multiview_sample(slice_path2, label_path2)
            slice2, label2 = self.transform((slice2, label2))

            # Mix two images
            r = random.uniform(0, 1)
            if self.bc_learning == 'bc_plus':
                g1 = slice1.std()
                g2 = slice2.std()
                p = 1.0 / (1 + g1 / g2 * (1 - r) / r)
                sample_img = ((slice1 * p + slice2 * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2)).float()
            elif self.bc_learning == 'bc':
                sample_img = (slice1 * r + slice2 * (1 - r)).float()

            # Mix two labels
            encoded_label1 = torch.zeros(self.n_classes, *list(label1.size()))
            encoded_label1.scatter_(0, label1.unsqueeze(0), 1)
            encoded_label2 = torch.zeros(self.n_classes, *list(label1.size()))
            encoded_label2.scatter_(0, label2.unsqueeze(0), 1)

            sample_mask = encoded_label1 * r + encoded_label2 * (1 - r)

        else:
            slice_path, label_path = self.phases[inx]
            slice, label = self.read_multiview_sample(slice_path, label_path)
            sample_img, sample_mask = self.transform((slice, label))

        return (sample_img, sample_mask)

    def read_multiview_sample(self, slice_path, label_path):
        """ read multiview sample (image and mask) from slice_path and label_path """
        if self.multi_view:
            axis_names = ['applicate', 'abscissa', 'ordinate']
        else:
            axis_names = ['applicate']

        for a_inx, axis_name in enumerate(axis_names):
            slice_path_axis = slice_path.replace('ordinate', axis_name)
            label_path_axis = label_path.replace('ordinate', axis_name)
            try:
                slice_axis = io.imread(slice_path_axis)
            except Exception as e:
                print("{} error happened in {}".format(e, slice_path))
                slice_axis = np.zeros((512, 512), dtype=np.int16)
            try:
                label_axis =  io.imread(label_path_axis)
            except Exception as e:
                print("{} error happened in {}".format(e, label_path))
                label_axis = np.zeros((512, 512), dtype=np.uint8)

            if axis_name == 'applicate':
                new_h, new_w = slice_axis.shape
                slice = np.zeros((*slice_axis.shape, len(axis_names)), dtype=np.int16)
                label = label_axis

            else:
                # if slice size doesn't match with each other, resize them into the same as applicate slice
                if slice_axis.shape != (new_h, new_w):
                    slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                  preserve_range=True).astype(np.int16)
            slice[:, :, a_inx] = slice_axis

            return slice, label

class CPRPlaquePlotDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', multi_view=False, config='config'):
        """ dataloader for plotting the test results """
        super(CPRPlaquePlotDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.multi_view = multi_view # whether to use multi-view inputs or not
        self.config = config

        self.phases = self.get_phases()

    def get_phases(self):
        phases = []
        with open(osp.join("../configs/{}".format(self.config), self.mode+'.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)

            for artery in sorted(listdir(sample_path)):
                artery_path = osp.join(sample_path, artery)
                # since there are less slices along ordinate/abscissa axis, we extract along ordinate axis
                image_path = osp.join(artery_path, 'ordinate', 'image')
                mask_path = osp.join(artery_path, 'ordinate', 'mask')
                phases.append((image_path, mask_path))

        return phases

    def __len__(self):
        return len(self.phases)

    def __getitem__(self, inx):
        image_path, mask_path = self.phases[inx]
        sample_name = '/'.join(image_path.split('/')[-4:-2])

        # extract slice files
        slice_files = sorted(
            [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
        start_file, end_file = slice_files[0], slice_files[-1]
        start, end = int(start_file.split('.')[0]), int(end_file.split('.')[0])

        for s_inx, slice_file in enumerate(slice_files):
            if self.multi_view:
                axis_names = ['applicate', 'abscissa', 'ordinate']
            else:
                axis_names = ['applicate']

            for a_inx, axis_name in enumerate(axis_names):
                slice_path_axis = osp.join(image_path.replace('ordinate', axis_name), slice_file)
                label_path_axis = osp.join(mask_path.replace('ordinate', axis_name), slice_file)

                slice_axis = io.imread(slice_path_axis)
                label_axis =  io.imread(label_path_axis)

                if axis_name == 'applicate':
                    new_h, new_w = slice_axis.shape
                    slice = np.zeros((*slice_axis.shape, len(axis_names)), dtype=np.int16)
                    label = label_axis
                else:
                    if slice_axis.shape != (new_h, new_w):
                        slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                      preserve_range=True).astype(np.int16)
                slice[:, :, a_inx] = slice_axis

            slice, label = self.transform((slice, label))

            if s_inx == 0:
                # for Hyper DenseNet input (this part can be omitted cause Hyper DenseNet doesn't work well)
                if isinstance(slice, list):
                    sample_img = [torch.zeros((len(slice_files), *list(slice[0].size())), dtype=slice[0].dtype)
                              for _ in range(len(slice))]
                else:
                    sample_img = torch.zeros((len(slice_files), *list(slice.size())), dtype=slice.dtype)

                sample_mask = torch.zeros((len(slice_files), *list(label.size())), dtype=label.dtype)

            if isinstance(slice, list):
                for i in range(len(slice)):
                    sample_img[i][s_inx] = slice[i]
            else:
                sample_img[s_inx] = slice

            sample_mask[s_inx] = label

        return (sample_img, sample_mask, sample_name, start)

def dataloader(data_dir, transform, mode, metric_prev_epoch=None, phases_prev_epoch=None, shuffle=True,
               is_hard_mining=False, num_workders=8, batch_size= 128, percentile=85, multi_view=False, only_plaque=False,
               config='config', bc_learning=None):

    cprplaque_data = CPRPlaqueTrainDataset(data_dir, metric_prev_epoch, phases_prev_epoch, transform, mode, is_hard_mining,
                                      percentile, multi_view, only_plaque, config, bc_learning)
    dataloader = DataLoader(dataset=cprplaque_data, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workders, worker_init_fn=lambda x: random.seed(x))

    return dataloader

def read_train_data(data_dir, transform, mode='train', metric_prev_epoch=None, phases_prev_epoch=None, shuffle=True,
              is_hard_mining=False, num_workers= 8, batch_size=128, percentile=85, multi_view=False, onlyrisk=False,
              config='config', bc_learning=None):
    """ read images and masks into DataLoader object in either train or test mode """
    dataloaders = {}
    phases = ['train', 'val', 'test'] if mode=='train' else ['test']
    transform['val'] = transform['test']
    for phase in phases:
        dataloaders[phase] = dataloader(data_dir, transform[phase], phase, metric_prev_epoch, phases_prev_epoch,
                shuffle, is_hard_mining, num_workers, batch_size, percentile, multi_view, onlyrisk, config, bc_learning)

    return dataloaders

def read_plot_data(data_dir, transform, plot_data, shuffle=False, num_workers= 16, batch_size=128, multi_view=False,
                   config='config'):
    """ read data for test and plot """
    dataloaders = {}
    transform['val'] = transform['test']
    cprplaque = CPRPlaquePlotDataset(data_dir, transform[plot_data], plot_data, multi_view, config)
    dataloaders[plot_data] = DataLoader(dataset=cprplaque, shuffle=shuffle, num_workers=num_workers,
                                        batch_size=1, worker_init_fn=lambda x: random.seed(x))

    return dataloaders

def show_train_dataloader():
    start = time.time()
    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    sample_dir = "./data_samples"
    bound_weight_dir = "./bound_weight"
    trans_params = {
        'rescale' : 96,
        'central_crop' : 160,
        'output_channel' : 3,
        'mode' : 'train',
        'num_workers' : 16,
        'batch_size' : 64,
        'bc_learning' : None,
        'do_plot' : True,
        'onlyrisk' : False
    }

    if trans_params['output_channel'] == 2:
        ToMask = Gray2Binary()
    elif trans_params['output_channel'] == 3:
        ToMask = Gray2Triple()
    elif trans_params['output_channel'] == 4:
        ToMask = Gray2TripleWithBound(n_classes=4)
    elif trans_params['output_channel'] == 5:
        ToMask = Gray2TripleWithBound(n_classes=5)

    composed = {'train': transforms.Compose([CentralCrop(trans_params['central_crop']),
                                     Rescale(trans_params['rescale']),
                                     ToMask,
                                     ToTensor()]),
                'test': transforms.Compose([CentralCrop(trans_params['central_crop']),
                                     Rescale(trans_params['rescale']),
                                     ToMask,
                                     ToTensor()])}

    dataloaders = read_train_data(data_dir, composed, trans_params['mode'], None, None, False, False,
                trans_params['num_workers'], trans_params['batch_size'], 85, False, trans_params['onlyrisk'], 'config', trans_params['bc_learning'])

    phases = ['train'] if trans_params['mode'] == 'train' else ['test']

    num_pixel = np.zeros(trans_params['output_channel'], dtype=np.uint32)
    # scan dataloader and calculate weight for each class
    for phase in phases:
        fig_phase = phase if trans_params['bc_learning'] is None else phase+'_bc'
        fig_dir = osp.join(sample_dir, fig_phase)
        bound_fig_dir = osp.join(bound_weight_dir, phase)

        if not osp.exists(bound_fig_dir):
            os.makedirs(bound_fig_dir)

        if not osp.exists(fig_dir):
            os.makedirs(fig_dir)
        for i, sample in enumerate(dataloaders[phase]):
            image, mask = sample
            print("image size: {}".format(image.shape))
            print("mask size: {}".format(mask.shape))

            if image.size(1) == 1:
                image = torch.squeeze(image, dim=1).numpy()
            else:
                image = image[:,0].numpy()

            mask_var = Variable(mask.cuda())
            weights = bound_weight(mask_var, w0=100.0, sigma=1.0, n_classes=trans_params['output_channel']).cpu().numpy()
            fig_name = bound_fig_dir + "/" + "{}".format(i)
            sample_stack(weights, rows=10, cols=10, start_with=0, show_every=1, scale=4, fig_name = fig_name)

            mask = mask.numpy() # [N, H, W]

            # for plot data samples
            if trans_params['do_plot']:
                if trans_params['bc_learning'] is None or phase != 'train':
                    data = np.concatenate([np.stack((input, label)) for (input, label) in zip(image, mask)], axis=0)
                    fig_name =  fig_dir + "/" + "{}".format(i)
                    sample_stack(data, rows=100, cols=2, start_with=0, show_every=1, fig_name=fig_name)
                else:
                    data = np.concatenate([np.stack((input, *label)) for (input, label) in zip(image, mask)], axis=0)
                    fig_name = fig_dir + "/" + "{}".format(i)
                    sample_stack(data, rows=100, cols=1+trans_params['output_channel'], start_with=0, show_every=1, fig_name=fig_name)

            for i, label in enumerate(mask):
                for j in range(trans_params['output_channel']):
                    num_pixel[j] += np.sum(label == j)

            print("# of pixels for each class: {}".format(num_pixel))

    # nlf_weight = -1.0 * np.log(num_pixel / float(num_pixel.sum()))
    # print("negative log frequency weight: {}".format(nlf_weight))
    # if trans_params['onlyrisk']:
    #     np.save("./nlf_weight_onlyrisk.npy", nlf_weight)
    # else:
    #     np.save("./nlf_weight_all_{}.npy".format(trans_params['output_channel']), nlf_weight)
    #
    # class_freq = num_pixel / num_pixel.sum()
    # mfb_weight = np.median(class_freq) / class_freq
    # print("median frequency balancing weight: {}".format(mfb_weight))
    # if trans_params['onlyrisk']:
    #     np.save("./mfb_weight_onlyrisk.npy", mfb_weight)
    # else:
    #     np.save("./mfb_weight_all_{}.npy".format(trans_params['output_channel']), mfb_weight)

def cal_mean_std_dataloader():
    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    trans_params = {
        'rescale' : 96,
        'central_crop' : 160,
        'output_channel' : 5,
        'mode' : 'train',
        'num_workers' : 16,
        'batch_size' : 64,
        'bc_learning' : None,
        'do_plot' : False,
        'onlyrisk' : False
    }

    if trans_params['output_channel'] == 2:
        ToMask = Gray2Binary()
    elif trans_params['output_channel'] == 3:
        ToMask = Gray2Triple()
    elif trans_params['output_channel'] == 4:
        ToMask = Gray2TripleWithBound(n_classes=4)
    elif trans_params['output_channel'] == 5:
        ToMask = Gray2TripleWithBound(n_classes=5)

    composed = {'train': transforms.Compose([RandomRotation(),
                                     RandomFlip(),
                                     RandomCentralCrop(),
                                     Rescale(trans_params['rescale']),
                                     ToMask,
                                     ToTensor(norm=False)]),
                'test': transforms.Compose([CentralCrop(trans_params['central_crop']),
                                     Rescale(trans_params['rescale']),
                                     ToMask,
                                     ToTensor(norm=False)])}

    dataloaders = read_train_data(data_dir, composed, trans_params['mode'], None, None, True, False,
                trans_params['num_workers'], trans_params['batch_size'], 85, False, trans_params['onlyrisk'], 'config', trans_params['bc_learning'])

    phases = ['train'] if trans_params['mode'] == 'train' else ['test']

    images = []
    # scan dataloader and calculate weight for each class
    for phase in phases:
        for i, sample in enumerate(dataloaders[phase]):
            image, mask = sample
            print("image size: {}".format(image.shape))
            print("mask size: {}".format(mask.shape))

            if image.size(1) == 1:
                image = torch.squeeze(image, dim=1).numpy()
            else:
                image = image[:,0].numpy()
            print("max HU: {}, min HU: {}, mean HU: {}".format(image.max(), image.min(), image.mean()))

            images.append(image)

    images = np.concatenate(images, axis=0)
    img_mean = images.mean()
    img_std = images.std()
    print("Mean of all pixels: {}".format(img_mean))
    print("Std of all pixels: {}".format(img_std))

if __name__ == "__main__":
    cal_mean_std_dataloader()
    # plaque_statistic_multi_preocess(num_workers=18)