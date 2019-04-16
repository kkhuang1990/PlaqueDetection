# _*_ coding: utf-8 _*_

""" functions used to load images and masks """

import matplotlib as mpl
mpl.use('Agg')

import os
import os.path as osp
from os import listdir
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from skimage import io
from skimage import transform

from .transforms import ToTensor, RandomCrop, GaussianCrop, HU2Gray, CentralCrop, Gray2Binary, Gray2Mask
from .transforms import RandomRotation, RandomFlip, RandomCentralCrop, Rescale
from torchvision import transforms
from vision import sample_stack

class CPRPlaqueTrainDataset(Dataset):
    """ dataloader of train and validation dataset.
        Patches are randomly extracted within the central part of given volume
    """

    def __init__(self, data_dir, metric_prev_epoch = None, phases_prev_epoch = None, transform = None, mode = 'train',
                is_hard_mining = False, percentile = 85, multi_view = False, interval=32, down_sample=1, config='config'):
        """ read images from img_dir and save them into a list
        Args:
            data_dir: string, from where to read image
            transform: transform, what transforms to operate on input images
            interval: int, interval of sub-volume
            n_samples_art: int, how many samples to extract per artery
            hard_mining: bool, whether use bad mining or not
            metric_prev_epoch: numpy ndarray, metric obtained from the previous epoch
            phases_prev_epoch: list, phases of the previous epoch
            multi_view: whether use multi_view input or not
            interval: int, how many slices in one batch volume
            down_sample: int, down sampling rate (every how many slices)
        """

        super(CPRPlaqueTrainDataset, self).__init__()
        self.interval = interval
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform
        self.down_sample = down_sample
        self.is_hard_mining = is_hard_mining
        self.percentile = percentile
        self.multi_view = multi_view  # whether to use multi-view inputs or not
        self.slice_range = self.interval * self.down_sample
        self.config = config

        # initialize phases for different modes
        if self.mode == 'train':
            self.phases = self.update_phases(metric_prev_epoch, phases_prev_epoch)
        else:
            self.phases = self.get_phases()

    def update_phases(self, metric_prev_epoch, phases_prev_epoch):
        """ update the phases by mining the bad samples
        :return: phases: refined phases after mining the bad samples
        """
        if phases_prev_epoch is None:
            phases = self.get_phases()

        else:
            if self.is_hard_mining:
                thres = np.percentile(metric_prev_epoch, self.percentile)
                phases = [phase for phase, metric in zip(phases_prev_epoch, metric_prev_epoch) if metric <= thres]
            else:
                phases = phases_prev_epoch

        return phases

    def __len__(self):
        return len(self.phases)

    def get_phases(self):
        phases = []
        with open(osp.join('../configs/{}'.format(self.config), self.mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)

            for artery in sorted(listdir(sample_path)):
                # artery_path = osp.join(sample_path, artery)
                image_path = osp.join(sample_path, artery, 'ordinate', 'image')
                mask_path = osp.join(sample_path, artery, 'ordinate', 'mask')
                # extract slice files
                slice_files = sorted(
                    [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])

                start_file, end_file = slice_files[0], slice_files[-1]
                start, end = int(start_file.split('.')[0]), int(end_file.split('.')[0])
                for s_inx in range(start, end + 1 - self.slice_range + self.down_sample):
                    phases.append((image_path, mask_path, s_inx))

        print("{} : {} samples".format(self.mode, len(phases)))
        return phases


    def __getitem__(self, inx):
        sample = self.phases[inx]
        image_path, mask_path, rand_inx  = sample

        if self.multi_view:
            axis_names = ['applicate', 'abscissa', 'ordinate']
        else:
            axis_names = ['applicate']

        for a_inx, axis_name in enumerate(axis_names):
            image_path_axis = image_path.replace('ordinate', axis_name)
            mask_path_axis = mask_path.replace('ordinate', axis_name)

            slice_files_axis = [osp.join(image_path_axis, "{:03d}.tiff".format(i))
                           for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)]
            label_files_axis = [osp.join(mask_path_axis, "{:03d}.tiff".format(i))
                           for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)]

            image_axis = np.stack([io.imread(slice_file) for slice_file in slice_files_axis])
            mask_axis = np.stack([io.imread(label_file) for label_file in label_files_axis])

            if axis_name == 'applicate':
                new_d, new_h, new_w = image_axis.shape
                image = np.zeros((*image_axis.shape, len(axis_names)), dtype=np.int16)
                image[:, :, :, a_inx] = image_axis
                mask = mask_axis

            else:
                # if slice size doesn't match with each other, resize them into the same as applicate slice
                for s_inx in range(new_d):
                    slice_axis = image_axis[s_inx]
                    if slice_axis.shape != (new_h, new_w):
                        slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                  preserve_range=True).astype(np.int16)
                    image[s_inx, :, :, a_inx] = slice_axis
        # transform 3D image and mask
        sample_img, sample_mask = self.transform((image, mask))

        return (sample_img, sample_mask)

class CPRPlaqueTestDataset(Dataset):
    """ dataloader for test dataset
        the whole artery is extracted with given stride along applicate axis
    """

    def __init__(self, data_dir, transform = None, mode = 'train', multi_view = False, interval=32, down_sample=1,
                 config='config'):
        """ read images from img_dir and save them into a list """

        super(CPRPlaqueTestDataset, self).__init__()
        self.interval = interval
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform
        self.down_sample = down_sample
        self.multi_view = multi_view  # whether to use multi-view inputs or not
        self.stride = self.interval * self.down_sample
        self.config = config
        self.phases = self.get_phases()

    def get_phases(self):
        phases = []
        with open(osp.join('../configs/{}'.format(self.config), self.mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)

            for artery in sorted(listdir(sample_path)):
                # artery_path = osp.join(sample_path, artery)
                image_path = osp.join(sample_path, artery, 'ordinate', 'image')
                mask_path = osp.join(sample_path, artery, 'ordinate', 'mask')

                phases.append((image_path, mask_path))

        print("{} : {} samples".format(self.mode, len(phases)))
        return phases

    def __len__(self):
        return len(self.phases)

    def __getitem__(self, inx):
        sample = self.phases[inx]
        image_path, mask_path = sample
        sample_name = '/'.join(image_path.split('/')[-4:-2])

        # extract slice files
        slice_files = sorted(
            [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
        start_file, end_file = slice_files[0], slice_files[-1]
        start, end = int(start_file.split('.')[0]), int(end_file.split('.')[0])

        n_sample = len(range(start, end + 2 - self.stride, self.stride)) * self.down_sample
        for s_inx in range(start, end + 2 - self.stride, self.stride):
            for shift in range(self.down_sample):
                if self.multi_view:
                    axis_names = ['applicate', 'abscissa', 'ordinate']
                else:
                    axis_names = ['applicate']

                for a_inx, axis_name in enumerate(axis_names):
                    image_path_axis = image_path.replace('ordinate', axis_name)
                    mask_path_axis = mask_path.replace('ordinate', axis_name)

                    slice_files_axis = [osp.join(image_path_axis, "{:03d}.tiff".format(i))
                                   for i in range(s_inx + shift, s_inx + shift + self.stride, self.down_sample)]
                    label_files_axis = [osp.join(mask_path_axis, "{:03d}.tiff".format(i))
                                   for i in range(s_inx + shift, s_inx + shift + self.stride, self.down_sample)]

                    image_axis = np.stack([io.imread(slice_file) for slice_file in slice_files_axis])
                    mask_axis = np.stack([io.imread(label_file) for label_file in label_files_axis])

                    if axis_name == 'applicate':
                        new_d, new_h, new_w = image_axis.shape
                        image = np.zeros((*image_axis.shape, len(axis_names)), dtype=np.int16)
                        image[:, :, :, a_inx] = image_axis
                        mask = mask_axis

                    else:
                        # if slice size doesn't match with each other, resize them into the same as applicate slice
                        for slice_inx in range(new_d):
                            slice_axis = image_axis[slice_inx]
                            if slice_axis.shape != (new_h, new_w):
                                slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                              preserve_range=True).astype(np.int16)
                            image[slice_inx, :, :, a_inx] = slice_axis

                image, mask = self.transform((image, mask))

                if s_inx == start and shift == 0:
                    if isinstance(image, list):
                        sample_img = [torch.zeros([n_sample, *list(image.size())]).float() for _ in range(len(image))]
                    else:
                        sample_img = torch.zeros([n_sample, *list(image.size())]).float()

                    sample_mask = torch.zeros([n_sample, *list(mask.size())]).long()

                if isinstance(image, list):
                    for i in range(len(image)):
                        sample_img[i][(s_inx-start) // self.interval + shift] = image[i]
                else:
                    sample_img[(s_inx - start) // self.interval + shift] = image

                sample_mask[(s_inx-start) // self.interval + shift] = mask

        return (sample_img, sample_mask, sample_name, start)

def read_train_data(data_dir, metric_prev_epoch = None, phases_prev_epoch = None, transform = None, mode = 'train',
                    is_hard_mining = False, percentile = 85, multi_view = False, interval=32, down_sample=1,
                    batch_size= 32, num_workers= 12, shuffle=True, config='config'):
    """ read data for train/validation """
    dataloaders = {}
    phases = ['train', 'val', 'test'] if mode=='train' else ['test']
    transform['val'] = transform['test']
    for phase in phases:
        cprplaque = CPRPlaqueTrainDataset(data_dir, metric_prev_epoch, phases_prev_epoch, transform[phase],
                    phase, is_hard_mining, percentile, multi_view, interval, down_sample, config)
        dataloaders[phase] = DataLoader(dataset=cprplaque, shuffle=shuffle,
                                              num_workers=num_workers, batch_size= batch_size)

    return dataloaders

def read_plot_data(data_dir, transform, plot_data, multi_view=False, interval=16, down_sample=1,
                   num_workers= 16, shuffle=False, config='config'):
    """ read data for train/validation """
    dataloaders = {}
    transform['val'] = transform['test']
    cprplaque = CPRPlaqueTestDataset(data_dir, transform[plot_data], plot_data, multi_view, interval, down_sample, config)
    dataloaders[plot_data] = DataLoader(dataset=cprplaque, shuffle=shuffle,
                                          num_workers=num_workers, batch_size=1)

    return dataloaders


def show_dataloader():
    """ show each data sample to verify the correctness of dataloader """

    since = time.time()
    data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    # data_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # data_dir = "/Users/AlbertHuang/CT_Anomaly_Detection/Plaque_CPR/20180213"
    trans_params = {
        'central_crop' : 160,
        'random_crop' : (64, 64),
        'rescale' : (64, 64),
        'output_channel' : 5
    }
    composed = {'train': transforms.Compose([HU2Gray(),
                                   RandomCentralCrop(),
                                   RandomRotation(),
                                   RandomFlip(),
                                   Rescale(trans_params['rescale']),
                                   Gray2Binary() if trans_params['output_channel'] == 2 else Gray2Mask(),
                                   # AddNoise(),
                                   # RandomTranslate(),
                                   ToTensor()]),
                'test': transforms.Compose([HU2Gray(),
                                            CentralCrop(160),
                                            Rescale(trans_params['rescale']),
                                            Gray2Binary() if trans_params['output_channel'] == 2 else Gray2Mask(),
                                            ToTensor()])}


    dataloaders = read_train_data(data_dir, None, None, composed, 'train', False, 85, True, interval=32,
                                  down_sample=1, batch_size=8, num_workers=8, shuffle=True)

    # dataloaders = read_test_data(data_dir, composed, 'test', True, interval=16, down_sample=1, num_workers=8, shuffle=False)

    num_pixel = np.zeros(5, dtype=np.uint32)
    for inx in range(1):
        datasizes = {'train':0, 'val':0}
        # datasizes = {'test': 0}
        for phase in ['train', 'val']:
        # for phase in ['test']:
            # des_path = osp.join(des_dir, phase, str(inx))
            # if not osp.exists(des_path):
            #     os.makedirs(des_path)
            for i, sample in enumerate(dataloaders[phase]):
                image, mask  = sample
                print("image size: {}".format(image.size()))
                print("mask size: {}".format(mask.size()))
                if image.size(1) == 1:
                    image_np = image.squeeze(1).numpy()
                else:
                    image_np = image[:, 0, ::].numpy()

                mask_np = mask.numpy()
                img_dir = "./data_samples/{}".format(i)
                if not osp.exists(img_dir):
                    os.makedirs(img_dir)
                image_name = "./data_samples/{}/image".format(i)
                mask_name = "./data_samples/{}/mask".format(i)
                sample_stack(image_np[0], rows=10, cols=10, start_with=0, show_every=2, scale=4, fig_name=image_name)
                sample_stack(mask_np[0], rows=10, cols=10, start_with=0, show_every=2, scale=4, fig_name=mask_name)

                # image, mask, _, _ = sample
                datasizes[phase] += image.size(0)
                for i, label in enumerate(mask.numpy()):
                    for j in range(trans_params['output_channel']):
                        num_pixel[j] += np.sum(label == j)

    #     print("Train: {}, Val: {}".format(datasizes['train'], datasizes['val']))
    # class_freq = num_pixel / num_pixel.sum()
    # class_weight = np.median(class_freq) / class_freq

    # print("median frequency balancing: {}".format(class_weight))
    # np.save("./class_weight_mfb.npy", class_weight)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

if __name__ == "__main__":
    show_dataloader()