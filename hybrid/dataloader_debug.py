# _*_ coding: utf-8 _*_

""" Dataloader used for debug
    Here we want to check whether each time the same order of slices can be ensured by
    setting the random seed as fixed
"""

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
from .transforms import Gray2InnerOuterBound
from torchvision import transforms
from vision import sample_stack


class CPRPlaqueTrainDataset(Dataset):
    """ dataloader of train and validation dataset.
        Patches are randomly extracted within the central part of given volume
    """

    def __init__(self, data_dir, mode = 'train', config='config', interval=15, down_sample=1):
        """ read images from img_dir and save them into a list """
        super(CPRPlaqueTrainDataset, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.config = config
        self.interval = interval
        self.down_sample = down_sample
        self.slice_range = self.interval * self.down_sample
        self.phases = self.get_phases()

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
        path_list = image_path.split('/')
        sample_name, artery_name = path_list[-4], path_list[-3]
        path_abbrev = '/'.join([sample_name, artery_name, str(rand_inx)])

        return path_abbrev


class CPRPlaqueTestDataset(Dataset):
    """ dataloader for test dataset
        the whole artery is extracted with given stride along applicate axis, then divide them into mini-batch for test
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

    def get_phases(self): # number of arteries
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

        n_sample = len(range(start, end + 1 - self.stride + self.down_sample)) # num of samples

        for s_inx in range(start, end + 1 - self.stride + self.down_sample):
            # whether multi-view data or not
            if self.multi_view:
                axis_names = ['applicate', 'abscissa', 'ordinate']
            else:
                axis_names = ['applicate']

            # read multi-view slices and concatenate them together
            for a_inx, axis_name in enumerate(axis_names):
                image_path_axis = image_path.replace('ordinate', axis_name)
                mask_path_axis = mask_path.replace('ordinate', axis_name)

                slice_files_axis = [osp.join(image_path_axis, "{:03d}.tiff".format(i))
                               for i in range(s_inx, s_inx + self.stride, self.down_sample)]
                label_files_axis = [osp.join(mask_path_axis, "{:03d}.tiff".format(i))
                               for i in range(s_inx, s_inx + self.stride, self.down_sample)]

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
            mask = mask[self.interval // 2]  # only remain the central slice

            if s_inx == start: # the first mini-batch
                if isinstance(image, list): # for Hyper DenseNet (this part can be omitted)
                    sample_img = [torch.zeros([n_sample, *list(image.size())]).float() for _ in range(len(image))]
                else:
                    sample_img = torch.zeros([n_sample, *list(image.size())]).float()

                sample_mask = torch.zeros([n_sample, *list(mask.size())]).long()

            if isinstance(image, list):
                for i in range(len(image)):
                    sample_img[i][s_inx-start] = image[i]
            else:
                sample_img[s_inx - start] = image

            sample_mask[s_inx - start] = mask

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


def debug_dataloader(mode='train', config='config', shuffle=True, num_workers=16, batch_size=4):
    """ show each data sample to verify the correctness of dataloader """

    since = time.time()

    # def worker_init_fn(worker_id):
    #     np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Please refer to https://qiita.com/chat-flip/items/4c0b71a7c0f5f6ae437f
    torch.manual_seed(42)  # for shuffle=True

    data_dir = "/data/ugui0/antonio-t/CPR_multiview_interp2_huang"
    cprplaque = CPRPlaqueTrainDataset(data_dir, mode=mode, config=config)
    dataloader = DataLoader(dataset=cprplaque, shuffle=shuffle,
                            num_workers=num_workers, batch_size=batch_size,
                            worker_init_fn=lambda x: random.seed(x))

    paths_abbrev = []
    paths_dir = "./data_samples"
    if not osp.exists(paths_dir):
        os.makedirs(paths_dir)

    for i, sample in enumerate(dataloader):
        path_abbrev = list(sample)
        paths_abbrev += path_abbrev

    with open(osp.join(paths_dir, "{}.txt".format(random.randint(1, 1000))), 'w') as writer:
        for line in paths_abbrev:
            writer.write("%s\n" % line)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

def show_train_dataloader():
    """ show each data sample to verify the correctness of dataloader """

    since = time.time()
    # data_dir = "/home/mil/huang/Dataset/CPR_multiview"
    data_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # data_dir = "/Users/AlbertHuang/CT_Anomaly_Detection/Plaque_CPR/20180213"
    trans_params = {
        'central_crop' : 192,
        'random_crop' : (96, 96),
        'rescale' : (96, 96),
        'output_channel' : 2
    }
    composed = {'train': transforms.Compose([HU2Gray(),
                                   RandomCentralCrop(),
                                   RandomRotation(),
                                   RandomFlip(),
                                   Rescale(trans_params['rescale']),
                                   Gray2InnerOuterBound() if trans_params['output_channel'] == 2 else Gray2Mask(),
                                   # AddNoise(),
                                   # RandomTranslate(),
                                   ToTensor()]),
                'test': transforms.Compose([HU2Gray(),
                                            CentralCrop(160),
                                            Rescale(trans_params['rescale']),
                                            Gray2InnerOuterBound() if trans_params['output_channel'] == 2 else Gray2Mask(),
                                            ToTensor()])}


    # dataloaders = read_train_data(data_dir, None, None, composed, 'train', False, 85, True, interval=32,
    #                               down_sample=1, batch_size=8, num_workers=8, shuffle=True)

    dataloaders = read_plot_data(data_dir, composed, 'test', False, interval=15, down_sample=1, num_workers=8, shuffle=False)

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
    in_channels = 1
    out_channels = 2
    unet = ResUNet18(in_channels, out_channels, p=0.0)
    print(unet)
    x = torch.FloatTensor(6, 1, 31, 96, 96)  # the smallest patch size is 16 * 16
    y = unet(x)


    show_plot_dataloader()