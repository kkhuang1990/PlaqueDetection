# _*_ coding: utf-8 _*_

""" main code for train and test U-Net """

from __future__ import print_function

import sys
sys.path.append("..")

import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import argparse
import shutil
from loss import dice_score_slicewise, GeneralizedDiceLoss, WeightedKLDivLoss
from loss import WeightedCrossEntropy, FocalLoss, DiceLoss
from loss import MaxPoolLoss, CrossEntropyBoundLoss
from loss import WeightedHausdorffDistanceLoss, ModifiedWeightedHausdorffDistanceLoss
from loss import WeightedHausdorffDistanceDoubleBoundLoss, WeightedMaximumHausdorffDistanceDoubleBoundLoss
from loss import ModifiedWeightedHausdorffDistanceDoubleBoundLoss, WeightedHausdorffDistanceDoubleBoundLossWithSnake
import os.path as osp
from train import train_model, model_reference
from torchvision import transforms
from lr_scheduler import PolyLR
from image.models.deeplab_resnet import get_1x_lr_params_NOscale, get_10x_lr_params

from torch.optim import lr_scheduler

import matplotlib as mpl
mpl.use('Agg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, help="from where to read data")
    parser.add_argument('--central_crop', type=int, default=192)
    parser.add_argument('--rescale', type=int, default=96)
    parser.add_argument('--output_channel', type=int, default=5, choices=(2, 3, 4, 5))
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--w_decay', type=float, default=0.005)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--criterion', type=str, default='nll')
    parser.add_argument('--opt', type=str, default='Adam', help="optimizer")
    parser.add_argument('--weight', type=lambda x: True if x.lower()=='true' else None, default=True)
    parser.add_argument('--weight_type', type=lambda x: None if x.lower()=='none' else x, default=None)
    parser.add_argument('--only_test', type=lambda x: x.lower()=='true')
    parser.add_argument('--rotation', type=lambda x: x.lower()=='true')
    parser.add_argument('--flip', type=lambda x: x.lower()=='true')
    parser.add_argument('--r_central_crop', type=lambda x: x.lower()=='true')
    parser.add_argument('--random_trans', type=lambda x: x.lower()=='true')
    parser.add_argument('--noise', type= lambda x: x.lower()=='true', help="whether add Gaussian noise or not")
    parser.add_argument('--use_pre_train', type=lambda x: x.lower()=='true')
    parser.add_argument('--fig_dir', type=str, help="directory for saving segmentation results")
    parser.add_argument('--pre_train_path', type=str)
    parser.add_argument('--with_shallow_net', type= lambda x: x.lower()=='true')
    parser.add_argument('--n_epoch_hardmining', type=int, default=15, help="every how many epochs for hard mining")
    parser.add_argument('--percentile', type=int, default=85, help="how much percent samples to save for hard mining")
    parser.add_argument('--plot_data', type=str, default='test', help="what data to plot")
    parser.add_argument('--do_plot', type=lambda x: x.lower()=='true', help="whether plot test results or not")
    parser.add_argument('--multi_view', type=lambda x: x.lower()=='true', help="whether to use multi-view inputs")
    parser.add_argument('--model', type=str, choices=('tiramisu', 'unet', 'res_unet', 'hyper_tiramisu', 'deeplab_resnet',
                                                      'res_unet_dp', 'res_unet_reg'), help="which model to use")
    parser.add_argument('--theta', type=float, help="compression ratio for DenseNet")
    parser.add_argument('--only_plaque', type=lambda x: x.lower()=='true', help="whether only use plaque samples")
    parser.add_argument('--interval', type=int, help="interval of slices in volume")
    parser.add_argument('--down_sample', type=int, default=1, help="down sampling step")
    parser.add_argument('--model_type', type=str, default='2d', help="use 2D or 3D model")
    parser.add_argument('--config', type=str, default='config', help="config file name for train/val/test data split")
    parser.add_argument('--alpha', type=float, default=0.5, help="ratio of false positive in generalized dice loss")
    parser.add_argument('--bc_learning', type=lambda x: None if x.lower()=='false' else x, default=None)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help="learning scheduler")
    parser.add_argument('--mpl', type=lambda x: x.lower()=='true', default=False, help="whether max-pooling loss or not")
    parser.add_argument('--cal_zerogt', type= lambda x: x.lower() == 'true', default=False,
                        help= "whether calculate F1 score for case of all GT pixels are zero")
    parser.add_argument('--drop_out', type=float, default=0.0,
                        help= "drop out rate for Res-UNet model")
    parser.add_argument('--ignore_index', type=lambda x: None if x.lower()=='none' else int(x),
                        help= "ignore index")
    parser.add_argument('--w1', type=float, default=20.0, help="outer bound amptitude")
    parser.add_argument('--sigma1', type=float, default=5.0, help="outer bound variance")
    parser.add_argument('--w2', type=float, default=10.0, help="inner bound amptitude")
    parser.add_argument('--sigma2', type=float, default=5.0, help="inner bound variance")
    parser.add_argument('--bound_out', type=lambda x: x.lower()=='true', default=False,
                        help="whether output with bound")
    parser.add_argument('--width', default=1, type=int, help="bound width")
    parser.add_argument('--mod_outline', default=False, type=lambda x: x.lower()=='true',
                        help="whether modify outline or not")
    parser.add_argument('--bound_type', default='innerouter', type=str,
                        help="what kind of bound to extract: inner or outer or both")
    parser.add_argument('--whd_alpha', default=4, type=int, help="alpha in WHD loss")
    parser.add_argument('--whd_beta', default=1, type=float, help="beta in WHD loss")
    parser.add_argument('--whd_ratio', default=0.5, type=float, help="ratio in WHD loss")

    args = parser.parse_args()
    shutil.copy('./main.sh', './{}'.format(args.fig_dir)) # save current bash file for replicating experiment results

    args.model_save_name = "./{}/model.pth".format(args.fig_dir)

    # transforms and augmentations
    if args.model_type == '2d':
        from image.transforms import Gray2TripleWithBound, Gray2Bound, Normalize, Gray2InnerOuterBound
        from image.transforms import Gray2InnerBound, Gray2OuterBound
        from image.transforms import CentralCrop, Rescale, Gray2Triple, Gray2Mask, ToTensor, Gray2Binary, Identical, HU2Gray, RandomFlip
        from image.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, HU2GrayMultiStreamToTensor
    else:  # 2.5D and 3D
        from volume.transforms import Gray2InnerBound, Gray2OuterBound, Gray2InnerOuterBound
        from volume.transforms import Gray2Bound, Gray2TripleWithBound
        from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Gray2Binary, Identical, HU2Gray, RandomFlip
        from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, HU2GrayMultiStreamToTensor

    # choose transforms of annotation under different settings
    if args.output_channel == 2: # 2 options: (1) binary class seg (2) bound detection
        if args.bound_out:
            if args.bound_type == 'inner':
                ToMask = Gray2InnerBound(width=args.width)
            elif args.bound_type == 'outer':
                ToMask = Gray2OuterBound(width=args.width)
            elif args.bound_type == 'innerouter':
                ToMask = Gray2Bound(width=args.width, n_classes=args.output_channel)
        else:
            ToMask = Gray2Binary()

    elif args.output_channel == 3: # 2 options: (1) triple class seg (2) inner bound + outer bound
        if args.bound_out:
            ToMask = Gray2InnerOuterBound(width=args.width)
        else:
            ToMask = Gray2Triple()

    elif args.output_channel == 4:
        ToMask = Gray2TripleWithBound(n_classes=4, width=args.width)

    elif args.output_channel == 5:
        if args.bound_out:
            ToMask = Gray2TripleWithBound(n_classes=5, width=args.width)
        else:
            ToMask = Gray2Mask()

    args.compose = {'train': transforms.Compose([HU2Gray(),
                                                 RandomRotation() if args.rotation else Identical(),
                                                 RandomFlip() if args.flip else Identical(),
                                                 RandomCentralCrop() if args.r_central_crop else CentralCrop(args.central_crop),
                                                 # Rescale((args.rescale)),
                                                 RandomTranslate() if args.random_trans else Identical(),
                                                 AddNoise() if args.noise else Identical(),
                                                 ToMask,
                                                 ToTensor(norm=True)]),

                    'test': transforms.Compose([HU2Gray(),
                                                CentralCrop(args.central_crop),
                                                # Rescale(args.rescale),
                                                ToMask,
                                                ToTensor(norm=True)])}

    # whether use pre_train model or not
    if args.use_pre_train:
        model = torch.load("{}/model.pth".format(args.pre_train_path),
                           map_location=lambda storage, loc: storage)
    else:
        args.color_channel = 3 if args.multi_view else 1

        if args.model_type == '2d':
            if args.model == 'unet':
                if args.with_shallow_net:
                    from image.models.unet import UNet18 as UNet
                else:
                    from image.models.unet import UNet28 as UNet

                model = UNet(args.color_channel, args.output_channel)

            elif args.model == 'res_unet':
                print("res_unet is called")
                if args.with_shallow_net:
                    from image.models.res_unet import ResUNet18 as ResUNet
                else:
                    from image.models.res_unet import ResUNet28 as ResUNet

                model = ResUNet(args.color_channel, args.output_channel)

            elif args.model == 'res_unet_dp':
                print("res_unet is called")
                if args.with_shallow_net:
                    from image.models.res_unet_dp import ResUNet18 as ResUNet
                else:
                    from image.models.res_unet_dp import ResUNet28 as ResUNet

                model = ResUNet(args.color_channel, args.output_channel, args.drop_out)


            elif args.model == 'tiramisu':
                if args.with_shallow_net:
                    from image.models.tiramisu import FCDenseNet43 as FCDenseNet
                else:
                    from image.models.tiramisu import FCDenseNet67 as FCDenseNet

                model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

            # elif args.model == 'hyper_tiramisu':
            #     if args.with_shallow_net:
            #         from image.models.hyper_tiramisu import FCDenseNet43 as FCDenseNet
            #     else:
            #         from image.models.hyper_tiramisu import FCDenseNet67 as FCDenseNet
            #
            #     model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

            elif args.model == 'deeplab_resnet':
                from image.models.deeplab_resnet import Res_Ms_Deeplab
                model = Res_Ms_Deeplab(args.color_channel, args.output_channel)

        elif args.model_type == '3d':
            if args.model == 'unet':
                if args.with_shallow_net:
                    from volume.models.unet import UNet18 as UNet
                else:
                    from volume.models.unet import UNet28 as UNet

                model = UNet(args.color_channel, args.output_channel)

            elif args.model == 'res_unet': # for 3D network, Res-UNet and Res-UNet with dropout is not distinguished
                print("res_unet is called")
                if args.with_shallow_net:
                    from volume.models.res_unet import ResUNet18 as ResUNet
                else:
                    from volume.models.res_unet import ResUNet28 as ResUNet

                model = ResUNet(args.color_channel, args.output_channel, args.drop_out)

            elif args.model == 'tiramisu':
                if args.with_shallow_net:
                    from volume.models.tiramisu import FCDenseNet43 as FCDenseNet
                else:
                    from volume.models.tiramisu import FCDenseNet67 as FCDenseNet

                model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

            # elif args.model == 'hyper_tiramisu':
            #     if args.with_shallow_net:
            #         from volume.models.hyper_tiramisu import FCDenseNet43 as FCDenseNet
            #     else:
            #         from volume.models.hyper_tiramisu import FCDenseNet67 as FCDenseNet
            #
            #     model = FCDenseNet(args.color_channel, args.output_channel, args.theta)

        elif args.model_type == "2.5d": # Hybrid model with 3D input and 2D output
            if args.model == 'res_unet':
                print("Hybrid res_unet is called")
                if args.with_shallow_net:
                    from hybrid.models.hybrid_res_unet import ResUNet18 as ResUNet # 15 slices
                else:
                    from hybrid.models.hybrid_res_unet import ResUNet23 as ResUNet # 31 slices

                model = ResUNet(args.color_channel, args.output_channel, args.interval, args.rescale)
            elif args.model == 'res_unet_reg':
                # hybrid res-unet with regularization in original paper which introduced WHD loss
                print("Hybrid res_unet is called")
                if args.with_shallow_net:
                    from hybrid.models.hybrid_res_unet_reg import ResUNet18 as ResUNet # 15 slices
                else:
                    from hybrid.models.hybrid_res_unet_reg import ResUNet23 as ResUNet # 31 slices

                model = ResUNet(args.color_channel, args.output_channel, args.interval, args.rescale)

    # whether use gpu or not
    if args.use_gpu:
        model = model.cuda()

    # whether introduce prior weight into loss function or not
    if args.weight:
        if args.weight_type is None:
            if args.bound_out:
                weight = torch.from_numpy(np.load('../class_weights/nlf_weight_all_bound_{}.npy'.format(args.output_channel))).float()
            else:
                if args.output_channel == 5:
                    weight = torch.from_numpy(np.load('../class_weights/class_weight.npy')).float()
                else:
                    weight = torch.from_numpy(np.load('../class_weights/nlf_weight_all_{}.npy'.format(args.output_channel))).float()

        weight = Variable(weight.cuda())

    else: # no prior weight, especially for bound detection
        weight = args.weight
    print("weight: {}".format(weight))

    # criterion
    if args.criterion == 'nll':
        criterion = nn.NLLLoss(weight=weight)
    elif args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif args.criterion == 'dice':
        criterion = DiceLoss(weight=weight, ignore_index=None, weight_type=args.weight_type, cal_zerogt=args.cal_zerogt)

    elif args.criterion == 'gdl_none':
        criterion = GeneralizedDiceLoss(weight=weight, ignore_index=None, weight_type=None,
                                        alpha=args.alpha)

    elif args.criterion == 'focal':
        criterion = FocalLoss()
    elif args.criterion == 'wce':
        criterion = WeightedCrossEntropy()
    elif args.criterion == 'ceb': # cross entropy bound loss
        criterion = CrossEntropyBoundLoss(n_classes=args.output_channel, weight=weight, ignore_index=args.ignore_index,
                                         ws=[args.w1, args.w2], sigmas=[args.sigma1, args.sigma2], bound_output=args.bound_out)

    elif args.criterion == 'whd': # weighted Hausdorff distance
        criterion = WeightedHausdorffDistanceLoss(alpha=args.whd_alpha, beta=args.whd_beta)
    elif args.criterion == 'mwhd':
        criterion = ModifiedWeightedHausdorffDistanceLoss(alpha=args.whd_alpha, thres=0.5)
    elif args.criterion == 'whddb': # whd with double bounds
        criterion = WeightedHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=True, alpha=args.whd_alpha,
                                                             beta=args.whd_beta, ratio=args.whd_ratio)
    elif args.criterion == 'whddbmax': # maximum whd with double bounds
        criterion = WeightedMaximumHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=True, alpha=args.whd_alpha,
                                                             beta=args.whd_beta, ratio=args.whd_ratio)
    elif args.criterion == 'mwhddb': # modified whd with double bounds
        criterion = ModifiedWeightedHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=True, alpha=args.whd_alpha,
                                                             beta=args.whd_beta, ratio=args.whd_ratio)
    elif args.criterion == "whddbsnake":
        criterion = WeightedHausdorffDistanceDoubleBoundLossWithSnake(return_multi_loss=True, alpha=args.whd_alpha,
                                                             beta=args.whd_beta, ratio=args.whd_ratio)
    # whddb loss with regularization
    elif args.criterion == "whddb_cereg":
        criterion = WeightedHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=True, alpha=args.whd_alpha, beta=args.whd_beta, ratio=args.whd_ratio)

    # criterion for BC learning
    if args.criterion.startswith('gdl'):
        args.criterion_bc = criterion
    else:
        args.criterion_bc = WeightedKLDivLoss(weight=weight)


    # optimizer
    if args.opt == 'Adam':
        if args.model == 'deeplab_resnet':
            optimizer = optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': args.lr},
                                   {'params': get_10x_lr_params(model), 'lr': 10 * args.lr}],
                                  lr=args.lr, weight_decay=args.w_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    elif args.opt == 'sgd':
        if args.model == 'deeplab_resnet':
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.lr},
                                   {'params': get_10x_lr_params(model), 'lr': 10 * args.lr}],
                                  lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)

    # learning schedule
    if args.lr_scheduler == 'StepLR':
        my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'PolyLR':
        my_lr_scheduler = PolyLR(optimizer, max_iter=args.num_train_epochs, power=0.9)

    # print arguments setting
    for arg in vars(args):
        print("{} : {}".format(arg, getattr(args, arg)))

    # plot samples used for train, val and test respectively
    print("Dataset:")
    for mode in ['train', 'val', 'test']:
        config_file = osp.join('../configs/{}'.format(args.config), mode+'.txt')
        print(mode)
        with open(config_file, 'r') as reader:
            for line in reader.readlines():
                print(line.strip('\n'))

    since = time.time()
    if not args.only_test:
        train_model(model, criterion, optimizer, my_lr_scheduler, args)

    # model reference
    model_reference(args, sample_stack_rows=50)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))