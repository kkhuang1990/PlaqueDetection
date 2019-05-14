# _*_ coding: utf-8 _*_

""" define train and test functions here """

import matplotlib as mpl
mpl.use('Agg')

import imageio
import warnings
warnings.filterwarnings('ignore', module='imageio')

import sys
sys.path.append("..")
import matplotlib.pyplot as plt

from sklearn.metrics import auc
import copy
from collections import Counter
from vision import sample_list_hdf
import numpy as np
np.set_printoptions(precision=4)


from tqdm import tqdm
import torch
from torch.autograd import Variable

import torch.nn.functional as F
from torch import nn

import os.path as osp
import os
import pickle

from metric import cal_f_score, cal_f_score_slicewise, volumewise_hd95, volumewise_asd, volumewise_ravd
from loss import WeightedKLDivLoss

from vision import plot_metrics, plaque_detection_rate, plot_class_f1
from vision import plot_slice_wise_measures
from image.models.deeplab_resnet import adjust_learning_rate, cal_loss
from snake import probmap2bound
from loss import WeightedHausdorffDistanceDoubleBoundLoss

from utils import innerouterbound2mask

from image.models.deeplab_resnet import adjust_learning_rate, cal_loss


def train_model(model, criterion, optimizer, scheduler, args):
    """ train the model
    Args:
        model: model inheriting from nn.Module class
        criterion: criterion class, loss function used
        optimizer: optimizer, optimization strategy
        scheduler: lr scheduler
        args: parser arguments
    """

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0e9
    loss_keep = 0 # check how many times the val loss has decreased continuously
    epoch_loss_prev = 1.0e9 # loss at the previous epoch

    # metrics for each epoch
    epoch_acc = {'train': [], 'val': [], 'test': []}
    epoch_loss = {'train': [], 'val': [], 'test': []}
    epoch_loss_boundwise = {'train': [], 'val': [], 'test': []}  # for prediction and regularization respectively
    epoch_hdist = {'train': [], 'val': [], 'test': []}
    epoch_reghdf = {'train': [], 'val': [], 'test': []}
    epoch_asd = {'train': [], 'val': [], 'test': []}
    epoch_vd = {'train': [], 'val': [], 'test': []}
    epoch_f1_score = {'train': [], 'val': [], 'test': []}
    epoch_f1_score_class = {'train': [], 'val': [], 'test': []}

    # for hard mining
    metric_prev_epoch = None
    phases_prev_epoch = None

    # start training
    for epoch in range(args.num_train_epochs):
        print("{}/{}".format(epoch+1, args.num_train_epochs))
        if epoch != 0 and epoch % args.n_epoch_hardmining == 0:
            is_hard_mining = True
        else:
            is_hard_mining = False

        if args.model_type == '2d':
            from image.dataloader import read_train_data
            if args.only_plaque:  # only use samples containing plaques for training
                dataloaders = read_train_data(args.data_dir, args.compose, 'train', None, None, True,
                                    is_hard_mining, args.num_workers, args.batch_size, args.percentile, args.multi_view,
                                    args.only_plaque, args.config, args.bc_learning)
            else:
                dataloaders = read_train_data(args.data_dir, args.compose, 'train', metric_prev_epoch, phases_prev_epoch, True,
                                  is_hard_mining, args.num_workers, args.batch_size, args.percentile, args.multi_view,
                                  args.only_plaque, args.config, args.bc_learning)

        else:  # parameters of dataloader for 2.5D and 3D is the same
            if args.model_type == '3d':
                from volume.dataloader import read_train_data
            elif args.model_type == '2.5d':
                from hybrid.dataloader import read_train_data

            dataloaders = read_train_data(args.data_dir, metric_prev_epoch, phases_prev_epoch, args.compose, 'train',
                                          is_hard_mining, args.percentile, args.multi_view, args.interval, args.down_sample,
                                          args.batch_size, args.num_workers, True, args.config)

        # during hard mining, if # of training samples is lower than threshold, stop training
        if len(dataloaders['train'].dataset.phases) <= 20:
            break

        dataset_sizes = {'train': 0, 'val': 0, 'test': 0}
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                if args.model == 'deeplab_resnet':
                    adjust_learning_rate(optimizer, scheduler)

                model.train()  # Set model to training mode
                slicewise_metric_epoch = [] # for hard mining

            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_hdist, running_reghdf, running_asd, running_vd = 0.0, 0.0, 0.0, 0.0
            running_corrects, running_f1_score = 0.0, 0.0
            running_fscores = np.zeros(args.output_channel, dtype=np.float32) # class-wise F1 score
            # record # of effective samples for each class for segmentation
            running_effect_samples = np.zeros(args.output_channel, dtype=np.uint32)

            if args.criterion.startswith('whddb') or args.criterion == 'mwhddb': # record inner and outer bound respectively
                running_boundwise_loss = np.zeros(args.output_channel-1)

            dl_pbar = tqdm(dataloaders[phase])
            for sample_inx, sample in enumerate(dl_pbar):
                dl_pbar.update(100)
                inputs, labels = sample

                patch_size = len(inputs)
                dataset_sizes[phase] += patch_size

                # wrap them in Variable
                if args.use_gpu:
                    inputs = Variable(inputs.cuda()).float()
                    labels = Variable(labels.cuda()).long()

                else:
                    inputs = Variable(inputs).float()
                    labels = Variable(labels).long()

                optimizer.zero_grad()
                outputs = model(inputs)
                # snake constraint
                regs = probmap2bound(F.softmax(outputs, 1), n_workers=32, thres=0.7, kernel_size=9)

                if args.model == 'deeplab_resnet':
                    loss = cal_loss(outputs, labels, args.criterion, criterion)
                    outputs = outputs[-1] # max fusion output is saved
                    outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs)

                elif args.model_type == "2.5d" and args.model == "res_unet_reg": # Hybrid res-unet with regularization
                    prob_map, reg = outputs
                    n_gt_pts = torch.sum(labels.view(labels.size(0), -1) != 0, 1).float()
                    criterion_reg = nn.SmoothL1Loss()
                    loss_reg = criterion_reg(reg, n_gt_pts)
                    assert (args.criterion.startswith('whddb') or args.criterion == 'mwhddb'), \
                        "Hybrid Res-UNet should match with WHD loss"
                    loss_whd, loss_boundwise = criterion(F.softmax(prob_map, dim=1), labels)
                    loss = loss_whd + 1.0 * loss_reg
                    outputs = prob_map

                else: # with single output
                    if args.criterion == 'nll' and not args.mpl:
                        loss = criterion(F.log_softmax(outputs, dim=1), labels)
                    elif args.criterion == 'whd':
                        loss = criterion(F.softmax(outputs, dim=1)[:, 1], labels)
                    elif args.criterion == 'mwhd':
                        loss = criterion(F.softmax(outputs, dim=1)[:, 1], labels)
                    # whddb series loss
                    elif args.criterion == 'whddb' or args.criterion == 'mwhddb' or args.criterion == "whddbmax":
                        loss, loss_boundwise = criterion(F.softmax(outputs, dim=1), labels)
                    # snake constrained whddb loss
                    elif args.criterion == "whddbsnake":
                        if epoch <= 10:
                            criterion_base = WeightedHausdorffDistanceDoubleBoundLoss(return_boundwise_loss=True,
                                alpha=args.whd_alpha, beta=args.whd_beta, ratio=args.whd_ratio)
                            loss, loss_boundwise = criterion_base(F.softmax(outputs, dim=1), labels)
                        else:
                            # reg has already been calculated before
                            loss, loss_boundwise = criterion(F.softmax(outputs, dim=1), labels, regs)
                    elif args.criterion == 'whddb_cereg':
                        loss_whd, loss_boundwise = criterion(F.softmax(outputs, dim=1), labels)
                        loss_ce = nn.CrossEntropyLoss(ignore_index=0)(outputs, labels)
                        loss = 0.2 * loss_whd + 0.8 * loss_ce
                    else: # dice, ce, gdl1, gdl2, ceb
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if args.criterion.startswith('whddb') or args.criterion == 'mwhddb':
                    running_boundwise_loss += loss_boundwise.data.cpu().numpy() * patch_size

                # various metrics
                running_loss += loss.data.item() * patch_size
                running_corrects += float(torch.sum(preds == labels.data)) / preds[0].numel()

                # calculate hd95 and asd
                preds_bound_np, labels_bound_np = preds.cpu().numpy(), labels.data.cpu().numpy()
                regs_np = regs.data.cpu().numpy()
                mean_hdf, batch_hdf = volumewise_hd95(preds_bound_np, labels_bound_np, return_slicewise_hdf=True)
                mean_reghdf = volumewise_hd95(regs_np, labels_bound_np, return_slicewise_hdf=False)
                mean_asd = volumewise_asd(regs_np, labels_bound_np, n_classes=3)
                running_hdist += mean_hdf * patch_size
                running_reghdf += mean_reghdf * patch_size
                running_asd += mean_asd * patch_size

                # calculate F1, VD
                preds_np = np.stack([innerouterbound2mask(r, args.output_channel) for r in regs_np])
                labels_np = np.stack([innerouterbound2mask(label, args.output_channel) for label in labels_bound_np])
                cal_f1 = cal_f_score_slicewise if args.model_type == '3d' else cal_f_score
                _, f_scores, n_effect_samples = cal_f1(preds_np, labels_np, n_class=args.output_channel,
                                                       return_slice_f1=False, return_class_f1=True)
                running_fscores += f_scores
                running_effect_samples += n_effect_samples
                mean_vd = volumewise_ravd(preds_np, labels_np)
                running_vd += mean_vd * patch_size

                if phase == 'train':
                    slicewise_metric_epoch += batch_hdf

            dl_pbar.close()
            print()

            if args.criterion.startswith('whddb') or args.criterion == 'mwhddb':
                epoch_loss_boundwise[phase].append(running_boundwise_loss / dataset_sizes[phase])

            epoch_loss[phase].append(running_loss / dataset_sizes[phase])
            epoch_acc[phase].append(float(running_corrects) / dataset_sizes[phase])
            epoch_hdist[phase].append(running_hdist / dataset_sizes[phase])
            epoch_asd[phase].append(running_asd / dataset_sizes[phase])
            epoch_vd[phase].append(running_vd / dataset_sizes[phase])
            epoch_reghdf[phase].append(running_reghdf / dataset_sizes[phase])

            running_f1_class = running_fscores / running_effect_samples
            epoch_f1_score_class[phase].append(running_f1_class)  # f1 score for each class
            epoch_f1_score[phase].append(running_f1_class.mean())

            if args.criterion.startswith('whddb') or args.criterion == 'mwhddb':
                print("[{:5s}({} samples)] Loss: {:.4f} Loss_boundwise: {} Acc: {:.4f} Ave_F1: {:.4f} class-wise F1: {} "
                      "Ave_hdf: {:.4f} Ave_reghdf: {:.4f} Ave_ASD: {:.4f} Ave_VD: {:.4f}".format(phase, len(dataloaders[phase].dataset.phases),
                      epoch_loss[phase][-1], epoch_loss_boundwise[phase][-1], epoch_acc[phase][-1], epoch_f1_score[phase][-1],
                      running_f1_class, epoch_hdist[phase][-1], epoch_reghdf[phase][-1], epoch_asd[phase][-1], epoch_vd[phase][-1]))
            else:
                print("[{:5s}({} samples)] Loss: {:.4f} Acc: {:.4f} Ave_F1: {:.4f} class-wise F1: {} Ave_hdf: {:.4f} "
                      "Ave_reghdf: {:.4f} Ave_ASD: {:.4f} Ave_VD: {:.4f}".format(phase, len(dataloaders[phase].dataset.phases),
                epoch_loss[phase][-1], epoch_acc[phase][-1], epoch_f1_score[phase][-1], running_f1_class,
                epoch_hdist[phase][-1], epoch_reghdf[phase][-1], epoch_asd[phase][-1], epoch_vd[phase][-1]))

            # update metric_prev_epoch and phases_prev_epoch
            if phase == 'train':
                metric_prev_epoch = np.array(slicewise_metric_epoch)
                phases_prev_epoch = dataloaders['train'].dataset.phases

            # save the learnt best model evaluated on validation data
            if phase == 'val':
                val_loss_bf = sum(epoch_loss['val'][-5:]) / len(epoch_loss['val'][-5:])
                if val_loss_bf <= best_loss:
                    best_loss = val_loss_bf

                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, args.model_save_name)

                if val_loss_bf > epoch_loss_prev:
                    loss_keep += 1
                else:
                    loss_keep = 0

                epoch_loss_prev = val_loss_bf

        # plot temporal loss, acc, f1_score for train, val and test respectively.
        if (epoch+1) % 5 == 0 and phase == 'test':
            metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_vd, epoch_hdist, epoch_reghdf]
            labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'vd', 'hd95_pred', 'hd95_reg']
            plot_metrics(metrics, labels, fig_dir=args.fig_dir)
            plot_class_f1(epoch_f1_score_class, args.fig_dir)

            ## for plot innerbound and outerbound loss respectively
            # if args.criterion.startswith('whddb') or args.criterion == 'mwhddb':
            #     metrics= [{k:v[i] for k, v in epoch_loss_boundwise.items()} for i in range(args.output_channel-1)]
            #     labels = ['innerbound_loss', 'outerbound_loss']
            #     plot_metrics(metrics, labels, fig_dir=args.fig_dir)

        if loss_keep == 10:
            break

    metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_vd, epoch_hdist, epoch_reghdf]
    labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'vd', 'hd95_pred', 'hd95_reg']
    plot_metrics(metrics, labels, fig_dir=args.fig_dir)
    plot_class_f1(epoch_f1_score_class, args.fig_dir)

    # if args.criterion.startswith('whddb') or args.criterion == 'mwhddb':
    #     metrics = [{k: v[i] for k, v in epoch_loss_boundwise.items()} for i in range(args.output_channel - 1)]
    #     labels = ['innerbound_loss', 'outerbound_loss']
    #     plot_metrics(metrics, labels, fig_dir=args.fig_dir)

    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_save_name)


def model_reference(args, sample_stack_rows=50):
    """ model reference and plot the segmentation results
        for model reference, several epochs are used to balance the risks and other metrics
        for segmentation results plotting, only one epoch is used without data augmentation
    Args:
        args: parser arguments
        sample_stack_rows: int, how many slices to plot per image
    """

    #############################################################################################
    # Part 1: model reference and metric evaluations
    #############################################################################################
    model = torch.load(args.model_save_name, map_location=lambda storage, loc: storage)
    if args.use_gpu:
        model = model.cuda()

    dataset_sizes = 0
    running_hdist, running_reghdf, running_asd, running_vd = 0.0, 0.0, 0.0, 0.0
    running_corrects, running_f1, running_dice_score = 0.0, 0.0, 0.0
    running_fscores = np.zeros(args.output_channel, dtype=np.float32)
    running_effect_samples = np.zeros(args.output_channel, dtype=np.uint32)

    if args.model_type == '2d':
        from image.dataloader import read_train_data
        dataloaders = read_train_data(args.data_dir, args.compose, 'test', None, None, True,
                                      False, args.num_workers, args.batch_size, args.percentile,
                                      args.multi_view, args.only_plaque, args.config)
    else:  # parameters of dataloader for 2.5D and 3D is the same
        if args.model_type == '3d':
            from volume.dataloader import read_train_data
        elif args.model_type == '2.5d':
            from hybrid.dataloader import read_train_data

        dataloaders = read_train_data(args.data_dir, None, None, args.compose, 'test',
                                      False, args.percentile, args.multi_view, args.interval, args.down_sample,
                                      args.batch_size, args.num_workers, True, args.config)

    for samp_inx, sample in enumerate(dataloaders['test']):
        inputs, labels  = sample
        patch_size = len(inputs)
        dataset_sizes += patch_size

        # wrap them in Variable
        if args.use_gpu:
            inputs = Variable(inputs.cuda()).float()
            labels = Variable(labels.cuda()).long()
        else:
            inputs = Variable(inputs).float()
            labels = Variable(labels).long()

        outputs = model(inputs) # outputs can be tensor, tuple or list based on model we choose
        regs = probmap2bound(F.softmax(outputs, 1), n_workers=32, thres=0.7, kernel_size=9)

        if args.model == 'deeplab_resnet':
            outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs[-1])
        elif args.model_type == "2.5d" and args.model == "res_unet_reg": # multiple outputs
            outputs = outputs[0]

        _, preds = torch.max(outputs.data, 1)

        running_corrects += float(torch.sum(preds == labels.data)) / preds[0].numel()

        # calculate hd95 and asd
        preds_bound_np, labels_bound_np = preds.cpu().numpy(), labels.data.cpu().numpy()
        regs_np = regs.data.cpu().numpy()
        mean_reghdf = volumewise_hd95(regs, labels, return_slicewise_hdf=False)
        mean_hdf, batch_hdf = volumewise_hd95(preds_bound_np, labels_bound_np, return_slicewise_hdf=True)
        mean_asd = volumewise_asd(preds_bound_np, labels_bound_np, n_classes=3)
        running_hdist += mean_hdf * patch_size
        running_asd += mean_asd * patch_size
        running_reghdf += mean_reghdf * patch_size

        # calculate F1, VD
        preds_np = np.stack([innerouterbound2mask(r, args.output_channel) for r in regs_np])
        labels_np = np.stack([innerouterbound2mask(label, args.output_channel) for label in labels_bound_np])
        cal_f1 = cal_f_score_slicewise if args.model_type == '3d' else cal_f_score
        _, f_scores, n_effect_samples = cal_f1(preds_np, labels_np, n_class=args.output_channel,
                                               return_slice_f1=False, return_class_f1=True)
        running_fscores += f_scores
        running_effect_samples += n_effect_samples
        mean_vd = volumewise_ravd(preds_np, labels_np)
        running_vd += mean_vd * patch_size

    epoch_acc = float(running_corrects) / dataset_sizes
    epoch_class_f1 = running_fscores / running_effect_samples
    epoch_f1 = epoch_class_f1.mean()
    epoch_hdist = running_hdist / dataset_sizes
    epoch_asd = running_asd / dataset_sizes
    epoch_vd = running_vd / dataset_sizes
    epoch_reghdf = running_reghdf / dataset_sizes

    # print various metrics
    print("Acc: {:.4f} Ave_F1: {:.4f} Ave_hdf: {:.4f}, Ave_reghdf: {:.4f}. Ave_ASD: {:.4f} Ave_VD: {:.4f}".format(
        epoch_acc, epoch_f1, epoch_hdist, epoch_reghdf, epoch_asd, epoch_vd))

    for c_inx, each_f1 in enumerate(epoch_class_f1):
        print("Class-{}: F1-{:.4f}".format(c_inx, each_f1))

    ##########################################################################################
    # plot the prediction results
    ##########################################################################################
    if args.do_plot:
        plot_data = args.plot_data
        args.compose[plot_data] = args.compose['test']

        if args.model_type == '2d':
            from image.dataloader import read_plot_data
            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, False, args.num_workers,
                                         args.batch_size, args.multi_view, args.config)
        else:  # parameters of dataloader for 2.5D and 3D is the same
            if args.model_type == '3d':
                from volume.dataloader import read_plot_data
            elif args.model_type == '2.5d':
                from hybrid.dataloader import read_plot_data

            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, args.multi_view, args.interval,
                                         args.down_sample, args.num_workers, False, args.config)

        for samp_inx, sample in enumerate(dataloaders[plot_data]):
            inputs_batch, labels, sample_name, start = sample
            sample_name, start = sample_name[0], start.item()
            inputs_batch = torch.squeeze(inputs_batch, dim=0)  # [N, 1, T, H, W]
            labels = torch.squeeze(labels, dim=0)  # [N, T, H, W]
            patch_size = len(inputs_batch)

            # process each mini-batch
            for mb_inx in range(0, patch_size, args.batch_size):
                end = min(mb_inx + args.batch_size, patch_size)
                inputs =inputs_batch[mb_inx:end]

                # wrap them in Variable
                if args.use_gpu:
                    inputs = Variable(inputs.cuda()).float()
                else:
                    inputs = Variable(inputs).float()

                outputs = model(inputs) # both outputs and preds are tensors

                if args.model == 'deeplab_resnet':
                    outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs[-1])
                elif args.model_type == "2.5d" and args.model == "res_unet_reg":
                    outputs = outputs[0]

                outputs_mb_np = F.softmax(outputs, dim=1).data.cpu().numpy() # don't forget the softmax here
                # outputs_mb_np = outputs_mb_np[:, 1]  # only choose channel 1

                _, preds = torch.max(outputs.data, 1)
                preds_mb_np = preds.cpu().numpy()

                if mb_inx == 0:
                    preds_np = np.zeros((patch_size, *(preds_mb_np[0].shape)), dtype=preds_mb_np.dtype)
                    # outputs_np shape: [N * C * T * H * W] or [N * C * H * W]
                    outputs_np = np.zeros((patch_size, *(outputs_mb_np[0].shape)), dtype=outputs_mb_np.dtype)

                preds_np[mb_inx:end], outputs_np[mb_inx:end] = preds_mb_np, outputs_mb_np

            # convert into numpy
            labels_np = labels.cpu().numpy()
            if inputs_batch.size(1) == 1:  # only one channel
                inputs_np = torch.squeeze(inputs_batch, dim=1).cpu().numpy() # [N, T, H, W]
            else: # if 3 channels, only select the first channel
                inputs_np = inputs_batch[:, 0].cpu().numpy()

            if args.model_type == '2.5d':
                n_slices = inputs_np.shape[1]
                inputs_np = inputs_np[:, n_slices//2]

            # for 2D images, we can directly use it for plot, for 3D volume, transform is necessary
            if args.model_type == '3d':
                inputs_np, labels_np, preds_np, outputs_np = rearrange_volume(
                    inputs_np, labels_np, preds_np, outputs_np, args)

            if args.model_type == '2.5d': # shift start index if 2.5D model
                start += (args.interval // 2) * args.down_sample
            # save predictions into pickle and plot the results
            plot_save_result(labels_np, inputs_np, preds_np, outputs_np, start, sample_name, args.fig_dir,
                               sample_stack_rows, args.output_channel)

def rearrange_volume(inputs, labels, preds, outputs, args):
    """ rearrange volumes into the correct order
    :param inputs: list of ndarrays (N, D, H, W)
    :param labels: ndarray (N, D, H, W)
    :param preds: ndarray (N, D, H, W)
    :param outputs: ndarray (N, C, D, H, W)
    :return:
    """
    inputs = np.reshape(inputs, (-1, *(inputs.shape[2:])))
    labels = np.reshape(labels, (-1, *(labels.shape[2:])))
    preds = np.reshape(preds, (-1, *(preds.shape[2:])))
    outputs = outputs.transpose(0, 2, 1, 3, 4) # [N, C, T, H, W] -- > [N, T, C, H, W]
    outputs = np.reshape(outputs, (-1, *(outputs.shape[2:])))

    num_slices = len(inputs)
    indexes = []
    args.stride = args.down_sample * args.interval
    for s_inx in range(0, num_slices, args.stride):
        for i in range(args.interval):
            for j in range(args.down_sample):
                inx = s_inx + i + j * args.interval
                if inx < num_slices:
                    indexes.append(inx)

    inputs, labels, preds, outputs = inputs[indexes], labels[indexes], preds[indexes], outputs[indexes]

    return (inputs, labels, preds, outputs)


def plot_save_result(labels, inputs, preds, outputs, start, samp_art_name, root_fig_dir, sample_stack_rows, n_class):
    """ plot segmentation results and save the risks """

    fig_dir = root_fig_dir + '/' + samp_art_name
    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)

    data = {'input': inputs, 'label': labels, 'pred': preds, 'output': outputs,
            'sample_name': samp_art_name, 'start': start, 'n_class': n_class}

    # resave slices into gif animation
    data_types = {"input", "label", "pred"}
    for data_type in data_types:
        arrays = list(data[data_type])
        imageio.mimsave('{}/{}.gif'.format(fig_dir, data_type), arrays)

    with open(osp.join(fig_dir, 'data.pkl'), 'wb') as writer:
        pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)

    # plot the inputs, ground truth, outputs and F1 scores with sample_stack2
    for inx in range(0, len(inputs), sample_stack_rows):
        over = min(inx + sample_stack_rows, len(inputs))
        label_plot, input_plot, pred_plot, output_plot = labels[inx:over], inputs[inx:over], \
                                        preds[inx:over], outputs[inx:over]


        data_list = [{"input": input, "GT": label, "pred": pred, "output": output}
                     for (input, label, pred, output) in zip(input_plot, label_plot, pred_plot, output_plot)]

        file_name = "{}/{:03d}".format(fig_dir, inx + start)
        sample_list_hdf(data_list, rows=over - inx, start_with=0, show_every=1, fig_name=file_name,
                             start_inx=inx + start, n_class=n_class)