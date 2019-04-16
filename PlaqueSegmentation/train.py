# _*_ coding: utf-8 _*_

""" define train and test functions here """

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from sklearn.metrics import auc
import copy
from collections import Counter
import numpy as np
np.set_printoptions(precision=4)

from tqdm import tqdm
import torch
from torch.autograd import Variable

import torch.nn.functional as F
from torch import nn
import pickle

import os.path as osp
import os
from image.models.deeplab_resnet import outS
from metric import cal_f_score, cal_f_score_slicewise, volumewise_hd95, volumewise_asd
from loss import WeightedKLDivLoss
from utils import mask2innerouterbound

from vision import plot_metrics, plaque_detection_rate, plot_class_f1
from vision import plot_slice_wise_measures, sample_seg_with_hfd
from image.models.deeplab_resnet import adjust_learning_rate, cal_loss
from medpy.metric.binary import ravd

def train_model(model, criterion, optimizer, scheduler, args):
    """ train the model
        for obtain stable validation result, we use averaged val loss over several epochs
        as the principle of choosing best model weights
    Args:
        model: model inheriting from nn.Module class
        criterion: criterion class, loss function used
        optimizer: optimizer, optimization strategy
        scheduler: lr scheduler
        args: parser arguments
    """

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 1.0e9
    loss_keep = 0 # check how many times the val loss has decreased continuously
    epoch_loss_prev = 1.0e9 # loss at the previous epoch

    epoch_acc = {'train': [], 'val': [], 'test': []}
    epoch_f1_score = {'train': [], 'val': [], 'test': []}
    epoch_f1_score_class = {'train': [], 'val': [], 'test': []}
    epoch_loss = {'train': [], 'val': [], 'test': []}
    epoch_hdist = {'train': [], 'val': [], 'test': []}
    epoch_asd = {'train': [], 'val': [], 'test': []}
    epoch_vd = {'train': [], 'val': [], 'test': []}

    metric_prev_epoch = None
    phases_prev_epoch = None
    for epoch in range(args.num_train_epochs):
        print("{}/{}".format(epoch+1, args.num_train_epochs))
        if epoch != 0 and epoch % args.n_epoch_hardmining == 0:
            is_hard_mining = True
        else:
            is_hard_mining = False

        if args.model_type == '2d':
            from image.dataloader import read_train_data
            if args.onlyrisk:
                dataloaders = read_train_data(args.data_dir, args.compose, 'train', None, None, True,
                                    is_hard_mining, args.num_workers, args.batch_size, args.percentile, args.multi_view,
                                    args.onlyrisk, args.config, args.bc_learning)
            else:
                dataloaders = read_train_data(args.data_dir, args.compose, 'train', metric_prev_epoch, phases_prev_epoch, True,
                                  is_hard_mining, args.num_workers, args.batch_size, args.percentile, args.multi_view,
                                  args.onlyrisk, args.config, args.bc_learning)

        elif args.model_type == '3d':
            from volume.dataloader import read_train_data
            dataloaders = read_train_data(args.data_dir, metric_prev_epoch, phases_prev_epoch, args.compose, 'train',
                                          is_hard_mining, args.percentile, args.multi_view, args.interval, args.down_sample,
                                          args.batch_size, args.num_workers, True, args.config)

        if len(dataloaders['train'].dataset.phases) <= 20:
            break

        dataset_sizes = {'train': 0, 'val': 0, 'test': 0}
        for phase in ['train', 'val', 'test']:
            # print("processing {}".format(phase))
            if phase == 'train':
                scheduler.step()
                if args.model == 'deeplab_resnet':
                    adjust_learning_rate(optimizer, scheduler)

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            if phase == 'train':
                f1_slices_epoch = []

            running_loss = 0.0
            running_hdist, running_asd, running_vd = 0.0, 0.0, 0.0
            running_corrects, running_f1_score =  0.0, 0.0
            running_fscores = np.zeros(args.output_channel, dtype=np.float32)
            running_effect_samples = np.zeros(args.output_channel, dtype=np.uint32)

            running_cal_pgt, running_cal_pp, running_cal_tp = 0, 0, 0
            running_noncal_pgt, running_noncal_pp, running_noncal_tp = 0, 0, 0

            dl_pbar = tqdm(dataloaders[phase])
            for sample_inx, sample in enumerate(dl_pbar):
                dl_pbar.update(100)
                inputs, labels = sample
                patch_size = len(inputs) if args.model != 'hyper_tiramisu' else len(inputs[0])
                dataset_sizes[phase] += patch_size

                # wrap them in Variable
                if args.use_gpu:
                    if args.model == 'hyper_tiramisu':
                        inputs = [Variable(input.cuda()).float() for input in inputs]
                    else:
                        inputs = Variable(inputs.cuda()).float()

                    if phase == 'train' and args.bc_learning is not None:
                        labels = Variable(labels.cuda()).float()
                    else:
                        labels = Variable(labels.cuda()).long()

                else:
                    if args.model == 'hyper_tiramisu':
                        inputs = [Variable(input).float() for input in inputs]
                    else:
                        inputs = Variable(inputs).float()

                    if phase == 'train' and args.bc_learning is not None:
                        labels = Variable(labels).float()
                    else:
                        labels = Variable(labels).long()

                optimizer.zero_grad()
                outputs = model(inputs)

                if phase == 'train' and args.bc_learning is not None: # for bc learning
                    if args.model == 'deeplab_resnet':
                        loss = cal_loss(outputs, labels, args.criterion, args.criterion_bc)
                        outputs = outputs[-1] # max fusion output is saved
                        outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs)
                    else:
                        loss = args.criterion_bc(outputs, labels)
                else:
                    if args.model == 'deeplab_resnet':
                        loss = cal_loss(outputs, labels, args.criterion, criterion)
                        outputs = outputs[-1] # max fusion output is saved
                        outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs)
                    else:
                        if args.criterion == 'nll' and not args.mpl:
                            loss = criterion(F.log_softmax(outputs, dim=1), labels)
                        else: # dice, ce, gdl1, gdl2
                            loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.data, 1)


                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if args.output_channel >= 5 and not args.bound_out:
                    # calculate calcified and non-calcified plaque detection rate
                    cal_pgt, cal_pp, cal_tp, noncal_pgt, noncal_pp, noncal_tp = plaque_detection_rate(labels, preds,
                                                                                  args.output_channel)
                    # accumulate gt, positive and true positive from each minibatch
                    running_cal_pgt += cal_pgt
                    running_cal_pp += cal_pp
                    running_cal_tp += cal_tp
                    running_noncal_pgt += noncal_pgt
                    running_noncal_pp += noncal_pp
                    running_noncal_tp += noncal_tp

                if phase == 'train' and args.bc_learning is not None:
                    _, labels = torch.max(labels, 1)

                running_loss += loss.data.item() * patch_size
                running_corrects += float(torch.sum(preds == labels.data)) / preds[0].numel()

                # calculate hd95 and asd
                preds_np, labels_np = preds.cpu().numpy(), labels.data.cpu().numpy()
                preds_bound_np = np.stack([mask2innerouterbound(pred, args.width) for pred in preds_np])
                labels_bound_np = np.stack([mask2innerouterbound(label, args.width) for label in labels_np])
                mean_hdf = volumewise_hd95(preds_bound_np , labels_bound_np, return_slicewise_hdf=False)
                mean_asd = volumewise_asd(preds_bound_np, labels_bound_np, n_classes=3)
                running_hdist += mean_hdf * patch_size
                running_asd += mean_asd * patch_size

                # calculate F1, VD
                cal_f1 = cal_f_score if args.model_type == '2d' else cal_f_score_slicewise
                _, f_scores, n_effect_samples, f1_slices_batch = cal_f1(preds_np, labels_np, n_class=args.output_channel,
                                                    return_slice_f1=True, return_class_f1=True)
                running_fscores += f_scores
                running_effect_samples += n_effect_samples
                mean_vd = abs(ravd(preds_np, labels_np))
                running_vd += mean_vd * patch_size

                if phase == 'train':
                    f1_slices_epoch += f1_slices_batch

            dl_pbar.close()
            print()

            epoch_loss[phase].append(running_loss / dataset_sizes[phase])
            epoch_acc[phase].append(float(running_corrects) / dataset_sizes[phase])
            epoch_hdist[phase].append(running_hdist / dataset_sizes[phase])
            epoch_asd[phase].append(running_asd / dataset_sizes[phase])
            epoch_vd[phase].append(running_vd / dataset_sizes[phase])

            running_f1_class = running_fscores / running_effect_samples
            epoch_f1_score_class[phase].append(running_f1_class) # f1 score for each class
            epoch_f1_score[phase].append(running_f1_class.mean())

            print("[{:5s}({} samples)] Loss: {:.4f} Acc: {:.4f} Ave_F1: {:.4f} class-wise F1: {} Ave_hdf: {:.4f} "
                  "Ave_ASD: {:.4f} Ave_VD: {:.4f}".format(phase, len(dataloaders[phase].dataset.phases),
                epoch_loss[phase][-1], epoch_acc[phase][-1], epoch_f1_score[phase][-1], running_f1_class,
                epoch_hdist[phase][-1], epoch_asd[phase][-1], epoch_vd[phase][-1]))

            # update metric_prev_epoch and phases_prev_epoch
            if phase == 'train':
                metric_prev_epoch = np.array(f1_slices_epoch)
                phases_prev_epoch = dataloaders['train'].dataset.phases

            # deep copy the model
            if phase == 'val':
                val_loss_bf = sum(epoch_loss['val'][-5:]) / len(epoch_loss['val'][-5:])
                if val_loss_bf <= best_loss:
                    best_loss = val_loss_bf
                    best_epoch = epoch
                    # be careful when assign one tensor to another
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, args.model_save_name)

                if val_loss_bf > epoch_loss_prev:
                    loss_keep += 1
                else:
                    loss_keep = 0

                epoch_loss_prev = val_loss_bf

            if args.output_channel >= 5 and not args.bound_out:
                # calculate cal and non-cal detection rate for test data
                epoch_cal_pr = float(running_cal_tp) / running_cal_pp if running_cal_pp != 0 else 0.0
                epoch_cal_rc = float(running_cal_tp) / running_cal_pgt
                epoch_cal_f1 = 2.0 * running_cal_tp / (running_cal_pgt + running_cal_pp)
                epoch_noncal_pr = float(running_noncal_tp) / running_noncal_pp if running_noncal_pp != 0 else 0.0
                epoch_noncal_rc = float(running_noncal_tp) / running_noncal_pgt
                epoch_noncal_f1 = 2.0 * running_noncal_tp / (running_noncal_pgt + running_noncal_pp)

                print('Cal: PR - {:.4f} RC - {:.4f} F1 - {:.4f} Noncal: PR - {:.4f} RC - {:.4f} F1 - {:.4f}'.format(
                    epoch_cal_pr, epoch_cal_rc, epoch_cal_f1, epoch_noncal_pr, epoch_noncal_rc, epoch_noncal_f1))

        # plot temporal loss, acc, f1_score after test
        if (epoch+1) % 5 == 0 and phase == 'test':
            metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_vd, epoch_hdist]
            labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'vd', 'hd95']
            plot_metrics(metrics, labels, fig_dir=args.fig_dir)
            plot_class_f1(epoch_f1_score_class, args.fig_dir)

        if loss_keep == 10:
            break
    # plot loss, acc, f1, asd, vd, hd95
    metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_vd, epoch_hdist]
    labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'vd', 'hd95']
    plot_metrics(metrics, labels, fig_dir=args.fig_dir)
    plot_class_f1(epoch_f1_score_class, args.fig_dir)

    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_save_name)

def model_reference(args, sample_stack_rows=50):
    """ model reference and plot the segmentation results
        for model reference, several epochs are used to balance the risks and other metrics
        for segmentation results plotting, only one epoch is used without data augmentation
    Args:
        model: model
        dataloaders: DataLoader class, dataloader used to read test data
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
    running_hdist, running_asd, running_vd = 0.0, 0.0, 0.0
    running_corrects, running_f1, running_dice_score = 0.0, 0.0, 0.0

    # for class-wise F1 scores
    running_num_samples_class = np.zeros(args.output_channel  , dtype=np.uint32)
    running_class_f1 = np.zeros(args.output_channel, dtype=np.float32)

    if args.model_type == '2d':
        from image.dataloader import read_train_data
        dataloaders = read_train_data(args.data_dir, args.compose, 'test', None, None, True,
                                      False, args.num_workers, args.batch_size, args.percentile,
                                      args.multi_view, args.onlyrisk, args.config)
    elif args.model_type == '3d':
        from volume.dataloader import read_train_data
        dataloaders = read_train_data(args.data_dir, None, None, args.compose, 'test',
                                      False, args.percentile, args.multi_view, args.interval, args.down_sample,
                                      args.batch_size, args.num_workers, True, args.config)

    for samp_inx, sample in enumerate(dataloaders['test']):
        inputs, labels  = sample
        patch_size = len(inputs) if args.model != 'hyper_tiramisu' else len(inputs[0])
        dataset_sizes += patch_size

        # wrap them in Variable
        if args.use_gpu:
            if args.model == 'hyper_tiramisu':
                inputs = [Variable(input.cuda()).float() for input in inputs]
            else:
                inputs = Variable(inputs.cuda()).float()
            labels = Variable(labels.cuda()).long()
        else:
            if args.model == 'hyper_tiramisu':
                inputs = [Variable(input).float() for input in inputs]
            else:
                inputs = Variable(inputs).float()
            labels = Variable(labels).long()

        outputs = model(inputs)

        if args.model == 'deeplab_resnet':
            outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs[-1])

        _, preds = torch.max(outputs.data, 1)

        # calculate seg correct and risk correct within each minibatch
        running_corrects += float(torch.sum(preds == labels.data)) / preds[0].numel()

        # calculate hd95 and asd
        preds_np, labels_np = preds.cpu().numpy(), labels.data.cpu().numpy()
        preds_bound_np = np.stack([mask2innerouterbound(pred, args.width) for pred in preds_np])
        labels_bound_np = np.stack([mask2innerouterbound(label, args.width) for label in labels_np])
        mean_hdf = volumewise_hd95(preds_bound_np, labels_bound_np, return_slicewise_hdf=False)
        mean_asd = volumewise_asd(preds_bound_np, labels_bound_np, n_classes=3)
        running_hdist += mean_hdf * patch_size
        running_asd += mean_asd * patch_size

        # calculate F1 score
        cal_f1 = cal_f_score if args.model_type == '2d' else cal_f_score_slicewise
        batch_f1, batch_class_f1, n_effect_samples = cal_f1(preds_np, labels_np, n_class=args.output_channel,
                                                            return_class_f1=True)
        running_class_f1 += batch_class_f1
        running_num_samples_class += n_effect_samples

        mean_vd = abs(ravd(preds_np, labels_np))
        running_vd += mean_vd * patch_size

        if args.output_channel >= 5 and not args.bound_out:
            labels_np = labels.data.cpu().numpy()
            preds_np = preds.cpu().numpy()

            if samp_inx == 0:
                labels_all = labels_np
                preds_all = preds_np
            else:
                labels_all = np.concatenate([labels_all, labels_np], axis=0) # [N, H, W]
                preds_all = np.concatenate([preds_all, preds_np], axis=0)

    # plot slice-wise measurements under different thresholds
    if args.output_channel >= 5 and not args.bound_out:
        plot_slice_wise_measures(labels_all, preds_all, args)

    epoch_acc = float(running_corrects) / dataset_sizes
    epoch_class_f1 = running_class_f1 / running_num_samples_class
    epoch_f1 = epoch_class_f1.mean()
    epoch_hdist = running_hdist / dataset_sizes
    epoch_asd = running_asd / dataset_sizes
    epoch_vd = running_vd / dataset_sizes

    # print various metrics
    print("Acc: {:.4f} Ave_F1: {:.4f} Ave_hdf: {:.4f}, Ave_ASD: {:.4f} Ave_VD: {:.4f}".format(
        epoch_acc, epoch_f1, epoch_hdist, epoch_asd, epoch_vd))

    for c_inx, each_f1 in enumerate(epoch_class_f1):
        print("Class-{}: F1-{:.4f}".format(c_inx, each_f1))

    if args.do_plot:
        ############################################################################################
        # Part 2: plot segmentation results (这部分真蛋疼，已经写了无数遍了)
        ############################################################################################
        plot_data = args.plot_data
        args.compose[plot_data] = args.compose['test']

        if args.model_type == '2d':
            from image.dataloader import read_plot_data
            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, False, args.num_workers,
                                         args.batch_size, args.multi_view, args.config)
        elif args.model_type == '3d':
            from volume.dataloader import read_plot_data
            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, args.multi_view, args.interval,
                                         args.down_sample, args.num_workers, False, args.config)

        for samp_inx, sample in enumerate(dataloaders[plot_data]):
            inputs_batch, labels, sample_name, start = sample

            # convert inputs into list of tensor no matter whether 'hyper_tiramisu' model or not
            if args.model != 'hyper_tiramisu':
                inputs_batch = [inputs_batch]

            sample_name, start = sample_name[0], start.item()
            inputs_batch = [torch.squeeze(input, dim=0) for input in inputs_batch]  # [N, 1, T, H, W]
            labels = torch.squeeze(labels, dim=0)  # [N, T, H, W]
            patch_size = len(inputs_batch[0])

            for mb_inx in range(0, patch_size, args.batch_size):
                end = min(mb_inx + args.batch_size, patch_size)
                inputs = [input[mb_inx:end] for input in inputs_batch]

                # wrap them in Variable
                if args.use_gpu:
                    inputs = [Variable(input.cuda()).float() for input in inputs]
                else:
                    inputs = [Variable(input).float() for input in inputs]

                if args.model != 'hyper_tiramisu':
                    inputs = inputs[0]

                outputs = model(inputs) # both outputs and preds are tensors
                if args.model == 'deeplab_resnet':
                    outputs = nn.Upsample(size=(inputs.size(2), inputs.size(3)), mode='bilinear')(outputs[-1])

                outputs_mb_np = outputs.data.cpu().numpy()
                _, preds = torch.max(outputs.data, 1)

                preds_mb_np = preds.cpu().numpy()

                if mb_inx == 0:
                    preds_np = np.zeros((patch_size, *(preds_mb_np[0].shape)), dtype=preds_mb_np.dtype)
                    outputs_np = np.zeros((patch_size, *(outputs_mb_np[0].shape)), dtype=outputs_mb_np.dtype)

                preds_np[mb_inx:end], outputs_np[mb_inx:end] = preds_mb_np, outputs_mb_np

            # convert into numpy
            labels_np = labels.cpu().numpy()
            if inputs_batch[0].size(1) == 1:
                inputs_np = [torch.squeeze(input, dim=1).cpu().numpy() for input in inputs_batch]
            else:
                inputs_np = [input[:, 0].cpu().numpy() for input in inputs_batch]

            # for 2D images, we can directly use it for plot, for 3D volume, transform is necessary
            if args.model_type == '3d':
                inputs_np, labels_np, preds_np = rearrange_volume(inputs_np, labels_np, preds_np, args)

            plot_seg_save_risk(labels_np, inputs_np, preds_np, start, sample_name, args.fig_dir,
                               sample_stack_rows, args.output_channel, args.width)

def rearrange_volume(inputs, labels, preds, args):
    """ rearrange volumes into the correct order
    :param inputs: list of ndarrays (N, D, H, W)
    :param labels: ndarray (N, D, H, W)
    :param preds: ndarray (N, D, H, W)
    :return:
    """
    inputs = [np.reshape(input, (-1, *(input.shape[2:]))) for input in inputs]
    labels = np.reshape(labels, (-1, *(labels.shape[2:])))
    preds = np.reshape(preds, (-1, *(preds.shape[2:])))

    num_slices = len(inputs[0])
    indexes = []
    args.stride = args.down_sample * args.interval
    for s_inx in range(0, num_slices, args.stride):
        for i in range(args.interval):
            for j in range(args.down_sample):
                inx = s_inx + i + j * args.interval
                if inx < num_slices:
                    indexes.append(inx)

    inputs = [input[indexes] for input in inputs]
    labels = labels[indexes]
    preds = preds[indexes]

    return (inputs, labels, preds)

# this part varies for different task (segmentation or bound detection)
def plot_seg_save_risk(labels, inputs, preds, start, samp_art_name, root_fig_dir, sample_stack_rows, n_class,
                       width):
    """ plot segmentation results """

    fig_dir = root_fig_dir + '/' + samp_art_name
    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)

    data = {'input': inputs, 'label': labels, 'pred': preds,
            'sample_name': samp_art_name, 'start': start, 'n_class': n_class, 'width': width}

    with open(osp.join(fig_dir, 'data.pkl'), 'wb') as writer:
        pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)

    print("# of slices: {}".format(len(inputs[0])))  # number of input slices

    # plot the inputs, ground truth, outputs and F1 scores with sample_stack2
    for inx in range(0, len(inputs[0]), sample_stack_rows):
        # print("# of slices: {}".format(len(inputs[0])))  # number of input slices
        over = min(inx + sample_stack_rows, len(inputs[0]))
        label_plot, input_plot, pred_plot = labels[inx:over], [inputs[i][inx:over] for i in range(len(inputs))], \
                                            preds[inx:over]

        # print("inputplot size: {}".format([stream.shape for stream in input_plot]))
        input_plot = [input for input in zip(*[input_plot[i] for i in range(len(input_plot))])]
        data_list = [{"input": input[0], "GT": label, "pred": pred}
                     for (input, label, pred) in zip(input_plot, label_plot, pred_plot)]

        file_name = "{}/{:03d}".format(fig_dir, inx + start)
        sample_seg_with_hfd(data_list, rows=over - inx, start_with=0, show_every=1, fig_name=file_name,
                            start_inx=inx + start, n_class=n_class, width=width)