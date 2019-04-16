# _*_ coding: utf-8 _*_

""" metrics for accuracy evaluation """

from sklearn.preprocessing import label_binarize
import torch
import numpy as np
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore', module='sklearn')
import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from torch import nn

from medpy.metric.binary import hd95, asd
from medpy.metric.binary import ravd

torch.set_default_dtype(torch.float32)


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


def volumewise_ahd(preds, targets, return_slicewise_hdf=False, n_classes=3):
    """ calculate volume-wise Averaged Hausdorff Distance between preds and targets
    :param preds: Array/Tensor with size [B, D, H, W] or [B, H, W]
    :param targets: Array/Tensor with size [B, D, H, W] or [B, H, W]
    :param return_slicewise_hdf: bool, whether return slice-wise hdf or not
    :para n_classes, int, how many classes for calculating HDF
    """

    batch_res = []
    if not isinstance(preds, np.ndarray): # set1 and set2 are tensors
        preds = preds.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

    assert len(preds) == len(targets), \
        "length of preds should be equal to that of targets, but got {} and {}".format(len(preds), len(targets))

    if preds.ndim == 4 and targets.ndim == 4:  # convert 3D to 2D if preds and targets are volumes
        preds = np.reshape(preds, (preds.shape[0]*preds.shape[1], *preds.shape[2:]))
        targets = np.reshape(targets, (targets.shape[0]*targets.shape[1], *targets.shape[2:]))

    for pred, target in zip(preds, targets):
        if np.sum(target) != 0:
            slice_ahd = slicewise_ahd(pred, target, n_classes)
            batch_res.append(slice_ahd)

    mean_ahd = sum(batch_res) / len(batch_res)

    if return_slicewise_hdf:
        return mean_ahd, batch_res
    else:
        return mean_ahd

def volumewise_hd95(preds, targets, return_slicewise_hdf=False, n_classes=3):
    """ calculate volume-wise 95 percentile Hausdorff distance
    :param preds: Array/Tensor with size [B, D, H, W] or [B, H, W]
    :param targets: Array/Tensor with size [B, D, H, W] or [B, H, W]
    :param return_slicewise_hdf: bool, whether return slice-wise hdf or not
    :para n_classes, int, how many classes for calculating HDF
    """

    batch_res = []
    if not isinstance(preds, np.ndarray): # set1 and set2 are tensors
        preds = preds.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

    assert len(preds) == len(targets), \
        "length of preds should be equal to that of targets, but got {} and {}".format(len(preds), len(targets))

    if preds.ndim == 4 and targets.ndim == 4:  # convert 3D to 2D if preds and targets are volumes
        preds = np.reshape(preds, (preds.shape[0]*preds.shape[1], *preds.shape[2:]))
        targets = np.reshape(targets, (targets.shape[0]*targets.shape[1], *targets.shape[2:]))

    for pred, target in zip(preds, targets):
        if np.sum(target) != 0:
            slice_hd95 = slicewise_hd95(pred, target, n_classes)
            batch_res.append(slice_hd95)

    mean_hd95 = sum(batch_res) / len(batch_res)

    if return_slicewise_hdf:
        return mean_hd95, batch_res
    else:
        return mean_hd95

def slicewise_ahd(pred, target, n_classes=3):
    """ calculate Average Hausdorff distance between pred and target of single image
    :param pred: ndarray with size [H, W],  predicted bound
    :param target: ndarray with size [H, W], ground truth
    :param n_classes: int, # of classes
    """

    max_ahd = 2 * target.shape[0]
    slice_res = []
    for c_inx in range(1, n_classes):
        set1 = np.array(np.where(pred == c_inx)).transpose()
        set2 = np.array(np.where(target == c_inx)).transpose()
        if len(set2) != 0:
            if len(set1) == 0:
                res = max_ahd
            else:
                d2_matrix = pairwise_distances(set1, set2, metric='euclidean')
                res = np.average(np.min(d2_matrix, axis=0)) + \
                      np.average(np.min(d2_matrix, axis=1))

            slice_res.append(res)

    mean_res = sum(slice_res) / len(slice_res)

    return mean_res

def slicewise_hd95(pred, target, n_classes=3):
    """ calculate Average Hausdorff distance between pred and target of single image
    :param pred: ndarray with size [H, W],  predicted bound
    :param target: ndarray with size [H, W], ground truth
    :param n_classes: int, # of classes
    """
    max_hd95 = 2 * target.shape[0]
    slice_res = []
    for c_inx in range(1, n_classes):
        pred_cinx = (pred == c_inx)
        target_cinx = (target == c_inx)
        if np.sum(target_cinx) != 0:
            if np.sum(pred_cinx) == 0:
                res = max_hd95
            else:
                res = hd95(pred_cinx, target_cinx)

            slice_res.append(res)

    mean_res = sum(slice_res) / len(slice_res)

    return mean_res

def channelwise_ahd(pred, target):
    """ calculate channel wise average Hausdorff distance
    :param pred: ndarray with size [C, H, W], predicted boundary
    :param target: ndarray with size [C, H, W], GT boundary
    :return mean_res: float, average hdf
    """
    max_ahd = 2 * target.shape[1]
    slice_res = []
    for c_inx in range(0, len(pred)):
        set1 = np.array(np.where(pred[c_inx])).transpose()
        set2 = np.array(np.where(target[c_inx])).transpose()
        if len(set2) != 0:
            if len(set1) == 0:
                res = max_ahd
            else:
                d2_matrix = pairwise_distances(set1, set2, metric='euclidean')
                res = np.average(np.min(d2_matrix, axis=0)) + \
                      np.average(np.min(d2_matrix, axis=1))

            slice_res.append(res)

    mean_res = sum(slice_res) / len(slice_res)

    return mean_res

def channelwise_hd95(pred, target):
    """ calculate channel-wise 95 percentile symmetric Hausdorff distance
    :param pred: ndarray with size [C, H, W], predicted boundary
    :param target: ndarray with size [C, H, W], GT boundary
    :return mean_hd95: float, average hdf
    """
    max_hd95 = 2 * target.shape[1]
    channel_res = []
    for c_inx in range(0, len(pred)):
        if np.sum(target[c_inx]) != 0:
            if np.sum(pred[c_inx]) == 0:
                res = max_hd95
            else:
                res = hd95(pred[c_inx], target[c_inx])
            channel_res.append(res)

    mean_hd95 = sum(channel_res) / len(channel_res)

    return mean_hd95


def channelwise_asd(pred, target):
    """ calculate channel-wise average symmetric surface distance """
    max_asd = 2 * target.shape[1]
    channel_res = []
    for c_inx in range(0, len(pred)):
        if np.sum(target[c_inx]) != 0:
            if np.sum(pred[c_inx]) == 0:
                res = max_asd
            else:
                res = asd(pred[c_inx], target[c_inx])

            channel_res.append(res)

    mean_asd = sum(channel_res) / len(channel_res)

    return mean_asd

def slicewise_asd(pred, target, n_classes=3):
    """ calculate slice-wise average symmetric surface distance
        :param pred: ndarray with size [H, W],  predicted bound
        :param target: ndarray with size [H, W], ground truth
        :param n_classes: int, # of classes
    """

    slice_res = []
    max_asd = 2 * target.shape[0]
    for c_inx in range(1, n_classes):
        pred_cinx = (pred == c_inx)
        target_cinx = (target == c_inx)
        if np.sum(target_cinx) != 0:
            if np.sum(pred_cinx) == 0:
                res = max_asd
            else:
                res = asd(pred_cinx, target_cinx)

            slice_res.append(res)

    mean_res = sum(slice_res) / len(slice_res)

    return mean_res

def volumewise_asd(preds, targets, n_classes=3):
    """ calculate volume-wise asd """

    vol_res = []
    for pred, target in zip(preds, targets):
        if np.sum(target) != 0:
            res = slicewise_asd(pred, target, n_classes)
            vol_res.append(res)

    return sum(vol_res) / len(vol_res)

def slicewise_ravd(pred, target):
    # in this case ravd should not be calculated for this slice
    if np.sum(target) == 0:
        raise IOError("target should contain at least one nonzero pixel")
    else:
        return abs(ravd(pred, target))

def volumewise_ravd(preds, targets):
    """ calculate volume-wise ravd """
    ravds = []
    for pred, target in zip(preds, targets):
        if np.sum(target) != 0:
            ravds.append(slicewise_ravd(pred, target))

    return (sum(ravds) / len(preds))

def cal_f_score(preds, labels, n_class=5, return_class_f1= False, return_slice_f1=False):
    """ calculate average f1_score of given output and target batch
    outputs: Variable(n_batches, n_classes, *), model output (* means any dimensions)
    labels: Variable(n_batches, *), label (* means any dimensions)
    ignore_index: int, ignored index when calculating f1 score
    """

    if not isinstance(preds, np.ndarray):
        n_batch = preds.size(0)
        preds_np, labels_np = preds.cpu().numpy(), labels.data.cpu().numpy()
    else:
        n_batch = preds.shape[0]
        preds_np, labels_np = preds, labels

    f_scores = np.zeros(n_class, dtype=np.float32)
    n_effect_samples = np.zeros(n_class, dtype=np.uint32)
    ave_f_score_batch = 0.0
    f_scores_batch = []

    for pred, label in zip(preds_np, labels_np):
        if n_class > 2:
            label_binary = label_binarize(label.flatten(), classes=range(n_class))
            pred_binary = label_binarize(pred.flatten(), classes=range(n_class))
        else:
            label_binary = np.stack([1 - label.flatten(), label.flatten()], axis=1)
            pred_binary = np.stack([1 - pred.flatten(), pred.flatten()], axis=1)

        f_score = np.zeros(n_class, dtype=np.float32)
        effect_class_per_slice = np.ones(n_class, dtype=np.uint8)

        for i in range(n_class):
            if np.sum(label_binary[:, i]) == 0:
                f_score[i] = 0.0
                effect_class_per_slice[i] = 0

            else:
                n_effect_samples[i] += 1
                f_score[i] = f1_score(label_binary[:, i], pred_binary[:, i])

        f_scores += f_score
        if n_class > 2:
            ave_f_score_per_slice =  f_score.sum() / effect_class_per_slice.sum()
        else:
            ave_f_score_per_slice = f_score[1]
        # calculate f1 score for each batch to do hard mining
        f_scores_batch.append(ave_f_score_per_slice)
        ave_f_score_batch += ave_f_score_per_slice

    ave_f_score_batch = ave_f_score_batch / n_batch


    if return_class_f1 and return_slice_f1:
        return ave_f_score_batch, f_scores, n_effect_samples, f_scores_batch
    elif return_class_f1:
        return ave_f_score_batch, f_scores, n_effect_samples
    elif return_slice_f1:
        return ave_f_score_batch, f_scores_batch
    else:
        return ave_f_score_batch

def cal_f_score_slicewise(preds, labels, n_class=5, return_class_f1=False, return_slice_f1=False):
    """ calculate average f1_score of given output and target batch
    outputs: Variable(n_batches, n_classes, D, H, W), model output
    labels: Variable(n_batches, D, H, W), label
    return_class_f1: bool, whether return class-wise F1 score or not
    return_slice_f1: bool, whether return slice-wise/volume-wise F1 score or not
    """

    # if tensor data, convert to numpy first
    if not isinstance(preds, np.ndarray):
        n_batch = preds.size(0)
        preds_np, labels_np = preds.cpu().numpy(), labels.data.cpu().numpy()
    else:
        n_batch = preds.shape[0]
        preds_np, labels_np = preds, labels

    # print("preds shape: {}".format(preds_np.shape))
    f_scores = np.zeros(n_class, dtype=np.float32)
    n_effect_samples = np.zeros(n_class, dtype=np.uint32)
    ave_f_score_batch = 0.0
    f_scores_batch = []
    n_slices = 0

    for pred_vol, label_vol in zip(preds_np, labels_np):
        n_slices += len(pred_vol)
        # calculate average F1 score for a volume
        f_scores_vol = 0.0
        for pred, label in zip(pred_vol, label_vol):
            label_binary = label_binarize(label.flatten(), classes=range(n_class))
            pred_binary = label_binarize(pred.flatten(), classes=range(n_class))

            f_score = np.zeros(n_class, dtype=np.float32)
            effect_class_per_slice = np.ones(n_class, dtype=np.uint8)
            for i in range(n_class):
                if np.sum(label_binary[:, i]) == 0:
                    f_score[i] = 0.0
                    effect_class_per_slice[i] = 0

                else:
                    n_effect_samples[i] += 1
                    f_score[i] = f1_score(label_binary[:, i], pred_binary[:, i])

            f_scores += f_score
            ave_f_score_per_slice =  f_score.sum() / effect_class_per_slice.sum()

            f_scores_vol += ave_f_score_per_slice
            ave_f_score_batch += ave_f_score_per_slice

        f_scores_batch.append(f_scores_vol/len(pred_vol))

    ave_f_score_batch = ave_f_score_batch / n_slices

    if return_class_f1 and return_slice_f1:
        return ave_f_score_batch, f_scores, n_effect_samples, f_scores_batch
    elif return_class_f1:
        return ave_f_score_batch, f_scores, n_effect_samples
    elif return_slice_f1:
        return ave_f_score_batch, f_scores_batch
    else:
        return ave_f_score_batch

def slicewise_multiclass_f1(pred, label, n_class=3):
    """ calculate slice-wise F1 score with multi-class outputs
    :param pred: ndarray of size [H, W], predicted seg result
    :param label: ndarray of size [H, W], GT seg annotation
    :return: ave_slice_f1
    """

    label_binary = label_binarize(label.flatten(), classes=range(n_class))
    pred_binary = label_binarize(pred.flatten(), classes=range(n_class))

    f_score = np.zeros(n_class, dtype=np.float32)
    n_effect_class = 0
    for i in range(n_class):
        if np.sum(label_binary[:, i]) == 0:
            f_score[i] = 0.0
        else:
            n_effect_class += 1
            f_score[i] = f1_score(label_binary[:, i], pred_binary[:, i])

    ave_slice_f1 = np.sum(f_score) / n_effect_class

    return ave_slice_f1


# def bin_dice_score(preds, labels):
#     """ calculate binary dice from preds and labels
#     Args:
#         preds: Binary LongTensor [N, H, W, T], predicted label
#         labels: Binary LongTensor [N, H, W, T], GT labels
#     """
#     smooth = 1.0
#     if not isinstance(preds, np.ndarray):
#         preds = preds.cpu().numpy()
#         labels = labels.cpu().numpy()
#
#     n_batches = preds.shape[0]
#
#     dice_score = 0.0
#
#     for (pred, label) in zip(preds, labels):
#         intersection = pred * label
#         numerator = 2 * intersection.sum() + smooth
#
#         # output^2 is not used here
#         denominator = (pred + label).sum() + smooth
#         # print "numerator: {}".format(numerator)
#         # print "denominator: {}".format(denominator)
#         dice_score += (numerator / denominator)
#
#     return dice_score / n_batches
#
#
# def bin_f_score(preds, labels):
#     """ calculate binary f1 score from preds and labels
#     Args:
#         preds: Binary LongTensor, predited label
#         labels: Binary LongTensor, GT labels
#     """
#     if not isinstance(preds, np.ndarray):
#         preds = preds.cpu().numpy()
#         labels = labels.cpu().numpy()
#
#     n_batches = preds.shape[0]
#
#     f_score = 0.0
#     for (pred, label) in zip(preds, labels):
#         pred, label = pred.flatten(), label.flatten()
#         if np.sum(label) == 0:
#             if np.sum(pred) == 0:
#                 f_batch = 1.0
#             else:
#                 f_batch = 0.0
#         else:
#             f_batch = f1_score(label, pred)
#
#         f_score += f_batch
#
#     return f_score/n_batches

if __name__ == "__main__":
    label = torch.zeros(1, 64, 64, 16).long()
    label[0:8, 0:32, 0:32, 0:3] = 1
    pred = torch.zeros(1, 64, 64, 16).long()
    # pred[9:12, 16:48, 16:48, 2:7] = 1
    print("Dice score: {}".format(bin_dice_score(label, pred)))
    print("F1 score: {}".format(bin_f_score(label, pred)))