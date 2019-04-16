# _*_ coding: utf-8 _*_

""" define custom loss functions """

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import CrossEntropyLoss

from skimage import io
from utils import gray2mask, gray2innerouterbound, gray2bound

import torch
from matplotlib import pyplot as plt

from torch import nn
from torch.autograd import Function

torch.set_default_dtype(torch.float32)

# from mpl import mpl

import math
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy import ndimage

torch.set_default_dtype(torch.float32)

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


class DiceLoss(Function):
    """ Normal Dice Loss for multi-class segmentation """

    def __init__(self, weight=None, ignore_index=None, weight_type=None, reduce=True, cal_zerogt=False):
        self.weight = weight
        self.ignore_index = ignore_index
        self.weight_type = weight_type
        self.reduce = reduce
        self.cal_zerogt = cal_zerogt # whether calculate Dice for case of all GT pixels are zero

    def __call__(self, output, target):
        # output : N x C x *,  Variable of float tensor (* means any dimensions)
        # target : N x *, Variable of long tensor (* means any dimensions)
        # weights : C, float tensor
        # ignore_index : int, class index to ignore from loss (0 for background)
        smooth = 1.0e-9
        output = F.softmax(output, dim=1)
        n_classes = output.size(1)

        if output.size() != target.size():
            target = target.data
            encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros

            if self.ignore_index is not None:
                mask = (target == self.ignore_index)
                target = target.clone()
                target[mask] = 0
                encoded_target.scatter_(1, target.unsqueeze(1), 1)
                mask = mask.unsqueeze(1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                unseq = target.long()
                encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

            encoded_target = Variable(encoded_target, requires_grad = False)

        else:
            encoded_target = target

        # calculate gt, t and p from perspective of 1
        intersection = output * encoded_target
        numerator = 2 * torch.sum(intersection.view(*intersection.size()[:2], -1), 2) + smooth
        denominator1 = torch.sum(output.view(*output.size()[:2], -1), 2)
        denominator2 = torch.sum(encoded_target.view(*encoded_target.size()[:2], -1), 2)
        denominator = denominator1 + denominator2 + smooth
        mask_gt = (denominator2 == 0)

        if self.weight is None:
            if self.weight_type is None:
                weight =  Variable(torch.ones(n_classes).cuda(), requires_grad=False)
            else:
                tmp = denominator2.sum(0)
                tmp =  tmp / tmp.sum()

                if self.weight_type == 'nlf':
                    weight = -1.0 * torch.log(tmp + smooth)

                elif self.weight_type == 'mfb':
                    weight = torch.median(tmp) / (tmp + smooth)

            weight = weight.detach()

        else: # prior weight is setting manually
            weight = self.weight

        loss_per_channel = weight * (1.0 - (numerator / denominator))

        # calculate Dice for special case of all GT pixels are zero
        if self.cal_zerogt:
            output_com = 1.0 - output
            encoded_target_com = 1 - encoded_target
            intersection_com = output_com * encoded_target_com
            numerator_com = 2 * torch.sum(intersection_com.view(*intersection_com.size()[:2], -1), 2) + smooth
            denominator1_com = torch.sum(output_com.view(*output_com.size()[:2], -1), 2)
            denominator2_com = torch.sum(encoded_target_com.view(*encoded_target_com.size()[:2], -1), 2)
            denominator_com = denominator1_com + denominator2_com + smooth
            loss_per_channel_com = weight * (1.0 - (numerator_com / denominator_com))

        loss_per_channel = loss_per_channel.clone()
        # if all GT pixels are zero, use the complementary pixels for calculation
        if self.cal_zerogt:
            loss_per_channel[mask_gt] = loss_per_channel_com[mask_gt]
        else:
            loss_per_channel[mask_gt] = 0

        if self.reduce:
            if self.cal_zerogt:
                dice_loss = loss_per_channel.mean()
            else:
                loss_ave_class = loss_per_channel.sum() / (mask_gt == 0).sum(1).float()
                dice_loss = loss_ave_class.mean()
        else:
            dice_loss = loss_per_channel  # [N, C]

        return dice_loss

class MaxPoolLoss(Function):
    """ loss max-pooling defined in 'Loss Max-Pooling for Semantic Image Segmentation' """
    def __init__(self, criterion, ratio=0.3, p=1.3):
        self.criterion = criterion
        self.mpl = mpl.MaxPoolingLoss(ratio, p, reduce=True)

    def __call__(self, output, target):
        self.criterion.reduce = False
        if isinstance(self.criterion, nn.NLLLoss):
            output = F.log_softmax(output, dim=1)

        loss = self.criterion(output, target)
        loss = self.mpl(loss)

        return loss

class WeightedKLDivLoss(Function):
    """ weighted KL-divergence loss for batch with imbalanced class distribution
        correctness has already been checked
    """

    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

    def __call__(self, output, target):
        """ forward propagation
        :param output: Variable of output [N x C x *]
        :param target: Variable of GT prob [N x C x *]
        """
        smooth = 1.0e-9
        output = F.log_softmax(output, 1)
        kl_div = target * (torch.log(target + smooth) - output)
        kl_div = kl_div.permute(0, *range(2, len(kl_div.size())), 1)

        if self.weight is not None:
            kl_div = self.weight * kl_div

        if self.size_average:
            loss = kl_div.mean()
        else:
            loss = kl_div.sum()

        return loss

def dice_score(output, target, ignore_index=None, weight=None):
    """ calculate batch-wise dice score """
    smooth = 1.0e-9
    target = target.data
    output = F.softmax(output, dim=1)
    _, pred = torch.max(output.data, 1)

    encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros
    encoded_pred = output.data.clone().zero_()
    encoded_pred.scatter_(1, pred.unsqueeze(1), 1)

    if ignore_index is not None:
        mask = (target == ignore_index)
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        unseq = target.long()
        encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

    intersection = encoded_pred * encoded_target
    numerator = 2 * torch.sum(intersection.view(*intersection.size()[:2], -1), 2) + smooth
    denominator1 = torch.sum(encoded_pred.view(*encoded_pred.size()[:2], -1), 2)
    denominator2 = torch.sum(encoded_target.view(*encoded_target.size()[:2], -1), 2)

    mask_gt = (denominator2 == 0)
    # print(mask_gt)
    denominator = denominator1 + denominator2 + smooth
    if weight is not None:
        dice_per_channel = weight * (numerator / denominator)
    else:
        dice_per_channel = numerator / denominator

    dice_per_channel = dice_per_channel.clone()
    dice_per_channel[mask_gt] = 0
    dice_ave_class = dice_per_channel.sum(1)/(mask_gt==0).sum(1).float()

    dice_score = dice_ave_class.mean().item()

    return dice_score

def dice_score_slicewise(output, target, ignore_index=None, weight=None):
    """ calculate dice score slice-wisely for 3D volume """
    smooth = 1.0e-9
    target = target.data
    output = F.softmax(output, dim=1)
    _, pred = torch.max(output.data, 1)

    encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros
    encoded_pred = output.data.clone().zero_()
    encoded_pred.scatter_(1, pred.unsqueeze(1), 1)

    if ignore_index is not None:
        mask = (target == ignore_index)
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        unseq = target.long()
        encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

    intersection = encoded_pred * encoded_target
    numerator = 2 * intersection.sum(4).sum(3) + smooth

    denominator1 = encoded_pred.sum(4).sum(3)
    denominator2 = encoded_target.sum(4).sum(3)  # [N, C, D]
    mask_gt = (denominator2 == 0)
    # print(mask_gt)
    denominator = denominator1 + denominator2 + smooth
    if weight is not None:
        dice_per_channel = weight * (numerator / denominator)
    else:
        dice_per_channel = numerator / denominator

    dice_per_channel = dice_per_channel.clone()
    dice_per_channel[mask_gt] = 0
    dice_ave_class = dice_per_channel.sum(1)/(mask_gt==0).sum(1).float()  # [N, D]

    dice_score = dice_ave_class.mean().item()

    return dice_score

class GeneralizedDiceLoss(Function):
    """ generalized dice score for multi-class segmentation defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
            loss function for highly unbalanced segmentations. DLMIA 2017
        weight for each class is calculated from the distribution of Ground
        Truth pixel/voxel belonging to each class
    """

    def __init__(self, weight=None, ignore_index=None, weight_type='inv_square',
                 alpha=0.5):
        """
        :param weight_type: str, in which way to calculate class weight
            here we consider 3 strategies for deciding proper class weight
            (1) inv_square class weight is set as inverse of summing up square of
                ground truth for each class
            (2) others_one_pred: class weight is set as others_over_one ratio of predicted
                probabilities for each class
            (3) others_one_gt: class weight is set as others_over_one ratio of ground truth
                for each class
        :param alpha: float, ratio of false positive
        :param beta: float, ratio of false negative
            increase alpha if you care more about false positive and beta otherwise
        """
        self.ignore_index = ignore_index
        self.weight = weight
        self.weight_type = weight_type
        self.alpha= alpha
        self.beta = 1- self.alpha

    def __call__(self, output, target):
        # output : N x C x *,  Variable of float tensor (* means any dimensions)
        # target : N x *, Variable of long tensor (* means any dimensions)
        # weights : C, float tensor
        # ignore_index : int, class index to ignore from loss (0 for background)
        # back propagation is checked to be correct

        smooth = 1.0e-9
        output = F.softmax(output, 1)
        n_pixels = output[:, 0].numel()
        n_classes = output.size(1)

        if output.size() != target.size():
            # for normal input
            target = target.data
            encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros

            if self.ignore_index is not None:
                mask = (target == self.ignore_index)
                target = target.clone()
                target[mask] = 0
                encoded_target.scatter_(1, target.unsqueeze(1), 1)
                mask = mask.unsqueeze(1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                unseq = target.long()
                encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

            encoded_target = Variable(encoded_target, requires_grad=False)

        else:
            # for BC learning input
            encoded_target = target

        tp = output * encoded_target
        fp = output * (1-encoded_target)
        fn = (1.0 - output) * encoded_target

        # add along all dimensions except the first (n_batch) and the second (n_class) dim
        gt_sum = torch.sum(encoded_target.view(*encoded_target.size()[:2], -1), 2).sum(0)
        mask_gt = (gt_sum == 0)
        tp_sum = torch.sum(tp.view(*tp.size()[:2], -1), 2).sum(0)
        fp_sum = torch.sum(fp.view(*fp.size()[:2], -1), 2).sum(0)
        fn_sum = torch.sum(fn.view(*fn.size()[:2], -1), 2).sum(0)

        numerator = tp_sum
        denominator = tp_sum + self.alpha * fp_sum + self.beta * fn_sum

        if self.weight is None:
            if self.weight_type is None:
                weight =  Variable(torch.ones(n_classes).cuda(), requires_grad=False)
            else:
                if self.weight_type == 'inv_square':
                    weight = 1.0 / (gt_sum.pow(2) + smooth)
                    weight[gt_sum==0] = 0.0
                elif self.weight_type == 'others_one_pred':
                    prob_sum_per_class = torch.sum(output.view(*output.size()[:2], -1), 2).sum(0)
                    weight = (n_pixels - prob_sum_per_class) / prob_sum_per_class
                elif self.weight_type == 'others_one_gt':
                    weight = (n_pixels - gt_sum) / (gt_sum + smooth)
                    weight[gt_sum==0] = 0.0

            weight = weight.detach()

        else:
            weight = self.weight

        loss = 1.0 - (weight * numerator).sum() / (weight * denominator).sum()

        return loss

class WeightedCrossEntropy(CrossEntropyLoss):
    """ weighted cross entropy for multi-class semantic segmentation defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True,
                 weight_type='log_inv_freq'):
        super(WeightedCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce)
        self.weight_type = weight_type

    def forward(self, output, target):
        """ weighted cross entropy where weight is calculated from input data
        :param output: Variable, N x C x *, probabilities for each class
        :param target: Variable, N x *, GT labels
        """
        if self.weight is None:
            output_prob = F.softmax(output, 1)
            prob_per_class = torch.sum(output_prob.view(*output_prob.size()[:2], -1), 2).sum(0)
            prob_sum = prob_per_class.sum()

            if self.weight_type == 'others_over_one':
                weight = (prob_sum - prob_per_class)/prob_per_class
            elif self.weight_type == 'log_inv_freq':
                weight = torch.log(prob_sum/prob_per_class)
        else:
            weight = self.weight

        return F.cross_entropy(output, target, weight, self.size_average,
                                   self.ignore_index, self.reduce)

class FocalLoss(Function):
    """ focal loss for multi-class object detection defined in
    Tsung-Yi Lin et. al. Focal Loss for Dense Object Detection. CVPR 2017
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, output, target):
        """ weighted cross entropy where weight is calculated from input data
        :param output: Variable, N x C x *, probabilities for each class
        :param target: Variable, N x *, GT labels
        """
        encoded_target = output.data.clone().zero_()
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        encoded_target = Variable(encoded_target, requires_grad=False)

        prob = output.sigmoid()
        pt = prob*encoded_target + (1-prob)*(1-encoded_target)         # pt = p if t > 0 else 1-p
        # w is given for every element
        weight = self.alpha*encoded_target + (1-self.alpha)*(1-encoded_target)  # w = alpha if t > 0 else 1-alpha
        weight = weight * (1-pt).pow(self.gamma)

        # since loss decay very fast, sum is returned instead of average
        return F.binary_cross_entropy_with_logits(output, encoded_target, weight, size_average=False)


def bound_weight_withdiff(target, sigmas=[5.0, 5.0], ws=[20.0, 10.0], n_classes=3, bound_output=True):
    """ calculate boundary weight of each pixel given GT mask
        this is a GPU version implementation
        For more information, please refer to the original paper of U-Net
        :param target: tensor, [N, H, W], target
        :param sigmas: float, variance of Gaussian pdf of bounds
        :param ws: float, amplitude of Gaussian pdf of bounds
        :param n_classes: int, # of classes
        :param bound_output: bool, whether bound input or not
        :return: weights, tensor, [N, H, W], weights for each pixel
    """

    h, w = target.size()[1:]
    if n_classes == 3: # target is boundary map [1--inner bound, 2--outer bound]
        if bound_output:
            inner_bound = (target == 1)
            outer_bound = (target == 2)
            bounds = torch.stack([outer_bound, inner_bound], dim=1)

        else: # target is background + central_part + outline
            t_pad = F.pad(target, (1, 1, 1, 1))
            mask = torch.zeros(t_pad.size(0), n_classes, *t_pad.size()[1:]).cuda()
            mask.scatter_(1, t_pad.unsqueeze(1), 1)

            conv_filter = torch.ones(1, 1, 3, 3).cuda()

            y = torch.zeros(target.size(0), n_classes, h, w).cuda()

            for i in range(n_classes):
                tmp = mask[:, i].unsqueeze(1)
                y[:,i] = F.conv2d(tmp, conv_filter, padding=0).squeeze(1)

            y = y.long() # [N, C, H, W]
            y[y == 9] = 0 # all pixels within the kernel are equal to 1
            bounds = y[:, :2].long() # outer bound and inner bound

    elif n_classes == 5: # treat inner bound and outer bound differently
        inner_bound = (target == 3)
        outer_bound = (target == 4)
        bounds = torch.stack([outer_bound, inner_bound], dim=1)

    pixel_cords = torch.meshgrid([torch.arange(h).cuda(), torch.arange(w).cuda()]) # 2, H, W
    pixel_cords = torch.stack(pixel_cords).float()

    weights = target.clone().zero_().float()

    for ib, bound in enumerate(bounds):
        for ic, bound_c in enumerate(bound): # bound in different channel [outer, inner]
            b_cords = torch.nonzero(bound_c).float()  # N, 2
            if len(b_cords) > 0: # in case of no inner bound
                tmp = pixel_cords.repeat(len(b_cords), 1, 1, 1).permute(2, 3, 0, 1).float()
                # print(tmp.size())
                tmp = (tmp - b_cords).norm(dim=-1)
                tmp = torch.min(tmp, dim=-1)[0] # shortest distance between bound and each cord
                weights[ib] += ws[ic] * torch.exp(-0.5 * tmp ** 2 / sigmas[ic] ** 2) # [H, W]

    return weights

def bound_weight_wodiff(target, sigma=5.0, w0=10.0, n_classes=2, k=2):
    """ calculate boundary weight of each pixel given GT mask
        this is a GPU version implementation
        inner bounds and outer bounds are treated as the same
        For more information, please refer to the original paper of U-Net
        :param target: tensor, [N, H, W], target
        :param sigma: float, variance of Gaussian pdf
        :param w0: float, aptitude of Gaussian pdf
        :param n_classes: int, # of classes
        :param k: int, top k shortest distances to calculate
        :return: weights, tensor, [N, H, W], weights for each pixel
    """

    h, w = target.size()[1:]
    if n_classes == 2: # boundary detection
        bounds = target
    elif n_classes == 3:
        t_pad = F.pad(target, (1, 1, 1, 1))
        mask = torch.zeros(t_pad.size(0), n_classes, *t_pad.size()[1:]).cuda()
        mask.scatter_(1, t_pad.unsqueeze(1), 1)

        conv_filter = torch.ones(1, 1, 3, 3).cuda()

        y = torch.zeros(target.size(0), n_classes, h, w).cuda()

        for i in range(n_classes):
            tmp = mask[:, i].unsqueeze(1)
            y[:,i] = F.conv2d(tmp, conv_filter, padding=0).squeeze(1)

        y = y.long()
        y[y == 9] = 0 # all pixels within the kernel are equal to 1
        bounds = y.sum(1).long()

    elif n_classes >= 4:
        bounds = (target == 3) | (target == 4)

    pixel_cords = torch.meshgrid([torch.arange(h).cuda(), torch.arange(w).cuda()]) # 2, H, W
    pixel_cords = torch.stack(pixel_cords).float()

    weights = target.clone().zero_().float()

    for i, b in enumerate(bounds):
        b_cords = torch.nonzero(b).float()  # N, 2
        # print(b_cords.size())
        tmp = pixel_cords.repeat(len(b_cords), 1, 1, 1).permute(2, 3, 0, 1).float()
        # print(tmp.size())
        tmp = (tmp - b_cords).norm(dim=-1)
        tmp = torch.topk(tmp, k, largest=False, dim=-1)[0]
        weights[i] = w0 * torch.exp(-0.5 * torch.sum(tmp, dim=-1) ** 2 / sigma ** 2) # [H, W]

    return weights

class CrossEntropyBoundLoss(Function):
    """ define a new cross entropy loss considering boundaries
        pixels which are closer to the boundaries, higher weight is assigned
    """

    def __init__(self, sigmas=[5.0, 5.0], ws=[20.0, 10.0], n_classes=3,
                 weight=None, ignore_index=None, bound_output=True, k=2):
        self.sigmas = sigmas
        self.ws = ws
        self.n_classes = n_classes
        self.weight = weight
        self.ignore_index = ignore_index
        self.bound_output = bound_output
        self.k = k

    def __call__(self, output, target):
        """ cross entropy loss considering boundaries
            :param output: Variable, N x C x *, probabilities for each class
            :param target: Variable, N x *, GT labels
        """

        # make sure to calculate the bound weight before target changes
        if self.n_classes == 2:
            w_bound = bound_weight_wodiff(target, self.sigmas[0], self.ws[0], self.n_classes, self.k)
        elif self.n_classes == 3:
            w_bound = bound_weight_withdiff(target, self.sigmas, self.ws, self.n_classes, self.bound_output) # [N, H, W]
        weight = w_bound.repeat(self.n_classes, 1, 1, 1).permute(1, 0, 2, 3)  # [N, C, H, W]

        encoded_target = output.data.clone().zero_()

        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            unseq = target.long()
            encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

        if self.weight is not None:
            w_prior = encoded_target.permute(0, 2, 3, 1) * self.weight  # [N, H, W, C]
            weight = weight + w_prior.permute(0, 3, 1, 2)

        weight = weight.detach()

        encoded_target = Variable(encoded_target, requires_grad=False)

        return F.binary_cross_entropy_with_logits(output, encoded_target, weight, reduction="elementwise_mean")


############### weighted Hausdorff distance loss ###############
#   probability of each class is included so that the loss can be back-propagated

def cdist(x, y):
    ''' :param x: Tensor of size Nxd
    :param y: Tensor of size Mxd
    :return dist: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:]
                  i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

class WeightedHausdorffDistanceLoss(Function):
    """ weighted HausdorffDistanceLoss defined in
        "Weighted Hausdorff Distance: A Loss Function For Object Localization". Javier Ribera.
        for more information, please refer to the original paper.
    """
    def __init__(self, return_2_terms=False, alpha=4, beta=2):
        """
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        """
        self.return_2_terms = return_2_terms
        self.alpha = alpha
        self.beta = beta

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) or (B x D x H x W), Tensor of the probability map of the estimation.
        :param gt: (B x H x W) or (B x D x H x W), Tensor of the GT annotation
        """

        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)
        # assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        if prob_map.dim() == 4 and gt.dim() == 4:
            prob_map = prob_map.contiguous().view(prob_map.size(0) * prob_map.size(1), *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(gt.size(0) * gt.size(1), *gt.size()[2:])                         # [B*D, H, W]

        batch_size, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            gt_pts = torch.nonzero(gt_b).float() # N, 2
            n_gt_pts = gt_pts.size()[0]

            if n_gt_pts > 0:
                d_matrix = cdist(all_img_locations, gt_pts)
                p = prob_map_b.view(prob_map_b.numel())

                n_est_pts = (p**beta).sum()
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                # term 1
                term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                d_div_p = torch.min((d_matrix + eps) /
                                    (p_replicated**alpha + eps / max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, max_dist)
                term_2 = torch.mean(d_div_p)

                terms_1.append(term_1)
                terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res

class WeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # here we consider inner bound and outer bound respectively

        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel)]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds]
        res_boundwise = torch.stack(res_bounds_mean)  # convert list into torch array
        # ratio: inner bound ratio
        res = res_boundwise[0] * self.ratio + res_boundwise[1] * (1.0 - self.ratio)

        if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
            return res, res_boundwise
        else:
            return res


class WeightedHausdorffDistanceDoubleBoundLossWithSnake(Function):
    def __init__(self, return_multi_loss=False, alpha=4, beta=1, ratio=0.5, eps=1e-6):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
            snake constraint is applied to
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_multi_loss = return_multi_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.eps = eps

    def __call__(self, prob_map, gt, snake):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        :param snake: (B x H x W) Tensor of snake annotation
        """

        _assert_no_grad(gt)
        _assert_no_grad(snake)

        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == gt.size(0), 'prob map and GT must have the same size'
        assert batch_size == snake.size(0), 'prob_map and snake must have the same size'

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # here we consider GT annotation and snake respectively
        res_bounds_lst = [[] for _ in range(0, n_channel-1)]
        for b in range(batch_size):
            prob_map_b, gt_b, snake_b = prob_map[b], gt[b], snake[b]
            res_gt = self.weighted_hausdorff_distance(gt_b, prob_map_b, all_img_locations)
            res_snake = self.weighted_hausdorff_distance(snake_b, prob_map_b, all_img_locations)
            res_bounds_lst[0].append(res_gt)
            res_bounds_lst[1].append(res_snake)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(0, n_channel-1)]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds]
        res_boundwise = torch.stack(res_bounds_mean)  # convert list into torch array
        # mean of pred loss and reg loss
        res = res_boundwise.mean()

        if self.return_multi_loss:  # return inner bound loss and outer bound loss respectively
            return res, res_boundwise
        else:
            return res

    def weighted_hausdorff_distance(self, gt_b, prob_map_b, all_img_locations):
        """ calculate weighted Hausdorff distance
        :param gt: long tensor of size [H, W]
        :param prob_map: float tensor of size [C, H, W]
        """
        n_channel, height, width = prob_map_b.size()
        max_dist = math.sqrt(height ** 2 + width ** 2)
        res_bounds_lst = []

        for bound_inx in range(1, n_channel):
            gt_bb = (gt_b == bound_inx)  # for different bounds (1 - inner, 2 - outer)
            gt_pts = torch.nonzero(gt_bb).float()  # N, 2
            n_gt_pts = gt_pts.size()[0]
            prob_map_bb = prob_map_b[bound_inx]
            # print("# of GT points: {}".format(n_gt_pts))
            if n_gt_pts > 0:
                # loss1(GT bound and pred) and loss2(GT snake and pred) are calculated
                d_matrix = cdist(all_img_locations, gt_pts)
                p = prob_map_bb.view(prob_map_bb.numel())
                n_est_pts = (p ** self.beta).sum()
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                term_1 = (1 / (n_est_pts + self.eps)) * torch.sum(p ** self.beta * torch.min(d_matrix, 1)[0])
                d_div_p = torch.min((d_matrix + self.eps) /
                                    (p_replicated ** self.alpha + self.eps / max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, max_dist)
                term_2 = torch.mean(d_div_p)

                res_bounds_lst.append(term_1 + term_2)
        if len(res_bounds_lst) >= 2:
            res = res_bounds_lst[0] * self.ratio + res_bounds_lst[1] * (1.0 - self.ratio)
        else:
            res = res_bounds_lst[0]

        return res

class WeightedMaximumHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
        """ whd loss for inner and outer bound separately
            instead of averaged whd, maximum whd is used
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())
                    
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
                    term_1 = torch.max(p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.max(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel)]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds]
        res_boundwise = torch.stack(res_bounds_mean)  # convert list into torch array
        # ratio: inner bound ratio
        res = res_boundwise[0] * self.ratio + res_boundwise[1] * (1.0 - self.ratio)

        if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
            return res, res_boundwise
        else:
            return res

class ModifiedWeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5, thres=0.5):
        """ whd loss for inner and outer bound separately
            the loss is modified to only calculate distances between GT pixels and predicted boundary pixels with
            probability higher than pre-set threshold
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.thres = thres

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]

                # filter out estimated probs those are lower than threshold
                p = prob_map_bb.view(prob_map_bb.numel())
                p_sel = p[p >= self.thres]
                all_img_locations_sel = all_img_locations[p >= self.thres]
                # print("# img locations : {}, # GT locations : {}".format(len(all_img_locations_sel), n_gt_pts))

                if n_gt_pts > 0 and len(all_img_locations_sel) > 0:
                    d_matrix = cdist(all_img_locations_sel, gt_pts)
                    n_est_pts = (p_sel**beta).sum()
                    p_replicated = p_sel.view(-1, 1).repeat(1, n_gt_pts)

                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p_sel**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel)]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds]
        res_boundwise = torch.stack(res_bounds_mean)  # convert list into torch array
        # ratio: inner bound ratio
        res = res_boundwise[0] * self.ratio + res_boundwise[1] * (1.0 - self.ratio)

        if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
            return res, res_boundwise
        else:
            return res

# class ModifiedWeightedHausdorffDistanceDoubleBoundLoss(Function):
#
#     def __init__(self, return_2_terms=False, alpha=4, beta=1, ratio=0.5, thres=0.5):
#         """ modified whd loss for inner and outer bound separately in which
#             d(x, y) in term1 is modified as exp{d(x, y)} - 1
#             inner bound -- 1, outer bound -- 2
#         :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
#         :param resized_height: int, height after resize
#         :param resized_width: int, width after resize
#         :param alpha: int, decay factor
#         :param ratio: float, ratio of inner bound, default is 0.5
#         """
#         self.return_2_terms = return_2_terms
#         self.alpha = alpha
#         self.beta = beta
#         self.ratio = ratio
#         self.thres = thres
#
#     def __call__(self, prob_map, gt):
#         """ Compute the Weighted Hausdorff Distance function between the estimated probability map
#         and ground truth points. The output is the WHD averaged through all the batch.
#         :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
#         :param gt: (B x H x W) Tensor of the GT annotation
#         """
#         eps = 1e-6
#         alpha = self.alpha
#         beta = self.beta
#         _assert_no_grad(gt)
#
#         # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
#         # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
#         if prob_map.dim() == 5:  # 3D volume
#             prob_map = prob_map.permute(0, 2, 1, 3, 4)
#             prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:])  # combine first 2 dims
#             gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims
#
#         batch_size, n_channel, height, width = prob_map.size()
#         assert batch_size == len(gt), 'prob map and GT must have the same size'
#
#         max_dist = math.sqrt(height ** 2 + width ** 2)
#         n_pixels = height * width
#         all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
#         all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2
#
#         terms_1 = []
#         terms_2 = []
#         for b in range(batch_size):
#             prob_map_b, gt_b = prob_map[b], gt[b]
#             for bound_inx in range(1, n_channel):
#                 gt_bb = (gt_b == bound_inx)  # for different bounds (1 - inner, 2 - outer)
#                 gt_pts = torch.nonzero(gt_bb).float()  # N, 2
#                 n_gt_pts = gt_pts.size(0)
#                 prob_map_bb = prob_map_b[bound_inx]
#
#                 # filter out estimated probs those are lower than threshold
#                 p = prob_map_bb.view(prob_map_bb.numel())
#                 p_sel = p[p >= self.thres]
#                 all_img_locations_sel = all_img_locations[p >= self.thres]
#
#                 if n_gt_pts > 0 and len(all_img_locations_sel) > 0:
#                     d_matrix = cdist(all_img_locations_sel, gt_pts)
#                     n_est_pts = (p_sel ** beta).sum()
#                     p_replicated = p_sel.view(-1, 1).repeat(1, n_gt_pts)
#
#                     term_1 = (1 / (n_est_pts + eps)) * torch.sum(p_sel ** beta * torch.min(d_matrix, 1)[0])
#                     d_div_p = torch.min((d_matrix + eps) /
#                                         (p_replicated ** alpha + eps / max_dist), 0)[0]
#                     d_div_p = torch.clamp(d_div_p, 0, max_dist)
#                     term_2 = torch.mean(d_div_p)
#                     # set different ratio for inner and outer bound
#                     ratio = 2 * self.ratio if bound_inx == 1 else 2 * (1.0 - self.ratio)
#
#                     terms_1.append(ratio * term_1)
#                     terms_2.append(ratio * term_2)
#
#         terms_1 = torch.stack(terms_1)
#         terms_2 = torch.stack(terms_2)
#
#         if self.return_2_terms:
#             res = terms_1.mean(), terms_2.mean()
#         else:
#             res = terms_1.mean() + terms_2.mean()
#
#         return res


class ModifiedWeightedHausdorffDistanceLoss(Function):
    def __init__(self, return_2_terms=False, alpha=4, thres=0.5):
        """ modified weighted Hausdorff Distance based on pixels whose prob are higher than pre-set threshold
            in experiment, this modification doesn't work well
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        """
        self.return_2_terms = return_2_terms
        self.alpha = alpha
        self.thres = thres

    def __call__(self, prob_map, gt):
        """ Compute modified Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
        :param gt: (B x H x W) Tensor of the GT annotation
        """

        eps = 1e-6
        alpha = self.alpha
        _assert_no_grad(gt)
        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        batch_size, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            gt_pts = torch.nonzero(gt_b).float() # N, 2
            n_gt_pts = gt_pts.size()[0]

            if n_gt_pts > 0:
                p = prob_map_b.view(prob_map_b.numel())
                d_matrix = cdist(all_img_locations, gt_pts)
                mask = (p <= self.thres)
                px = p.clone()
                px[mask] = 0.0 # filter out pixels whose prob are lower than pre-set threshold

                n_est_pts = px.sum()
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                # term 1
                term_1 = (1 / (n_est_pts + eps)) * torch.sum(px * torch.min(d_matrix, 1)[0])
                d_div_p = torch.min((d_matrix + eps) /
                                    (p_replicated**alpha + eps / max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, max_dist)
                term_2 = torch.mean(d_div_p)

                terms_1.append(term_1)
                terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res

def grad_check():
    """ check whether the implementation is correct or not """
    # prepare weight and output
    weight = Variable(torch.ones(5))
    output = torch.rand(20, 5, 96, 96).float()
    output = F.softmax(output, 1)
    output1 = Variable(output, requires_grad=True)
    # output2 = Variable(output, requires_grad=True)

    # prepare target
    # encoded_target = output.data.clone().zero_()
    target = torch.randint(0, 5, (20, 96, 96)).long()
    encoded_target = target
    # encoded_target.scatter_(1, target.unsqueeze(1), 1)

    # encoded_target = torch.rand(2, 5, 10, 10).float()
    encoded_target = Variable(encoded_target, requires_grad=False)
    # print(target)

    gdl = GeneralizedDiceLoss()
    g2 = gdl.cal_loss(output1, encoded_target)
    loss = gdl(output1, encoded_target)
    loss.backward()
    g1 = output1.grad

    # g2 = output2.grad
    diff = torch.abs(g1 - g2).sum()
    g1_sum = torch.abs(g1).sum()
    print(g1_sum)
    print("grad difference: {}".format(diff.data.item()))
    # # kl_div = nn.KLDivLoss()(output1, encoded_target)
    # print(my_kl_div)
    # print(kl_div)
    #
    # kl_div.backward()
    # g1 = output1.grad
    # my_kl_div.backward()

def check_bound_weight():
    """ check whether boundary weight is correctly calculated or not """

    w1, w2 = 50.0, 10.0
    sigma1, sigma2 = 1.0, 1.0
    img_dir = "/data/ugui0/antonio-t/CPR_multiview_interp2/S218801d0c_S2052ee2457ad29_20160809" \
              "/I10/applicate/mask/038.tiff"
    image = io.imread(img_dir)
    image = image[176:336, 176:336]

    images = np.tile(image, (10, 1, 1))
    bounds = [gray2bounds(image, width=2) for image in images]
    bounds = np.stack(bounds)
    target = torch.from_numpy(bounds).long().cuda()

    # prepare prob map
    prob = torch.rand(10, 160, 160).float().cuda()

    whd_loss = WeightedHausdorffDistanceLoss(resized_height=96, resized_width=96)
    loss = whd_loss(prob, target)
    print("loss = {}".format(loss.item()))
    # weights = bound_weight_withdiff(target, sigmas=[sigma1, sigma2], ws=[w1, w2], n_classes=3, bound_output=True)
    # plt.figure()
    # plt.imshow(weights[0].cpu().numpy(), cmap='seismic')
    # plt.savefig("./bound_weight/debug_{:.1f}_{:.1f}_{:.1f}_{:.1f}.pdf".format(w1, w2, sigma1, sigma2))

if __name__ == "__main__":
    # img_dir = "../mask/038.tiff"
    # image = io.imread(img_dir)
    # image[image == 76] = 255
    # image[image == 151] = 255
    # image = gray2mask(image)
    # image = image[176:336, 176:336]
    # plt.figure()
    # plt.imshow(image)
    # plt.savefig("input.png")
    # # plt.show()
    # image = np.tile(image, (128, 1, 1))
    # target = torch.from_numpy(image).long().cuda()
    #
    # x = torch.FloatTensor(128, 3, 160, 160).float().cuda()
    #
    # weight = torch.from_numpy(np.load('nlf_weight_all_3.npy')).cuda().float()
    # loss = CrossEntropyBoundLoss(n_classes=3, weight=weight)(x, target)
    # print(loss.data.item())

    #
    # x = torch.rand(10, 20, 20).float()
    # y = x.repeat(1, 5, 1, 1)
    # print(y.size())

    check_bound_weight()