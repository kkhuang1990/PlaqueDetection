# _*_ coding: utf-8 _*_ 

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from utils import mask2rgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from metric import cal_f_score
import numpy as np
import _pickle as pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
# import torch
import os
import os.path as osp
import itertools
from sklearn.metrics import confusion_matrix
from utils import gray2rgb, gray2rgb, mask2outerbound, mask2innerouterbound
from sklearn.metrics import auc
from sklearn.metrics.pairwise import pairwise_distances
from metric import slicewise_hd95
import matplotlib.animation as animation
# for customed colormap
from matplotlib import cm
from matplotlib.colors import ListedColormap

def sample_stack(stack, rows=10, cols=10, start_with=0, show_every=2, scale=4, fig_name = None):
    """ show stacked image samples
    Args:
        stack: numpy ndarray, input stack to plot
    """
    _, ax = plt.subplots(rows, cols, figsize=[scale*cols, scale*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind < len(stack):
            ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
            ax[int(i/cols),int(i % cols)].imshow(stack[ind], cmap=plt.cm.gray)
            ax[int(i/cols),int(i % cols)].axis('off')

    if fig_name:
        plt.savefig(fig_name+'.png')
    plt.close()
    # plt.show()

def sample_stack_color(stack, metrics, rows=10, cols=10, start_with=0, show_every=2, scale=4, fig_name = None):
    """ show stacked image samples
    Args:
        stack: numpy ndarray, input stack to plot
    """
    _, ax = plt.subplots(rows, cols, figsize=[scale*cols, scale*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind < len(stack):
            ax[int(i/cols),int(i % cols)].set_title('F1= {:.4f}'.format(metrics[ind]))
            ax[int(i/cols),int(i % cols)].imshow(gray2rgb(stack[ind]))
            ax[int(i/cols),int(i % cols)].axis('off')

    if fig_name:
        plt.savefig(fig_name+'.png')
    plt.close()


def sample_dict_data(data_list, rows=15, start_with=0, show_every=2, scale=3, label=False, fig_name=None):
    """ show sample of a list of data
    Args:
        data_list: list, list of data in which each element is a dictionary
    """
    keys = data_list[0].keys()
    cols = len(keys)

    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])
    for i in range(rows):
        ind = start_with + i * show_every
        if ind < len(data_list):
            for j in range(cols):
                ax[i, j].imshow(data_list[ind][keys[j]], cmap='gray')
                ax[i, j].axis('off')
                if label:
                    ax[i, j].set_title("s{} {}".format(ind, keys[j]))
    if fig_name:
        plt.savefig(fig_name + '.png')
    # plt.close()
    plt.show()

def plot_metrics(metrics, labels, fig_dir):
    """ plot experiment metric results
    Args:
        metrics: list, with each element as measures for train and test
        labels: list, label for each metric
        file_name: str, where to save the plotted figure
    """

    if fig_dir:
        if not osp.exists(fig_dir):
            os.makedirs(fig_dir)

    for metric, label in zip(metrics, labels):
        plt.figure()
        for phase, value in metric.items():
            plt.plot(value, label=phase)
        plt.title("{} at different epoches".format(label))
        plt.xlabel("epoch")
        plt.ylabel(label)
        plt.legend()
        plt.savefig("./{}/{}.png".format(fig_dir, label))
        plt.close()

def plot_class_f1(metrics, fig_dir):
    """ plot class-wise f1
    :param metrics: dict, dict of f1 scores for different epochs
    :param fig_dir: str, to where to plot figure
    """
    keys = ['background', 'central part', 'outline', 'cal plaque', 'non-cal plaque']
    for phase, value in metrics.items():
        plt.figure()
        f1s = [[item[i] for item in value] for i in range(len(value[0]))]
        for key, f1 in zip(keys, f1s):
            plt.plot(f1, label=key)

        plt.title("class-wise F1-score at different epoches")
        plt.xlabel("epoch")
        plt.ylabel("F1-score")
        plt.legend()
        plt.savefig("./{}/class-wise_F1_{}.png".format(fig_dir, phase))
        plt.close()

def sample_list2(data_list, rows=15, start_with=0, show_every=2, scale=4, fig_name=None, start_inx=0, n_class=5):
    """ show sample of a list of data
        this function is mainly for plotting outputs, predictions as well as average F1 scores
    Args:
        data_list: list, list of data in which each element is a dictionary
        start_inx: int, starting slice index for current figure
    """
    input_cols = len(data_list[0]['input'])
    if input_cols == 1:
        input_names = ['input']
    elif input_cols == 2:
        input_names = ['input(cal)', 'input(non-cal)']
    cols = 4 +  input_cols - 1
    # n_class = data_list[0]['pred'].shape[0]
    n_batch = len(data_list)
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        # read data and calculate average precision
        input = data_list[ind]['input']
        # output = data_list[ind]['output']
        label = data_list[ind]['GT']
        pred = data_list[ind]['pred']

        # calculate F score and average precision
        # output = np.transpose(output, (1, 2, 0))
        # output = np.reshape(output, (-1, n_class))
        label_binary = label_binarize(label.flatten(), classes=range(n_class))
        pred_binary = label_binarize(pred.flatten(), classes=range(n_class))

        f_score = np.zeros(n_class, dtype=np.float32)
        slice_effect_class = 0
        for i in range(n_class):
            if np.sum(label_binary[:,i]) == 0:
                    f_score[i] = 0.0
            else:
                slice_effect_class += 1
                f_score[i] = f1_score(label_binary[:,i], pred_binary[:,i])

        ave_f_score = np.sum(f_score)/slice_effect_class

        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                for col in range(input_cols):
                    ax[i, col].imshow(input[col], cmap='gray')
                    ax[i, col].set_title("Slice {} : {}".format(ind+start_inx, input_names[col]))
                    ax[i, col].axis('off')

                ax[i, input_cols].imshow(mask2rgb(label))
                ax[i, input_cols].set_title('Slice %d : %s' % (ind+start_inx, 'ground truth'))
                ax[i, input_cols].axis('off')

                ax[i, input_cols+1].imshow(mask2rgb(pred))
                ax[i, input_cols+1].set_title('Slice %d : %s' % (ind+start_inx, 'prediction'))
                ax[i, input_cols+1].axis('off')

                ax[i, input_cols+2].scatter(range(0,n_class), f_score)
                ax[i, input_cols+2].set_title('Slice %d : Ave F-score = %0.2f' % (ind+start_inx, ave_f_score))
                ax[i, input_cols+2].set_ylabel('F score')
                ax[i, input_cols+2].set_ylim([-0.1, 1.1])

    # plt.show()
    if fig_name:
        plt.savefig(fig_name + '.png')
    plt.close()


def sample_list_hdf(data_list, rows=15, start_with=0, show_every=2, scale=4, fig_name=None, start_inx=0, n_class=5):
    """ show results as a list with Hausdorff distance calculated from each slice
    Args:
        data_list: list, list of data in which each element is a dictionary
        start_inx: int, starting slice index for current figure
    """

    output_cols = len(data_list[0]['output']) # whether single or multiple channels
    cols = 5 + output_cols - 1

    n_batch = len(data_list)
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        input = data_list[ind]['input']
        label = data_list[ind]['GT']
        pred = data_list[ind]['pred']
        output = data_list[ind]['output'] # [C, H, W]

        hdf = slicewise_hd95(pred, label, n_class)

        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                ax[i, 0].imshow(input, cmap='gray') # we don't consider multiple inputs here
                ax[i, 0].set_title("Slice {} : {}".format(ind+start_inx, 'input'))
                ax[i, 0].axis('off')

                ax[i, 1].imshow(mask2rgb(label))
                ax[i, 1].set_title('Slice %d : %s' % (ind+start_inx, 'ground truth'))
                ax[i, 1].axis('off')

                ax[i, 2].imshow(mask2rgb(pred))
                ax[i, 2].set_title("Slice {:d} : prediction (hdf={:.4f})".format(ind+start_inx, hdf))
                ax[i, 2].axis('off')

                # plot overlapping between pred ang GT annotation
                overlap = pred.copy()
                overlap[label != 0] = 4
                ax[i, 3].imshow(mask2rgb(overlap))
                ax[i, 3].set_title("Slice {:d} : {}".format(ind + start_inx, 'overlap of GT and pred'))
                ax[i, 3].axis('off')

                # plot prob map for different channels
                # if more than 3 channels, plot all channels which are not equal to 0
                output_title = ['prob map (inner bound)', 'prob map (outer bound)'] if output_cols >= 3 else ['prob map']
                for c_inx in range(1, output_cols):
                    ax[i, 3 + c_inx].imshow(output[c_inx], cmap='seismic')
                    ax[i, 3 + c_inx].set_title("Slice {:d} : {}".format(ind + start_inx, output_title[c_inx-1]))
                    ax[i, 3 + c_inx].axis('off')

    # plt.show()
    if fig_name:
        plt.savefig(fig_name + '.pdf')
    plt.close()

def sample_seg_with_hfd(data_list, rows=15, start_with=0, show_every=2, scale=4, fig_name=None, start_inx=0,
                        n_class=5, width=1):
    """ show segmentation result with bound and corresponding hdf calculated
        plot input, annotation, prediction, bounds and F1 scores
    :param data_list: list, list of data in which each element is a dictionary
    :param start_inx: int, starting slice index for current figure """

    cols = 5
    n_batch = len(data_list)
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        input = data_list[ind]['input']
        # print("input shape: {}".format(input.shape))
        label = data_list[ind]['GT']
        pred = data_list[ind]['pred']

        # calculate average F1 score
        label_binary = label_binarize(label.flatten(), classes=range(n_class))
        pred_binary = label_binarize(pred.flatten(), classes=range(n_class))

        f_score = np.zeros(n_class, dtype=np.float32)
        slice_effect_class = 0
        for i in range(n_class):
            if np.sum(label_binary[:,i]) == 0:
                    f_score[i] = 0.0
            else:
                slice_effect_class += 1
                f_score[i] = f1_score(label_binary[:,i], pred_binary[:,i])

        ave_f_score = np.sum(f_score)/slice_effect_class

        # calculate HDF between pred and GT bound
        label_bound = mask2innerouterbound(label, width=width)
        pred_bound = mask2innerouterbound(pred, width=width)
        hdf = slicewise_hd95(pred_bound, label_bound, n_class)

        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                ax[i, 0].imshow(input, cmap='gray')
                ax[i, 0].set_title("Slice {} : {}".format(ind+start_inx, 'input'))
                ax[i, 0].axis('off')

                ax[i, 1].imshow(mask2rgb(label))
                ax[i, 1].set_title('Slice %d : %s' % (ind+start_inx, 'ground truth'))
                ax[i, 1].axis('off')

                ax[i, 2].imshow(mask2rgb(pred))
                ax[i, 2].set_title('Slice %d : %s' % (ind+start_inx, 'prediction'))
                ax[i, 2].axis('off')

                # print("# of non-cal pixels in label: {}, in pred: {}".format(np.sum(label == 4), np.sum(pred == 4)))
                # plot overlapping between pred_bound and label_bound
                overlap = pred_bound.copy()
                overlap[label_bound != 0] = 4
                ax[i, 3].imshow(mask2rgb(overlap))
                ax[i, 3].set_title("Slice {:d} : bound hdf={:.4f}".format(ind + start_inx, hdf))
                ax[i, 3].axis('off')

                ax[i, 4].scatter(range(0, n_class), f_score)
                ax[i, 4].set_title('Slice %d : Ave F-score = %0.2f' % (ind+start_inx, ave_f_score))
                ax[i, 4].set_ylabel('F score')
                ax[i, 4].set_ylim([-0.1, 1.1])

    if fig_name:
        plt.savefig(fig_name + '.pdf')
    plt.close()

#####################################################################################
##  input | GT seg | pred seg | bound (overlap with GT bound) | inner probmap | outer probmap | F1 score
#####################################################################################
def sample_wnet(data_list, rows=15, start_with=0, show_every=2, scale=4, fig_name=None, start_inx=0,
                        n_class=5, width=1):
    """ show segmentation result with bound and corresponding hdf calculated
        plot input, annotation, prediction, bounds and F1 scores
    :param data_list: list, list of data in which each element is a dictionary
    :param start_inx: int, starting slice index for current figure """

    n_probmaps = data_list[0]['bound'].shape[0]  # number of bounds
    cols = 5 + n_probmaps - 1
    n_batch = len(data_list)
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        input = data_list[ind]['input']
        # print("input shape: {}".format(input.shape))
        label = data_list[ind]['GT']
        pred = data_list[ind]['pred']
        bound_probmap = data_list[ind]['bound'] # predicted bound probmap

        # calculate average F1 score
        label_binary = label_binarize(label.flatten(), classes=range(n_class))
        pred_binary = label_binarize(pred.flatten(), classes=range(n_class))

        f_score = np.zeros(n_class, dtype=np.float32)
        slice_effect_class = 0
        for i in range(n_class):
            if np.sum(label_binary[:,i]) == 0:
                    f_score[i] = 0.0
            else:
                slice_effect_class += 1
                f_score[i] = f1_score(label_binary[:,i], pred_binary[:,i])

        ave_f_score = np.sum(f_score)/slice_effect_class

        # calculate average HFD
        label_bound = mask2innerouterbound(label, width=width)
        pred_bound = mask2innerouterbound(pred, width=width)
        hdf = slicewise_hd95(pred_bound, label_bound, n_class)

        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                ax[i, 0].imshow(input, cmap='gray')
                ax[i, 0].set_title("Slice {} : {}".format(ind+start_inx, 'input'))
                ax[i, 0].axis('off')

                ax[i, 1].imshow(mask2rgb(label))
                ax[i, 1].set_title('Slice %d : %s' % (ind+start_inx, 'ground truth'))
                ax[i, 1].axis('off')

                ax[i, 2].imshow(mask2rgb(pred))
                ax[i, 2].set_title('Slice %d : %s' % (ind+start_inx, 'prediction'))
                ax[i, 2].axis('off')

                # plot overlapping between pred_bound and label_bound
                overlap = pred_bound.copy()
                overlap[label_bound != 0] = 4
                ax[i, 3].imshow(mask2rgb(overlap))
                ax[i, 3].set_title("Slice {:d} : bound hdf={:.4f}".format(ind + start_inx, hdf))
                ax[i, 3].axis('off')

                # plot prob maps for intermediate bounds
                output_title = ['prob map (inner bound)', 'prob map (outer bound)'] if n_probmaps >= 3 else ['prob map']
                for c_inx in range(1, n_probmaps):
                    ax[i, 3 + c_inx].imshow(bound_probmap[c_inx], cmap='seismic')
                    ax[i, 3 + c_inx].set_title("Slice {:d} : {}".format(ind + start_inx, output_title[c_inx - 1]))
                    ax[i, 3 + c_inx].axis('off')

                ax[i, 3 + n_probmaps].scatter(range(0, n_class), f_score)
                ax[i, 3 + n_probmaps].set_title('Slice %d : Ave F-score = %0.2f' % (ind+start_inx, ave_f_score))
                ax[i, 3 + n_probmaps].set_ylabel('F score')
                ax[i, 3 + n_probmaps].set_ylim([-0.1, 1.1])

    if fig_name:
        plt.savefig(fig_name + '.pdf')
    plt.close()


def plaque_detection_rate(labels, preds, n_classes=5, thres=0):
    """ calculate calcified and non-calcified plaque detection accuracy
        as well as slice-wise recall/precision/F1
    """
    if not isinstance(labels, np.ndarray):
        labels = labels.data.cpu().numpy()
        preds = preds.cpu().numpy()

    pgt_cnt = [0, 0]
    pp_cnt = [0, 0]
    tp_cnt = [0, 0]

    # 5-class segmentation or 2-class segmentation
    plaques = [3, 4] if n_classes == 5 else [0, 1]
    if labels.ndim == 3: # for 2D image
        for label, pred in zip(labels, preds):
            for inx, plaque in enumerate(plaques):
                if np.sum(label == plaque) != 0:
                    pgt_cnt[inx] += 1
                if np.sum(pred == plaque) >= thres:
                    pp_cnt[inx] += 1
                if np.sum(label == plaque) != 0 and np.sum(pred == plaque) >= thres:
                    tp_cnt[inx] += 1

    elif labels.ndim == 4:
        for label_vol, pred_vol in zip(labels, preds):
            for label, pred in zip(label_vol, pred_vol):
                for inx, plaque in enumerate(plaques):
                    if np.sum(label == plaque) != 0:
                        pgt_cnt[inx] += 1
                    if np.sum(pred == plaque) >= thres:
                        pp_cnt[inx] += 1
                    if np.sum(label == plaque) != 0 and np.sum(pred == plaque) >= thres:
                        tp_cnt[inx] += 1

    return pgt_cnt[0], pp_cnt[0], tp_cnt[0], pgt_cnt[1], pp_cnt[1], tp_cnt[1]


def plot_risk_confusion_matrix(y_test, y_pred, root_fig_path):

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title.split('/')[-1])
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if title:
            plt.savefig(title+'.png')

        plt.close()

    if not osp.exists(root_fig_path):
        os.makedirs(root_fig_path)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    class_names = ['1', '2', '3', '4']

    # Plot non-normalized confusion matrix
    plt.figure()
    title = root_fig_path + '/' + 'Confusion_matrix_without_normalization'
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=title)


    # Plot normalized confusion matrix
    plt.figure()
    title = root_fig_path + '/' + 'Normalized_confusion_matrix'
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title=title)

# sample data for assignment from Harada sensei
def sample_list3(data_list, rows=15, cols=4, start_with=0, show_every=2, scale=4, fig_name=None, start_inx=0):
    """ show sample of a list of data
        here we plot slice, label, hu0050, overlap
        this function is mainly for plotting outputs, predictions as well as average F1 scores
    Args:
        data_list: list, list of data in which each element is a dictionary
        start_inx: int, starting slice index for current figure
    """

    n_batch = len(data_list)
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        # read data and calculate average precision
        input1 = data_list[ind]['slice1']
        input2 = data_list[ind]['slice2']
        label = data_list[ind]['label']
        hu0050 = data_list[ind]['hu0050']
        overlap = data_list[ind]['overlap']
        f_score = data_list[ind]['f1']
        mix_overlap = data_list[ind]['mix_overlap']
        noncal_eval = data_list[ind]['noncal_eval']
        file_path = data_list[ind]['file_path']
        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                ax[i, 0].imshow(input1, cmap='gray')
                ax[i, 0].set_title("Slice {} ({}) \n {}".format(ind + start_inx, file_path, 'Input with HU(-100~155)'), loc='left')
                ax[i, 0].axis('off')

                ax[i, 1].imshow(input2, cmap='gray')
                ax[i, 1].set_title("{}".format('Input with HU(200~1200)'))
                ax[i, 1].axis('off')

                ax[i, 2].imshow(gray2rgb(label))
                ax[i, 2].set_title('{}'.format('Label'))
                ax[i, 2].axis('off')

                ax[i, 3].imshow(gray2rgb(hu0050))
                ax[i, 3].set_title('{}'.format('Mask HU(0~50)'))
                ax[i, 3].axis('off')

                ax[i, 4].imshow(gray2rgb(overlap))
                ax[i, 4].set_title('{} (F1= {:.4f})'.format('Overlap', f_score))
                ax[i, 4].axis('off')

                # not all red pixels are within HU range 0~50

                if(np.sum(overlap == 76)) != 0:
                    n_above50, n_below0, topk, buttomk = noncal_eval[0], noncal_eval[1], noncal_eval[2:7], noncal_eval[7:12]
                    ax[i, 4].text(5, 30, "top5 HU: {}".format(topk), color='red')
                    ax[i, 4].text(5, 60, "but5 HU: {}".format(buttomk), color='red')
                    ax[i, 4].text(5, 90, "Num of pixels HU>50: {}".format(n_above50), color='red')
                    ax[i, 4].text(5, 120, "Num of pixels HU<0: {}".format(n_below0), color='red')

                ax[i, 5].imshow(gray2rgb(mix_overlap))
                ax[i, 5].set_title('{} (F1= {:.4f})'.format('Label+Overlap', f_score))
                ax[i, 5].axis('off')

                # ax[i, 3].scatter(range(0, n_class), f_score)
                # ax[i, 3].set_title('Slice %d : Ave F-score = %0.2f' % (ind + start_inx, ave_f_score))
                # ax[i, 3].set_ylabel('F score')
                # ax[i, 3].set_ylim([-0.1, 1.1])

    # plt.show()
    if fig_name:
        plt.savefig(fig_name + '.pdf')
    plt.close()


def plot_slice_wise_measures(labels, preds, args):
    """ In test phase, plot various measures such as ROC, AUC, PR, RC, F1 et al """

    cal_roc = [[], []]
    cal_prrcf1 = [[], [], []]  # save PR, RC, F1 respectively
    noncal_prrcf1 = [[], [], []]
    thres_all = []
    noncal_roc = [[], []]
    n_slices = len(labels)
    for thres in range(500, -1, -5):
        print("[Threshold # of pixels: {}]".format(thres))
        thres_all.append(thres)
        cal_pgt, cal_pp, cal_tp, noncal_pgt, noncal_pp, noncal_tp = \
            plaque_detection_rate(labels, preds, thres=thres)


        cal_prrcf1[0].append(float(cal_tp) / cal_pp if cal_pp != 0 else 0.0)
        cal_prrcf1[1].append(float(cal_tp) / cal_pgt)
        cal_prrcf1[2].append(2.0 * cal_tp / (cal_pgt + cal_pp))
        noncal_prrcf1[0].append(float(noncal_tp) / noncal_pp if noncal_pp != 0 else 0.0)
        noncal_prrcf1[1].append(float(noncal_tp) / noncal_pgt)
        noncal_prrcf1[2].append(2.0 * noncal_tp / (noncal_pgt + noncal_pp))

        cal_roc[0].append((cal_pp - cal_tp) / (n_slices - cal_pgt))  # false negative ratio
        cal_roc[1].append(cal_tp / cal_pgt)  # true positive ratio
        noncal_roc[0].append((noncal_pp - noncal_tp) / (n_slices - noncal_pgt))  # false negative ratio
        noncal_roc[1].append(noncal_tp / noncal_pgt)  # true positive ratio

        print('Cal: PR - {:.4f} RC - {:.4f} F1 - {:.4f} Noncal: PR - {:.4f} RC - {:.4f} F1 - {:.4f}'.format(
            cal_prrcf1[0][-1], cal_prrcf1[1][-1], cal_prrcf1[2][-1],
            noncal_prrcf1[0][-1], noncal_prrcf1[1][-1], noncal_prrcf1[2][-1]))
        print('Cal: fpr - {:.4f} tpr - {:.4f} Noncal: fpr - {:.4f} tpr - {:.4f}'.format(
            cal_roc[0][-1], cal_roc[1][-1], noncal_roc[0][-1], noncal_roc[1][-1]))

    # plot the roc curve and calculate AUC
    fig_names = ['calcified', 'non-calcified']
    for plq_metrics, fig_name in zip([cal_roc, noncal_roc], fig_names):
        plt.figure()
        lw = 2
        auc_metric = auc(plq_metrics[0], plq_metrics[1])
        print("{} : {}".format(fig_name, auc_metric))
        plt.plot(plq_metrics[0], plq_metrics[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc_metric)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('slice-wise ROC curve of {} plaques'.format(fig_name))
        plt.legend(loc="lower right")
        plt.savefig("./{}/{}_roc.png".format(args.fig_dir, fig_name))

    for plq_metrics, fig_name in zip([cal_prrcf1, noncal_prrcf1], fig_names):
        plt.figure()
        lw = 2
        plt.plot(thres_all, plq_metrics[0], color='r', lw=lw, label='precision')
        plt.plot(thres_all, plq_metrics[1], color='g', lw=lw, label='recall')
        plt.plot(thres_all, plq_metrics[2], color='b', lw=lw, label='f1')

        plt.xlim([min(thres_all), max(thres_all)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold Number of Pixels')
        plt.title('{} measures under different thresholds'.format(fig_name))
        plt.legend(bbox_to_anchor=(1, 0.95), loc="upper right")
        plt.savefig("./{}/{}_prrcf1.png".format(args.fig_dir, fig_name))

def plot_seg_bound_comparison(data_list, rows, start_with, show_every, start_inx, n_class, fig_name=None, width=2, scale=4):
    """ plot result comparison between seg and bound detection """
    cols = 6  # [input, label_seg, label_bound, pred_bound(converted), pred_bound_2d, pred_bound_3d]
    n_batch = len(data_list)
    # print("number of slices: {}".format(n_batch))
    _, ax = plt.subplots(rows, cols, figsize=[scale * cols, scale * rows])

    for ind in range(n_batch):
        input = data_list[ind]['input']
        label_seg = data_list[ind]['GT_seg']
        pred_seg = data_list[ind]['pred_seg']  # seg prediction is not plotted here
        pred_bound_conv = mask2outerbound(pred_seg, width=width)  # convert seg to inner-outer bound
        label_bound = data_list[ind]['GT_bound']
        pred_bound_2d = data_list[ind]['pred_2d_bound']
        pred_bound_3d = data_list[ind]['pred_3d_bound']
        # print("input: {}, seg: {}, pred_seg: {}, label_bound: {}, pred_bound_2d: {}, pred_bound_3d: {}".format(input.shape,
        #         label_seg.shape, pred_seg.shape, label_bound.shape, pred_bound_2d.shape, pred_bound_3d.shape))
        # print()

        # # calculate average F1 score
        # label_binary = label_binarize(label_seg.flatten(), classes=range(n_class))
        # pred_binary = label_binarize(pred_seg.flatten(), classes=range(n_class))
        #
        # f_score = np.zeros(n_class, dtype=np.float32)
        # slice_effect_class = 0
        # for i in range(n_class):
        #     if np.sum(label_binary[:,i]) == 0:
        #             f_score[i] = 0.0
        #     else:
        #         slice_effect_class += 1
        #         f_score[i] = f1_score(label_binary[:,i], pred_binary[:,i])
        #
        # ave_f_score = np.sum(f_score) / slice_effect_class

        # calculate average HFD
        hdf_seg = slicewise_hd95(pred_bound_conv, label_bound, n_class)
        hdf_bound_2d = slicewise_hd95(pred_bound_2d, label_bound, n_class)
        hdf_bound_3d = slicewise_hd95(pred_bound_3d, label_bound, n_class)

        if (ind - start_with) % show_every == 0:
            i = (ind - start_with) // show_every
            if i < rows:
                ax[i, 0].imshow(input, cmap='gray')
                ax[i, 0].set_title("Slice {} : {}".format(ind+start_inx, 'input'))
                ax[i, 0].axis('off')

                ax[i, 1].imshow(mask2rgb(label_seg))
                ax[i, 1].set_title('Slice %d : %s' % (ind+start_inx, 'label_seg'))
                ax[i, 1].axis('off')

                label_bound_cp = label_bound.copy()
                label_bound_cp[label_bound != 0] = 4

                ax[i, 2].imshow(mask2rgb(label_bound_cp))
                ax[i, 2].set_title('Slice %d : %s' % (ind + start_inx, 'label_bound'))
                ax[i, 2].axis('off')

                # plot overlapping between pred_bound_conv and label_bound
                overlap_seg = pred_bound_conv.copy()
                overlap_seg[label_bound != 0] = 4

                ax[i, 3].imshow(mask2rgb(overlap_seg))
                ax[i, 3].set_title("Slice {:d} : bound from seg (hdf={:.4f})".format(ind + start_inx, hdf_seg))
                ax[i, 3].axis('off')

                overlap_bound_2d = pred_bound_2d.copy()
                overlap_bound_2d[label_bound != 0] = 4
                ax[i, 4].imshow(mask2rgb(overlap_bound_2d))
                ax[i, 4].set_title("Slice {:d} : 2D bound (hdf={:.4f})".format(ind + start_inx, hdf_bound_2d))
                ax[i, 4].axis('off')

                overlap_bound_3d = pred_bound_3d.copy()
                overlap_bound_3d[label_bound != 0] = 4
                ax[i, 5].imshow(mask2rgb(overlap_bound_3d))
                ax[i, 5].set_title("Slice {:d} : 3D bound (hdf={:.4f})".format(ind + start_inx, hdf_bound_3d))
                ax[i, 5].axis('off')

    if fig_name:
        plt.savefig(fig_name + '.pdf')

    plt.close()

def seg_bound_comparison(orig_label_path, seg_data_path, bound_data_2d_path, bound_data_3d_path, fig_save_dir, sample_stack_rows=50):
    """ compare segmentation and bound detection results and plot them into a single graph
    :param seg_data_path: str, segmentation result path
    :param bound_data_path: str, boundary detection result path
    :param fig_save_dir: str, to where to save the result comparison
    """

    for sample in os.listdir(seg_data_path):
        if not sample.startswith('.') and osp.isdir(osp.join(seg_data_path, sample)):
            sample_path = osp.join(seg_data_path, sample)
            for artery in os.listdir(sample_path):
                orig_label_pick_path = osp.join(orig_label_path, sample, artery, 'data.pkl')
                seg_pick_path = osp.join(seg_data_path, sample, artery, 'data.pkl')
                bound_2d_pick_path = osp.join(bound_data_2d_path, sample, artery, 'data.pkl')
                bound_3d_pick_path = osp.join(bound_data_3d_path, sample, artery, 'data.pkl')
                artery_save_dir = osp.join(fig_save_dir, sample, artery)

                if not osp.exists(artery_save_dir):
                    os.makedirs(artery_save_dir)

                # load original segmentation label
                with open(orig_label_pick_path, 'rb') as reader:
                    labels_gt = pickle.load(reader)['label']

                with open(seg_pick_path, 'rb') as reader:
                    data_seg = pickle.load(reader)
                    # inputs_seg here is a list of length 1 (not modified yet)
                    inputs_seg, labels_seg, preds_seg = data_seg['input'], data_seg['label'], data_seg['pred']
                    start, n_class, width = data_seg['start'], data_seg['n_class'], data_seg['width']

                with open(bound_2d_pick_path, 'rb') as reader:
                    data_bound = pickle.load(reader)
                    # inputs_bound here is a list of length 1 (not modified yet)
                    inputs_bound_2d, labels_bound_2d, preds_bound_2d, outputs_bound_2d = \
                        data_bound['input'], data_bound['label'], data_bound['pred'], data_bound['output']

                with open(bound_3d_pick_path, 'rb') as reader:
                    data_bound = pickle.load(reader)
                    # inputs_bound here is a list of length 1 (not modified yet)
                    inputs_bound_3d, labels_bound_3d, preds_bound_3d, outputs_bound_3d = data_bound['input'], \
                                        data_bound['label'], data_bound['pred'], data_bound['output']

                print("# of slices in total: {}".format(len(inputs_seg[0]))) # number of slices

                for inx in range(0, len(inputs_seg[0]), sample_stack_rows):
                    over = min(inx + sample_stack_rows, len(inputs_seg[0]))
                    input_plot, label_gt_plot, label_bound_2d_plot, pred_seg_plot, pred_bound_2d_plot, pred_bound_3d_plot\
                        = inputs_seg[0][inx:over], labels_gt[inx:over], labels_bound_2d[inx:over], preds_seg[inx:over], \
                          preds_bound_2d[inx:over], preds_bound_3d[inx:over]

                    # for result check
                    print("input: {}, label_seg: {}, label_bound_2d: {}, pred_seg: {}, pred_bound_2d: {}, pred_bound_3d: {}".format(
                        input_plot.shape, label_gt_plot.shape, label_bound_2d_plot.shape, pred_seg_plot.shape, pred_bound_2d_plot.shape,
                        pred_bound_3d_plot.shape))

                    data_list = [{"input": input, "GT_seg": label_seg, "pred_seg": pred_seg, "GT_bound": label_bound, "pred_2d_bound": pred_bound_2d,
                                  "pred_3d_bound" : pred_bound_3d} for (input, label_seg, pred_seg, label_bound, pred_bound_2d, pred_bound_3d)
                                 in zip(input_plot, label_gt_plot, pred_seg_plot, label_bound_2d_plot, pred_bound_2d_plot, pred_bound_3d_plot)]

                    # print("# of slices in batch: {}".format(len(data_list)))
                    file_name = "{}/{:03d}".format(artery_save_dir, inx + start)

                    plot_seg_bound_comparison(data_list, rows=over - inx, start_with=0, show_every=1, start_inx=inx + start,
                                              n_class=n_class, fig_name=file_name, width=width, scale=4)


def gif_generation(orig_label_path, bound_data_path):
    """ generate gif animation from slices saved in bound_data_path, plus the original label
    :param orig_label_path: str, from where to read original label
    :param bound_data_path: str, from where to read boundary detection results
    """
    for sample in os.listdir(bound_data_path):
        if not sample.startswith('.') and osp.isdir(osp.join(bound_data_path, sample)):
            sample_path = osp.join(bound_data_path, sample)
            for artery in os.listdir(sample_path):
                orig_label_pick_path = osp.join(orig_label_path, sample, artery, 'data.pkl')
                bound_pick_path = osp.join(bound_data_path, sample, artery, 'data.pkl')

                # function to save result of each artery into gif
                save_gif_artery(orig_label_pick_path, bound_pick_path)

def save_gif_artery(orig_label_pick_path, bound_pick_path):
    """
    :param orig_label_pick_path: str, path of original segmentation label
    :param bound_pick_path: str, path of boundary detection result
    figures are arranged in the order of
            input | GT_seg | GT_bound | pred_bound
            heatmap[0-256] | heatmap[0-100] | inner bound probmap | outer bound probmap
    besides, we only consider heatmap with range of
            0~(70)~256, namely 0.38438 ~ 0.41066 ~ 0.48048 and
            0~(50)~100, namely 0.38438 ~ 0.40315 ~ 0.42192 respectively
    """

    gif_save_dir = '/'.join(bound_pick_path.split('/')[:-1])
    print("Processing {}".format(gif_save_dir))

    # load original segmentation label
    with open(orig_label_pick_path, 'rb') as reader:
        data_seg = pickle.load(reader)
        labels_seg, start_seg = data_seg["label"], data_seg["start"]

    with open(bound_pick_path, 'rb') as reader:
        data_bound = pickle.load(reader)
        inputs_bound, labels_bound, preds_bound, start_bound, probmaps = \
            data_bound['input'], data_bound['label'], data_bound['pred'], data_bound['start'], data_bound['output']

    assert len(inputs_bound) == len(labels_bound) == len(preds_bound), "inputs, GT and preds should have the " \
                                                                       "same number of slices"
    print(len(inputs_bound), len(labels_seg), start_bound, start_seg)

    scale, rows, cols = 4, 2, 4
    fig = plt.figure(figsize=[scale * cols, scale * rows])
    artery_name = '/'.join(gif_save_dir.split('/')[-2:])

    # add subplots for each figure
    ax1 = fig.add_subplot(rows, cols, 1)
    ax2 = fig.add_subplot(rows, cols, 2)
    ax3 = fig.add_subplot(rows, cols, 3)
    ax4 = fig.add_subplot(rows, cols, 4)
    ax5 = fig.add_subplot(rows, cols, 5)
    ax6 = fig.add_subplot(rows, cols, 6)
    ax7 = fig.add_subplot(rows, cols, 7)
    ax8 = fig.add_subplot(rows, cols, 8)

    # create customed colormap
    top = cm.get_cmap('Reds', 186)
    bottom = cm.get_cmap('Blues', 70)
    newcolors = np.vstack((bottom(np.linspace(1, 0, 70)),
                           top(np.linspace(0, 1, 186))))
    bluered = ListedColormap(newcolors, name='BlueReds')

    labels_seg_cal = labels_seg[(start_bound-start_seg):] # seg labels after calibration

    lines = []
    for i in range(len(inputs_bound)):
        input, label_seg, label_bound, pred_bound, probmap = \
            inputs_bound[i], labels_seg_cal[i], labels_bound[i], preds_bound[i], probmaps[i]
        # calculate HDF distance between GT bound and pred bound
        hdf_bound = slicewise_hd95(pred_bound, label_bound, n_classes=3)

        ax1.set_title("{} \n {}".format(artery_name, 'Input'), loc='left')
        ax1.axis('off')
        line1 = ax1.imshow(input, cmap='gray', animated=True)
        line1_text = ax1.text(48, -3, "Slice {}".format(i + start_bound), color='red', fontsize=10)

        ax2.set_title('label_seg')
        ax2.axis('off')
        line2 = ax2.imshow(mask2rgb(label_seg), animated=True)

        ax3.set_title('label_bound')
        ax3.axis('off')
        line3 = ax3.imshow(mask2rgb(label_bound), animated=True)

        ax4.set_title("pred_bound", loc='left')
        ax4.axis('off')
        line4 = ax4.imshow(mask2rgb(pred_bound), animated=True)
        line4_text = ax1.text(400, -3, "Hdf: {:.4f}".format(hdf_bound), color='black', fontsize=10)

        # plot inputs with range [0~256] in colormap
        ax5.set_title("input colormap HU[0~250]")
        ax5.axis('off')
        line5 = ax5.imshow(input, cmap=bluered, vmin=0.38438, vmax=0.48048, animated=True) # crop HU range 0~255

        # plot inputs with range [0~100] in colormap
        ax6.set_title("input colormap HU[0~100]")
        ax6.axis('off')
        line6 = ax6.imshow(input, cmap=bluered, vmin=0.38438, vmax=0.42192, animated=True)  # crop HU range 0~100

        # inner bound probmap
        ax7.set_title("inner bound probmap")
        ax7.axis('off')
        line7 = ax7.imshow(probmap[1], cmap='seismic', animated=True)  # crop HU range 0~100

        # outer bound probmap
        ax8.set_title("outer bound probmap")
        ax8.axis('off')
        line8 = ax8.imshow(probmap[2], cmap='seismic', animated=True)  # crop HU range 0~100

        lines.append([line1, line1_text, line2, line3, line4, line4_text, line5, line6, line7, line8])

    # Build the animation using ArtistAnimation function
    ani = animation.ArtistAnimation(fig, lines, interval=50, blit=True)

    # save into gif and mp4 respectively

    # ani.save('{}/artery.gif'.format(gif_save_dir), writer="imagemagick")
    ani.save('{}/artery.mp4'.format(gif_save_dir), writer="ffmpeg", codec='mpeg4', fps=10)

if __name__ == "__main__":
    # file_name = 'test_result_2.pickle'
    # print("file name: {}".format(file_name))
    # with open(file_name, 'rb') as reader:
    #     data = pickle.load(reader)
    # labels = data['GT']  # [N, H, W]
    # outputs = data['output'] # [N, C, H, W]
    #
    # binary_class_slice_wise_pr(labels, outputs, fig_name= 'test_2_binary_pr')
    # multi_class_slice_wise_pr(labels, outputs, fig_name='test_2_multi_pr_micro')
    # average_precision(labels, outputs)
    # path of original annotation
    orig_label_path = "./PlaqueSegmentation/OrigAnnotation/2d_res_unet_dp_0.001_0.90_0.9_theta-1.0-0.0_100_2_10_dice_Adam_" \
                      "r-True_flip-True_w-True_rcp-True_tr-False_ns-Falseptr-False_mv-False_sl-False_ds-2_a-0.5_lr-StepLR_" \
                      "wt-None_o-5_b-False_cal0gt-False_cf-config_dp-0.0_ig-None_w0-10.0_sg-5.0_96_wt-1_mo_False"

    # seg_data_path = "/home/mil/huang/CPR_Segmentation_ver7/PlaqueSegmentation/Experiment23/2d_res_unet_dp_0.0001_0.90_" \
    #                 "0.9_theta-1.0-0.0_100_100_10_ceb_Adam_r-True_flip-True_w-True_rcp-True_tr-False_ns-Falseptr-False_mv" \
    #                 "-False_sl-False_ds-2_a-0.5_lr-StepLR_wt-None_o-3_b-False_cal0gt-False_cf-config_dp-0.0_ig-None_w0-10.0_" \
    #                 "sg-5.0_96_wt-1_mo_False"
    # bound_data_2d_path = "/home/mil/huang/CPR_Segmentation_ver7/PlaqueBound/Experiment3/2d_res_unet_dp_0.001_0.0_100_100_10" \
    #                   "_whd_Adam_r-True_flip-True_w-False_ptr-False_mv-False_sl-False_lr-StepLR_wt-None_o-2_b-True_cf-config" \
    #                   "_dp-0.0_w1-10.0_w2-10.0_sg1-5.0_sg2-5.0_rs-96_wt-2_bt-outer_whda-4_whdb-1"
    # bound_data_3d_path = "./BoundDetection/Experiment4/3d_res_unet_0.001_100_100_whd_Adam_w-False_sl-True_lr-StepLR_wt-None_o" \
    #                      "-2_b-True_cf-config_dp-0.0_rs-96_cc-192_wt-2_bt-outer_whda-4_whdb-1_whdr-0.5"
    # fig_save_dir = "/home/mil/huang/CPR_Segmentation_ver7/PlaqueDetection_20181127/ResultsComparison/seg_bound_comp_debug3"

    # seg_bound_comparison(orig_label_path, seg_data_path, bound_data_2d_path, bound_data_3d_path, fig_save_dir, sample_stack_rows=50)

    bound_data_path = "./BoundDetection/Experiment7/HybridResUNet_ds1int15_0.167"
    gif_generation(orig_label_path, bound_data_path)