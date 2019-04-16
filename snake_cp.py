# _*_ coding: utf-8 _*_

""" use SNAKE algorithm to actively find the contour of given image """

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import warnings

import _pickle as pickle
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io
from utils import hu2lut, hu2gray, mask2gray, gray2mask, innerouterbound2mask, mask2rgb
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from scipy import ndimage
from multiprocessing import Pool
from metric import volumewise_hd95, volumewise_ahd, slicewise_hd95, slicewise_ahd
from utils import lslist2bound
from skimage import img_as_float
from skimage import measure

warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

import os
import os.path as osp
import cv2 as cv2

###################################################################
#          obtain snake boundary from probability map
##################################################################
def slicewise_process(prob_map_b, n_loops=100, smoothing=1, balloon=-1, thickness=1):
    n_channel, height, width = prob_map_b.shape
    snake = np.zeros((height, width), dtype=np.uint8)

    for bound_inx in range(1, n_channel):  # inner and outer bound
        prob_map_bb = prob_map_b[bound_inx]
        gimage = inverse_gaussian_gradient(prob_map_bb, alpha=300, sigma=3.0)
        init = np.zeros(gimage.shape, dtype=np.int8)
        init[5:-5, 5:-5] = 1

        # np.ndarray with type np.int8
        ls = morphological_geodesic_active_contour(gimage, n_loops, init, smoothing=smoothing,
                                                   balloon=balloon, threshold='auto')
        tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
        mask = np.logical_and(tmp == 1, tmp <= thickness)
        # print("# of snake pixels: {}".format(np.sum(mask)))
        # print("max: {}, min: {}".format(mask.max(), mask.min()))
        snake[mask] = bound_inx

    return snake

def probmap2snake(prob_map, n_workers=32, n_loops = 100, smoothing = 1, balloon=-1, thickness=1):
    """ calculate snake from given probability map as post-processing
    :param prob_map: Tensor of size [B, C, D, H, W], probability map as the Network output
    :return: snake: Tensor of size [B, D, H, W], obtained snake contour
    """

    if prob_map.dim() == 5:  # 3D volume
        prob_map = prob_map.permute(0, 2, 1, 3, 4)
        prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:])  # combine first 2 dims

    batch_size, n_channel, height, width = prob_map.size()
    prob_map = F.softmax(prob_map, 1)
    prob_map_np = prob_map.data.cpu().numpy() #[B', C, H, W]

    args = []
    for b in range(batch_size):
        args.append((prob_map_np[b],n_loops, smoothing, balloon, thickness))

    pool = Pool(processes=n_workers)
    snakes = pool.starmap(slicewise_process, args)
    pool.close()
    snakes = np.stack(snakes)
    snake_cuda = Variable(torch.from_numpy(snakes).cuda()).long()  # with size [B', C, H, W]

    return snake_cuda


def prob2bound_slicewise(prob_map_b, thres=0.7, outer_ks=15, thickness=1):
    """ slice-wise process of probability map
    :param prob_map_b: ndarray of size [B, H, W], probability map
    :param thres: float, thres for filtering out pixels with prob lower than thres
    :param outer_ks: int, kernel size for outer bound detection
    :return: lses: list of obtained bounds
    """

    n_channel, height, width = prob_map_b.shape
    bound = np.zeros((height, width), dtype=np.uint8)

    for bound_inx in range(1, n_channel):  # inner and outer bound
        prob_map_bb = prob_map_b[bound_inx]
        pred = (prob_map_bb >= thres).astype(np.uint8)
        gray = mask2gray(pred)

        # Perform morphology
        if bound_inx == 1:
            se = np.ones((7, 7), dtype='uint8')
        else:
            se = np.ones((outer_ks, outer_ks), dtype='uint8')
        image_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)

        image, contours, hierarchy = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours =[contour for contour in contours
                   if len(contour) > 4 and cv2.contourArea(contour) / len(contour) >= 4.0]

        if len(contours) > 0: # a closed contour can be obtained by cv2.findContours
            mask = np.zeros(image.shape[:2], np.uint8)
            ls = cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
        else: # need use snake to obtain closed contour
            if bound_inx == 1:
                gimage = inverse_gaussian_gradient(prob_map_bb, alpha=100, sigma=5.0)
            else:
                gimage = inverse_gaussian_gradient(prob_map_bb, alpha=100, sigma=3.0)

            init = np.zeros(gimage.shape, dtype=np.int8)
            init[5:-5, 5:-5] = 1
            ls = morphological_geodesic_active_contour(gimage, 100, init, smoothing=1,
                                                       balloon=-1, threshold='auto')

        tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
        flag = np.logical_and(tmp == 1, tmp <= thickness)
        # print("# of snake pixels: {}".format(np.sum(mask)))
        # print("max: {}, min: {}".format(mask.max(), mask.min()))
        bound[flag] = bound_inx

    return bound

def probmap2bound(prob_map, n_workers=32, thres=0.5, outer_ks=15):
    """ calculate snake from given probability map as post-processing
    :param prob_map: Tensor of size [B, C, D, H, W], probability map as the Network output
    :return: snake: Tensor of size [B, D, H, W], obtained snake contour
    """

    if prob_map.dim() == 5:  # 3D volume
        prob_map = prob_map.permute(0, 2, 1, 3, 4)
        prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:])  # combine first 2 dims

    batch_size, n_channel, height, width = prob_map.size()
    prob_map = F.softmax(prob_map, 1)
    prob_map_np = prob_map.data.cpu().numpy() #[B', C, H, W]

    args = []
    for b in range(batch_size):
        args.append((prob_map_np[b],n_loops, smoothing, balloon, thickness))

    pool = Pool(processes=n_workers)
    snakes = pool.starmap(slicewise_process, args)
    pool.close()
    snakes = np.stack(snakes)
    snake_cuda = Variable(torch.from_numpy(snakes).cuda()).long()  # with size [B', C, H, W]

    return snake_cuda


##################################################################
#               obtain snake boundary from pred
##################################################################
def slicewise_process_pred(pred, n_channel=3, n_loops=100, smoothing=1, balloon=1, thickness=1):
    """ use snake to obtain closed boundary from prediction result """
    height, width = pred.shape
    snake = np.zeros((height, width), dtype=np.uint8)

    for bound_inx in range(1, n_channel):  # inner and outer bound
        pred_bb = (pred == bound_inx)
        pred_bb = mask2gray(pred_bb)
        gimage = inverse_gaussian_gradient(pred_bb)
        init = np.zeros(gimage.shape, dtype=np.int8)
        init[5:-5, 5:-5] = 1

        # np.ndarray with type np.int8
        ls = morphological_geodesic_active_contour(gimage, n_loops, init, smoothing=smoothing,
                                                   balloon=balloon, threshold='auto')
        tmp = ndimage.distance_transform_cdt(ls, 'taxicab')

        mask = np.logical_and(tmp == 1, tmp <= thickness)
        # print("# of snake pixels: {}".format(np.sum(mask)))
        # print("max: {}, min: {}".format(mask.max(), mask.min()))
        snake[mask] = bound_inx

    return snake

##################################################################
#       debug code for obtaining snake from probability map
##################################################################
def prob2snake_slicewise_debug(i, prob_map_b, label_b, thres=0.7, do_plot=True):
    """ slice-wise calculation """

    n_channel, height, width = prob_map_b.shape
    lses, fcs, images_close = [], [], []
    iter_max = 10

    # filter
    prob_map_filter = prob_map_b.copy()
    prob_map_filter[prob_map_b <= thres] = 0.0

    gray = (255 * prob_map_filter).astype(np.uint8)

    # original pred
    pred = np.argmax(prob_map_b, axis=0)

    for bound_inx in range(1, n_channel):  # inner and outer bound
        prob_map_bb = prob_map_b[bound_inx]
        pred_filter = (prob_map_bb >= thres).astype(np.uint8)

        # # Hit&Miss operation to find singleton pixels and dilate them
        # kernel_hs = np.array(
        #     [[0, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 0]])
        # singleton = cv2.morphologyEx(pred_filter, cv2.MORPH_HITMISS, kernel_hs)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # slt_dilate = cv2.dilate(singleton, kernel)
        # pred_filter = np.logical_or(pred_filter, slt_dilate).astype(np.uint8)

        # pred_filter = cv2.dilate(pred_filter, kernel)

        # Morphology
        image_close = gray[bound_inx]
        for iter in range(iter_max):
            image_close = cv2.morphologyEx(image_close, cv2.MORPH_CLOSE, kernel)
            _, contours, _ = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours =[contour for contour in contours
                       if len(contour) > 4 and cv2.contourArea(contour) / len(contour) >= 4.0]

            if len(contours) > 0: # a closed contour can be obtained by cv2.findContours
                print("the {}th iteration".format(iter+1))
                fc = 1
                break

        if fc == 1:
        # print("{} : cv2.findContour functions well".format(bound_type))
            mask = np.zeros(image_close.shape[:2], np.uint8)
            ls = cv2.drawContours(mask, contours, -1, 1, -1)

        else:
             fc = 0
             # print("{} : Snake is necessary".format(bound_type))
             if bound_inx == 1:
                 gimage = inverse_gaussian_gradient(img_as_float(image_close), alpha=100, sigma=5.0)
             else:
                 gimage = inverse_gaussian_gradient(img_as_float(image_close), alpha=100, sigma=3.0)
             init = np.zeros(gimage.shape, dtype=np.int8)
             init[5:-5, 5:-5] = 1

             # np.ndarray with type np.int8
             ls = morphological_geodesic_active_contour(gimage, 100, init, smoothing=1,
                                                        balloon=-1, threshold='auto')


        # image_close = cv2.erode(image_close, np.zeros((5, 5), dtype=np.uint8))

        # image_close = pred_filter

        # kernel = np.ones((2, 2), np.uint8)
        # image_close = cv2.erode(image_close, kernel, iterations=1)

        # _, contours, _ = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours =[contour for contour in contours
        #            if len(contour) > 16 and cv2.contourArea(contour) / len(contour) >= 4.0]
        #
        # if len(contours) > 0: # a closed contour can be obtained by cv2.findContours
        #     fc = 1
        #     # print("{} : cv2.findContour functions well".format(bound_type))
        #     mask = np.zeros(image_close.shape[:2], np.uint8)
        #     ls = cv2.drawContours(mask, contours, -1, 1, -1)
        #
        # else: # need use snake to obtain closed contour

        lses.append(ls)
        fcs.append(fc)
        images_close.append(image_close)

    if do_plot:
        # plot slice-wise experiment results
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4),
                               sharex=True, sharey=True)

        methods = {0: 'Snake', 1: 'FindContours'}

        # inner bound
        ax[0, 0].imshow(prob_map_b[1], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[0, 0].set_axis_off()
        ax[0, 0].set_title("probability map (inner)", fontsize=6)

        ax[0, 1].imshow(prob_map_filter[1], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[0, 1].set_axis_off()
        ax[0, 1].set_title("after filter (inner)", fontsize=6)

        ax[0, 2].imshow(images_close[0], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[0, 2].set_axis_off()
        ax[0, 2].set_title("after morphology (inner)", fontsize=6)

        ax[0, 3].imshow(prob_map_b[1], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[0, 3].set_axis_off()
        ax[0, 3].contour(lses[0], [0.5], colors='b')
        ax[0, 3].set_title("{} boundary (inner)".format(methods[fcs[0]]), fontsize=6)

        # outer bound
        ax[1, 0].imshow(prob_map_b[2], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[1, 0].set_axis_off()
        ax[1, 0].set_title("probability map (outer)", fontsize=6)

        ax[1, 1].imshow(prob_map_filter[2], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[1, 1].set_axis_off()
        ax[1, 1].set_title("after filter (outer)", fontsize=6)

        ax[1, 2].imshow(images_close[1], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[1, 2].set_axis_off()
        ax[1, 2].set_title("after morphology (outer)", fontsize=6)

        ax[1, 3].imshow(prob_map_b[2], cmap="seismic", vmin=0.0, vmax=1.0)
        ax[1, 3].set_axis_off()
        ax[1, 3].contour(lses[1], [0.5], colors='b')
        ax[1, 3].set_title("{} boundary (outer)".format(methods[fcs[1]]), fontsize=6)

        fig.tight_layout()
        # plt.show()

        if not osp.exists("./Snake_results/probmap"):
            os.makedirs("./Snake_results/probmap")
        # save snake results into figure
        plt.savefig("./Snake_results/probmap/{}.pdf".format(i))

    # calculate hd95 and ahd for inner and outer bound respectively
    reg = lslist2bound(lses)
    ahd_pred_reg = slicewise_ahd(pred, reg)
    ahd_pred, hd95_pred = slicewise_ahd(pred, label_b, 3), slicewise_hd95(pred, label_b, 3)
    if ahd_pred_reg > 10.0:
        ahd_reg, hd95_reg = ahd_pred, hd95_pred
    else:
        ahd_reg, hd95_reg = slicewise_ahd(reg, label_b, 3), slicewise_hd95(reg, label_b, 3)

    print("AHD: Pred - {:.4f}, Reg - {:.4f} Pred&Reg diff - {}, HD95: Pred - {:.4f}, Reg - {:.4f}".format(
        ahd_pred, ahd_reg, ahd_pred_reg, hd95_pred, hd95_reg))


    # # for contour in contours_list[1]:
    # #     contour = np.squeeze(contour)
    # #     rows, cols = zip(*contour)
    # #     image[2][cols, rows] = 0.5
    # ax[1].imshow(image[2], cmap="seismic", vmin=0.0, vmax=1.0)
    # ax[1].set_axis_off()
    # ax[1].contour(snake[1], [0.5], colors='b')
    # # ax[0].contour(ls_inner, [0.5], colors='r')
    # ax[1].set_title("outer bound ({})".format(methods[fc[1]]), fontsize=12)
    #
    # ax[2].imshow(image[1], cmap="seismic", vmin=0.0, vmax=1.0)
    # ax[2].set_axis_off()
    # ax[2].contour(snake[0], [0.5], colors='b')
    # # ax[0].contour(ls_inner, [0.5], colors='r')
    # ax[2].set_title("inner bound ({})".format(methods[fc[0]]), fontsize=12)

    return fcs, ahd_pred, ahd_reg, hd95_pred, hd95_reg

def probmap2snake_debug(prob_maps, labels, thres=0.7):
    """ obtain snake from probability map """

    if prob_maps.dim() == 5:  # 3D volume
        prob_maps = prob_maps.permute(0, 2, 1, 3, 4)
        prob_maps = prob_maps.contiguous().view(-1, *prob_maps.size()[2:])  # combine first 2 dims
        labels = labels.contiguous().view(-1, *labels.size()[2:])

    # convert tensor into ndarray for probmap, label and prediction
    batch_size, n_channel, height, width = prob_maps.size()
    prob_maps_np = prob_maps.data.cpu().numpy() #[B', C, H, W]
    labels_np  = labels.data.cpu().numpy()

    n_fcs = [0, 0] # inner and outer bound respectively
    ahds_reg, ahds_pred, hd95s_pred, hd95s_reg = [], [], [], []
    for i, (probmap, label) in enumerate(zip(prob_maps_np, labels_np)):
        print("Processing the {}th image".format(i))
        # snake for inner bound and outer bound (list)
        fcs, ahd_pred, ahd_reg, hd95_pred, hd95_reg = prob2snake_slicewise_debug(i, probmap,
                 label, thres)

        n_fcs = [a + b  for (a, b) in zip(n_fcs, fcs)]
        ahds_pred.append(ahd_pred)
        ahds_reg.append(ahd_reg)
        hd95s_pred.append(hd95_pred)
        hd95s_reg.append(hd95_reg)

    print("{}/{} images can obtain inner bound using cv2.findContours".format(n_fcs[0], batch_size))
    print("{}/{} images can obtain outer bound using cv2.findContours".format(n_fcs[1], batch_size))
    print("ave AHD: Pred - {:.4f}, Reg - {:.4f}  ave HD95: Pred - {:.4f}, Reg - {:.4f}".format(
        sum(ahds_pred)/batch_size, sum(ahds_reg)/batch_size, sum(hd95s_pred)/batch_size, sum(hd95s_reg)/batch_size))


def pred2snake(pred, n_channel=3, n_workers=2, n_loops = 100, smoothing = 1, balloon=-1, thickness=1):
    """ obtain snake from prediction result """
    if pred.dim() == 4:  # 3D volume
        pred = pred.contiguous().view(-1, *pred.size()[2:])  # combine first 2 dims

    batch_size, height, width = pred.size()
    pred_np = pred.data.cpu().numpy() #[B', H, W]

    for i, image in enumerate(pred_np):
        # display results [ax1 -- original image, ax2 -- detected contours]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3),
                                                 sharex=True, sharey=True)

        # Initial level set, Morphological GAC
        n_loops = 100
        image = mask2gray(image)

        gimage = inverse_gaussian_gradient(image)
        h, w = gimage.shape

        # outer boundary
        init = np.zeros(image.shape, dtype=np.int8)
        init[5:-5, 5:-5] = 1

        ls = morphological_geodesic_active_contour(gimage, n_loops, init, smoothing=smoothing, balloon=balloon,
                                                         threshold='auto')
        # print("contour shape: {}, type: {}".format(ls.shape, type(ls)))
        # print("max value : {}, min value : {}".format(ls.max(), ls.min()))

        # convert segmentation into contour
        # tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
        # bound = np.logical_and(tmp >= 1, tmp <= 1)

        # show original image
        ax[0].imshow(image, cmap="gray")
        # ax[0].imshow(mask2rgb(bound_outer))
        ax[0].set_axis_off()
        ax[0].contour(ls, [0.5], colors='r')
        # ax[0].contour(ls_inner, [0.5], colors='r')
        ax[0].set_title("original image", fontsize=12)

        # show image after passing through filter
        ax[1].imshow(gimage, cmap="gray")
        ax[1].set_axis_off()
        ax[1].contour(ls, [0.5], colors='r')
        # ax[1].contour(ls_inner, [0.5], colors='r')
        ax[1].set_title("image after filter", fontsize=12)

        # # show comparison of GT mask and predicted contour
        # bound = gray2innerouterbound(label, width=1)
        # ax[2].imshow(mask2rgb(bound))
        # ax[2].set_axis_off()
        # ax[2].contour(ls_outer, [0.5], colors='r')
        # ax[2].contour(ls_inner, [0.5], colors='r')
        # ax[2].set_title("Snake vs GT Annotation", fontsize=12)

        # ax[3].imshow(gray2rgb(label))
        # ax[3].axis('off')
        # ax[3].set_title('Snake result')
        # ax[3].plot(init_sk[:, 0], init_sk[:, 1], '-b', lw=3)
        # ax[3].plot(snake[:, 0], snake[:, 1], '-r', lw=3)

        fig.tight_layout()
        # plt.show()
        plt.savefig("./Snake_results/{}.png".format(i))

    ## snake operation and plot the results here


    # args = []
    # for b in range(batch_size):
    #     args.append((pred_np[b], n_channel, n_loops, smoothing, balloon, thickness))
    #
    # pool = Pool(processes=n_workers)
    # snakes = pool.starmap(slicewise_process_pred, args)
    # pool.close()
    # snakes = np.stack(snakes).astype(np.uint8)
    # snake_cuda = Variable(torch.from_numpy(snakes).cuda()).long()  # with size [B', C, H, W]
    # return snakes


def test_probamap2snake():
    """ test whether snake boundary can be perfectly obtained from probability map """
    # data_dir = "/Users/AlbertHuang/Documents/Programming/Python/CPR_Segmentation_ver7/PlaqueDetection_20181127/" \
    #            "BoundDetection/Experiment9/HybridResUNet_int15_ds1_whddbsnake_after10epoch/S21891c88c_S20bc404d932d8b_" \
               # "20160907/I000/data.pkl"

    data_dir = "/Users/AlbertHuang/Documents/Programming/Python/CPR_Segmentation_ver7/PlaqueDetection_20181127/" \
               "BoundDetection/Experiment9/HybridResUNet_int15_ds1_baseline_new/S21891c88c_S20bc404d932d8b_20160907/" \
               "I000/data.pkl"

    with open(data_dir, 'rb') as reader:
        data_bound = pickle.load(reader)
        prob_maps = data_bound['output']
        labels = data_bound['label']
        # convert ndarray into tensor
        prob_maps_cuda = torch.from_numpy(prob_maps)
        labels_cuda = torch.from_numpy(labels)
        probmap2snake_debug(prob_maps_cuda, labels_cuda)


def test_innerouter2mask():
    data_dir = "./BoundDetection/Experiment9/HybridResUNet_int15_ds1_baseline_new/S21891c88c_S20bc404d932d8b_20160907/" \
               "I000/data.pkl"

    save_dir = "./Innerouter2mask"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    with open(data_dir, 'rb') as reader:
        data_bound = pickle.load(reader)
        labels = data_bound['label']

    for i, label in enumerate(labels):
        mask = innerouterbound2mask(label)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3),
                               sharex=True, sharey=True)

        ax[0].imshow(mask2rgb(label))
        ax[0].set_title("bound")

        ax[1].imshow(mask2rgb(mask))
        ax[1].set_title("mask")

        plt.savefig(osp.join(save_dir, "{}.png".format(i)))


# def houghcircle():
#     import cv2 as cv2
#
#     img_color = cv2.imread('input.jpg')
#     img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#
#     img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
#
#     # Hough circle
#     circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, minDist=15,
#                                param1=50, param2=18, minRadius=12, maxRadius=22)
#
#     if circles is not None:
#         for i in circles[0, :]:
#             # draw the outer circle
#             cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             # draw the center of the circle
#             cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)
#
#     cv2.imwrite('with_circles.png', img_color)
#
#     cv2.imshow('circles', img_color)
#     cv2.waitKey(5000)

if __name__ == "__main__":
    # houghcircle()
    # test_innerouter2mask()
    test_probamap2snake()
    # img_file = "/data/ugui0/antonio-t/CPR_multiview/S218801d0c_S2052ee2457ad29_20160809/I10/" \
    #            "applicate/image/033.tiff"
    # mask_file = "/data/ugui0/antonio-t/CPR_multiview/S218801d0c_S2052ee2457ad29_20160809/I10/" \
    #            "applicate/mask/033.tiff"
    # img = io.imread(img_file)
    # gray = io.imread(mask_file)
    #
    # img_lut1 = hu2gray(img) / 255.0
    # # img_lut2 = hu2lut(img, window=600, level=300) / 255.0
    # # img_lut3 = hu2lut(img, window=400, level=100) / 255.0
    # snake_contour_detector(img_lut1, gray, fig_name='orig_image')
    # # snake_contour_detector(img_lut2, gray, fig_name='lut1')
    # # snake_contour_detector(img_lut3, gray, fig_name='lut2')