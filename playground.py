# _*_ coding: utf-8 _*_

""" play on how to use parser to collect all possible parameters """

# import matplotlib as mpl
# mpl.use('Agg')

from torch.autograd import Variable
import os
import os.path as osp
import os
from os import listdir
from skimage import io
import shutil
import math
import time
from torch import nn
import pydicom as dicom
from skimage import io
from utils import gray2rgb, gray2mask, gray2innerouterbound, mask2rgb
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable
import torch.functional as F
import torch
import _pickle as pickle
# from vision import plot_risk_confusion_matrix
from image.dataloader import show_train_dataloader, cal_mean_std_dataloader
from datasets.multiview import create_multiview_dataset_multi_preocess, create_multiview_dataset
from datasets.multiview import remove_redundant_slice_applicate_multi_preocess, remove_redundant_slice_applicate
from utils import dcm2hu
from vision import sample_stack, sample_stack_color
from torchvision import transforms
# from loss import MaxPoolLoss
from hybrid.dataloader_debug import debug_dataloader
from scipy import ndimage

import matplotlib.pyplot as plt

import operator
import random

import torch.nn.functional as F

from PIL import Image
import numpy as np

# from loss import grad_check
## test skimage and PIL, which one is faster

N = 128
N_SAMPLE = 10
SHOW_INTERVAL = 10

def PIL_image_loader():
    data_dir = "/home/mil/huang/Dataset/CPR_rawdata/test/PIL"
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    for s_inx in range(N_SAMPLE):
        start = time.time()
        for data_type in ['image', 'mask']:
            for i in range(N):
                if i % SHOW_INTERVAL == 0:
                    print("processing {}/{}".format(i, N))
                # generate random image for demo
                # img_arr = np.random.rand(-1000,1000, size=[10,10]).astype(np.int32)
                img_arr = 100 * np.random.rand(512, 512)
                # print("original min {}, max: {}".format(img_arr.min(),img_arr.max()))

                # create PIL image
                img1 = Image.fromarray(img_arr)
                # print("PIL min {}, max: {}".format(np.array(img1.getdata()).min(),np.array(img1.getdata()).max()))

                # save image
                save_dir = osp.join(data_dir, str(s_inx), data_type)
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                img1.save(osp.join(save_dir, "{}.tiff".format(i)))
                # io.imsave("test_file.tiff", img_arr)
                # reload image
                # img_file = Image.open("test_file.tiff")
                # print(type(img_file))

                # print("img_file min {}, max: {}".format(np.array(img_file.getdata()).min(),np.array(img_file.getdata()).max()))

        time_elapsed = time.time() - start
        print('PIL dataloading finished in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

def skimage_image_loader():
    data_dir = "/home/mil/huang/Dataset/CPR_rawdata/test/skimage"
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    for s_inx in range(N_SAMPLE):
        start = time.time()
        for data_type in ['image', 'mask']:
            for i in range(N):
                if i % SHOW_INTERVAL == 0:
                    print("processing {}/{}".format(i, N))
                # generate random image for demo
                # img_arr = np.random.rand(-1000,1000, size=[10,10]).astype(np.int32)
                img_arr = 100 * np.random.rand(512, 512)
                # print("original min {}, max: {}".format(img_arr.min(),img_arr.max()))

                # create PIL image
                # img1 = Image.fromarray(img_arr)
                # print("PIL min {}, max: {}".format(np.array(img1.getdata()).min(),np.array(img1.getdata()).max()))

                # save image
                save_dir = osp.join(data_dir, str(s_inx), data_type)
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                io.imsave(osp.join(save_dir, "{}.tiff".format(i)), img_arr)
                # io.imsave("test_file.tiff", img_arr)
                # reload image
                # img_file = Image.open("test_file.tiff")
                # print(type(img_file))

                # print("img_file min {}, max: {}".format(np.array(img_file.getdata()).min(),np.array(img_file.getdata()).max()))

        time_elapsed = time.time() - start
        print('PIL dataloading finished in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

def check_error_reason():
    from transforms import CentralCrop, Gray2Mask, ToTensor, HU2Gray, Rescale, Gray2Binary, Mask2Gray
    from transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, RandomFlip
    from torchvision import transforms

    image_path = "/Users/AlbertHuang/Downloads/IRB00000000001230400000000096454820110705HOSPNO/I000/image/029.tiff"
    mask_path = "/Users/AlbertHuang/Downloads/IRB00000000001230400000000096454820110705HOSPNO/I000/mask/029.tiff"

    trans_params = {
        'rescale': (96, 96),
        'central_crop' : (160, 160),
        'output_channel': 5,
        'mode': 'train',
        'num_workers': 16,
        'batch_size': 128
    }

    comp1 = transforms.Compose([Mask2Gray(),
                                RandomRotation()])
    comp2 = transforms.Compose([Mask2Gray(),
                                RandomRotation(),
                                RandomFlip()])
    comp3 = transforms.Compose([Mask2Gray(),
                                RandomRotation(),
                                RandomFlip(),
                                CentralCrop(trans_params['central_crop'])])
    comp4 = transforms.Compose([Mask2Gray(),
                                RandomRotation(),
                                RandomFlip(),
                                CentralCrop(trans_params['central_crop']),
                                Rescale(trans_params['rescale'])])
    comps = [comp1, comp2, comp3, comp4]
    sample = (io.imread(image_path), io.imread(mask_path))
    _, ax = plt.subplots(2, 2, figsize=[50, 50])

    for i, comp in enumerate(comps):
        image, mask = comp(sample)
        ax[i//2, i%2].imshow(mask)
    plt.show()

def plot_bc_sample():
    img1_path = "/data/ugui0/antonio-t/CPR_multiview/IRB00000000001219700000000211143720110623HOSPNO/I000/applicate/image/024.tiff"
    img2_path = "/data/ugui0/antonio-t/CPR_multiview/IRB00000000001219700000000211143720110623HOSPNO/I000/applicate/image/248.tiff"
    mask1_path = "/data/ugui0/antonio-t/CPR_multiview/IRB00000000001219700000000211143720110623HOSPNO/I000/applicate/mask/024.tiff"
    mask2_path = "/data/ugui0/antonio-t/CPR_multiview/IRB00000000001219700000000211143720110623HOSPNO/I000/applicate/mask/248.tiff"

    img1 = io.imread(img1_path)[96:416, 96:416]
    img2 = io.imread(img2_path)[96:416, 96:416]
    mask1 = gray2rgb(io.imread(mask1_path))[96:416, 96:416]
    mask2 = gray2rgb(io.imread(mask2_path))[96:416, 96:416]

    img_mix = 0.3 * img1 + 0.7 * img2
    mask_mix = (0.3 * mask1 + 0.7 * mask2).astype(np.uint8)

    plt.figure()
    plt.imshow(img1, cmap='gray')
    plt.savefig("img1.png")

    plt.figure()
    plt.imshow(img2, cmap='gray')
    plt.savefig("img2.png")

    plt.figure()
    plt.imshow(img_mix, cmap='gray')
    plt.savefig("img_mix.png")

    plt.figure()
    plt.imshow(mask1)
    plt.savefig("mask1.png")

    plt.figure()
    plt.imshow(mask2)
    plt.savefig("mask2.png")

    plt.figure()
    plt.imshow(mask_mix)
    plt.savefig("mask_mix.png")

def calcium_score_check():
    orig_data_path = "/Users/AlbertHuang/Downloads/original_data/S2189144F5_S1FF973CBBE53CC_20180913/S3020"
    result_data_path = "/Users/AlbertHuang/Downloads/results/S2189144F5_S1FF973CBBE53CC_20180913/S3020/tiff"
    # dcm files
    orig_images = sorted([img for img in listdir(orig_data_path) if not img.startswith('.')],
                         key= lambda x: int(x[1:]))
    # tiff files
    result_images = sorted([img for img in listdir(result_data_path) if not img.startswith('.')],
                         key=lambda x: int(x.split('.')[0][1:]))

    orig_image = np.stack([dcm2hu(dicom.read_file(osp.join(orig_data_path, file)))
                      for file in orig_images])

    result_image = np.stack([io.imread(osp.join(result_data_path, file))
                           for file in result_images])

    sample_stack(orig_image, rows=20, cols=4, start_with=0, show_every=1, scale=4, fig_name = "./orig_image")
    sample_stack(result_image, rows=20, cols=4, start_with=0, show_every=1, scale=4, fig_name="./result_image")


def mpl_debug():
    """ check whether mpl works well or not """
    criterion = nn.CrossEntropyLoss(weight=None, reduce=True)
    mpl = MaxPoolLoss(criterion)
    output = torch.rand(20, 5, 96, 96).float()
    output = Variable(output, requires_grad=True)

    target = torch.randint(0, 5, (20, 96, 96)).long()
    target = Variable(target, requires_grad=False)

    loss = criterion(output, target)
    print("loss before Max-Pooling: {}".format(loss))
    loss_mpl = mpl(output, target)
    print("loss after Max-Pooling: {}".format(loss_mpl))


def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.rand(100, 100) * 256
    print("data: MAX - {}, MIN - {}".format(data.max(), data.min()))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=0, vmax=256)
        fig.colorbar(psm, ax=ax)
    plt.show()


def colormap_test():
    """ test whether listed colormap works well or not """

    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    top = cm.get_cmap('Reds', 186)
    bottom = cm.get_cmap('Blues', 70)
    viridis = cm.get_cmap('viridis', 256)

    newcolors = np.vstack((bottom(np.linspace(1, 0, 70)),
                           top(np.linspace(0, 1, 186))))
    newcmp = ListedColormap(newcolors, name='BlueReds')
    plot_examples([viridis, newcmp])


def canny_edge_detector():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    from skimage import feature

    # read mask figure
    im = io.imread("/Users/AlbertHuang/Downloads/applicate/mask/076.tiff")
    im_pad = np.pad(im, (1, 1), 'edge')
    print("size after padding: {}".format(im_pad.shape))


    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=3)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)

    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()

def canny_sobel_comp():
    from scipy import ndimage
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import feature


    im_dir = "/Users/AlbertHuang/Downloads/applicate/mask"
    im_files = [file for file in listdir(im_dir) if file.endswith('.tiff')]
    for file in im_files:
        im = io.imread(osp.join(im_dir, file))
        # im = gray2mask(im)

        # display results
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 9),
                                            sharex=True, sharey=True)
        # calculate sobel edge
        sobel = gray2innerouterbound(im)
        n_pixels_sobel = np.sum(sobel != 0)

        canny = feature.canny(im, sigma=3)
        n_pixels_canny = np.sum(canny != 0)

        ax1.imshow(im, cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('input')

        ax2.imshow(sobel, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title('Sobel filter ({} pixels)'.format(n_pixels_sobel))

        ax3.imshow(canny, cmap=plt.cm.gray)
        ax3.axis('off')
        ax3.set_title('Canny filter ({} pixels)'.format(n_pixels_canny))

        plt.savefig("/Users/AlbertHuang/Downloads/{}.png".format(file.split('.')[0]))


def gradient_check():
    """ check the gradient of input image and corresponding mask """

    import numpy as np
    import matplotlib.pyplot as plt
    from image.transforms import CentralCrop, HU2Gray, RandomRotation, RandomFlip, Rescale, Gray2InnerOuterBound
    processing = transforms.Compose([HU2Gray(),
                                     RandomRotation(),
                                     RandomFlip(),
                                     CentralCrop(192),
                                     Rescale(96)])

    im_dir = "/Users/AlbertHuang/Downloads/applicate/image"
    mask_dir = "/Users/AlbertHuang/Downloads/applicate/mask"
    im_files = [file for file in listdir(im_dir) if file.endswith('.tiff')]
    for file in im_files:
        im = io.imread(osp.join(im_dir, file))
        label = io.imread(osp.join(mask_dir, file))
        # display results
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(24, 12),
                                            sharex=True, sharey=True)
        # calculate sobel edge
        im, mask = processing((im, label))
        im, edge = Gray2InnerOuterBound(width=1)((im, mask))
        grad_cords = np.gradient(im)
        grad = np.sqrt(grad_cords[0] ** 2 + grad_cords[1] ** 2)

        ax1.imshow(im, cmap='seismic', vmin=0.0, vmax=255.0)
        ax1.axis('off')
        ax1.set_title('input')

        # print("Mask: max -- {}, min -- {}".format(mask.max(), mask.min()))
        ax2.imshow(gray2rgb(mask))
        ax2.axis('off')
        ax2.set_title('GT mask')

        ax3.imshow(mask2rgb(edge))
        ax3.axis('off')
        ax3.set_title('GT boundary')

        ax4.imshow(grad, cmap='seismic')
        ax4.axis('off')
        ax4.set_title('Gradient map')

        plt.savefig("/Users/AlbertHuang/Downloads/applicate/{}.png".format(file.split('.')[0]))

def prune_dir(ref_dir, oper_dir):
    """ prune oper_dir based on ref_dir
    :param ref_dir: str, reference directory
    :param oper_dir: str, directory to operate on
    """
    files = [file for file in listdir(ref_dir)
             if file.startswith('S') and osp.isdir(osp.join(ref_dir, file))]

    for file in listdir(oper_dir):
        if file.startswith('S') and not file in files:
            shutil.rmtree(osp.join(oper_dir, file))


def snake_demo():
    from metric import slicewise_hdf
    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
        def _store(x):
            lst.append(np.copy(x))

        return _store

    import numpy as np
    import matplotlib.pyplot as plt
    from image.transforms import CentralCrop, HU2Gray, RandomRotation, RandomFlip, Rescale, Gray2InnerOuterBound
    from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
    # from skimage.filters import gaussian
    # from skimage.segmentation import active_contour

    processing = transforms.Compose([HU2Gray(),
                                     # RandomRotation(),
                                     # RandomFlip(),
                                     CentralCrop(192),
                                     Rescale(96)])

    im_dir = "/Users/AlbertHuang/Downloads/applicate/image"
    mask_dir = "/Users/AlbertHuang/Downloads/applicate/mask"
    hdfs = []
    im_files = [file for file in listdir(im_dir) if file.endswith('.tiff')]
    for file in im_files:
        image = io.imread(osp.join(im_dir, file))
        label = io.imread(osp.join(mask_dir, file))
        image, label  = processing((image, label))
        # display results [ax1 -- original image, ax2 -- detected contours]
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3),
                                                 sharex=True, sharey=True)

        # Initial level set, Morphological GAC
        n_loops = 200
        gimage = inverse_gaussian_gradient(image)
        # check whehter morphological GAC works well on randomly generated image
        # image = np.random.rand(*image.shape)
        # gimage = inverse_gaussian_gradient(image)

        h, w = gimage.shape
        # outer boundary
        init_outer = np.zeros(image.shape, dtype=np.int8)
        init_outer[5:-5, 5:-5] = 1
        evolution_outer = []
        callback = store_evolution_in(evolution_outer)
        ls_outer = morphological_geodesic_active_contour(gimage, n_loops, init_outer, smoothing=3, balloon=-1,
                                                         threshold='auto', iter_callback=callback)
        # print("contour shape: {}, type: {}".format(ls_outer.shape, type(ls_outer)))
        # print("max value : {}, min value : {}".format(ls_outer.max(), ls_outer.min()))

        # inner boundary
        init_inner = np.zeros(image.shape, dtype=np.int8)
        init_inner[(h//2-2):(h//2+2), (w//2-2):(w//2+2)] = 1
        evolution_inner = []
        callback = store_evolution_in(evolution_inner)
        ls_inner = morphological_geodesic_active_contour(gimage, n_loops, init_inner, smoothing=1, balloon=1,
                                                         threshold='auto', iter_callback=callback)
        # # convert segmentation into contour
        # tmp = ndimage.distance_transform_cdt(ls_outer, 'taxicab')
        # bound_outer = np.logical_and(tmp >= 1, tmp <= 1)

        # calculate hdf between label and bound detected by snake
        label_bound = gray2innerouterbound(label, width=1)
        inner_bound = ls2bound(ls_inner, width=1)
        outer_bound = ls2bound(ls_outer, width=1)
        bound = inner_bound.copy()
        bound[outer_bound != 0] = 2
        hdf = slicewise_hdf(bound, label_bound, n_classes=3)
        print("hdf: {}".format(hdf))
        hdfs.append(hdf)

        # show original image
        ax[0].imshow(image, cmap="gray", vmin=0.0, vmax=255.0)
        # ax[0].imshow(mask2rgb())
        ax[0].set_axis_off()
        ax[0].contour(ls_outer, [0.5], colors='b')
        # ax[0].contour(ls_inner, [0.5], colors='r')
        ax[0].set_title("original image", fontsize=12)

        # show image after passing through filter
        ax[1].imshow(gimage, cmap="gray")
        ax[1].set_axis_off()
        # ax[1].contour(ls_outer, [0.5], colors='b')
        ax[1].contour(ls_inner, [0.5], colors='r')
        ax[1].set_title("image after filter", fontsize=12)

        # show comparison of GT mask and predicted contour
        bound = gray2innerouterbound(label, width=1)
        ax[2].imshow(mask2rgb(bound))
        ax[2].set_axis_off()
        ax[2].contour(ls_outer, [0.5], colors='r')
        ax[2].contour(ls_inner, [0.5], colors='r')
        ax[2].set_title("Snake vs GT Annotation", fontsize=12)

        # ax[3].imshow(gray2rgb(label))
        # ax[3].axis('off')
        # ax[3].set_title('Snake result')
        # ax[3].plot(init_sk[:, 0], init_sk[:, 1], '-b', lw=3)
        # ax[3].plot(snake[:, 0], snake[:, 1], '-r', lw=3)

        fig.tight_layout()
        # plt.show()
        plt.savefig("/Users/AlbertHuang/Downloads/applicate/{}.png".format(file.split('.')[0]))

def ls2bound(ls, width=1):
    """ convert morphological snake result into boundary """
    tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
    bound = np.logical_and(tmp >= 1, tmp <= width)

    return bound

def test_probamap2snake():
    from snake import probmap2snake, pred2snake, probmap2snake_debug
    from vision import sample_stack
    """ test whether snake boundary can be perfectly obtained from probability map """
    # data_dir = "/Users/AlbertHuang/Documents/Programming/Python/CPR_Segmentation_ver7/PlaqueDetection_20181127/" \
    #            "BoundDetection/Experiment9/HybridResUNet_int15_ds1_whddbsnake_after10epoch/S21891c88c_S20bc404d932d8b_" \
               # "20160907/I000/data.pkl"

    data_dir = "/Users/AlbertHuang/Documents/Programming/Python/CPR_Segmentation_ver7/PlaqueDetection_20181127/" \
               "BoundDetection/Experiment9/HybridResUNet_int15_ds1_baseline_new/S21891c88c_S20bc404d932d8b_20160907/" \
               "I000/data.pkl"

    with open(data_dir, 'rb') as reader:
        data_bound = pickle.load(reader)
        preds, prob_maps =  data_bound['pred'], data_bound['output']
        prob_maps_cuda = torch.from_numpy(prob_maps)
        probmap2snake_debug(prob_maps_cuda)

        # pred_cuda = torch.from_numpy(preds)
        # print(preds.shape, prob_maps.shape)
        # pred2snake(pred_cuda)

        # print("snakes shape : {}".format(snakes.shape))
        # fig_dir = "./Snake_results"
        # if not osp.exists(fig_dir):
        #     os.makedirs(fig_dir)
        # fig_name = "{}/snake".format(fig_dir)
        # sample_stack(snakes, rows=10, cols=10, start_with=0, show_every=1, scale=4, fig_name=fig_name)

if __name__ == "__main__":
    test_probamap2snake()
    # gradient_check()
    # snake_demo()
    # ref_dir = "/data/ugui0/antonio-t/CPR_multiview_interp2_huang"
    # oper_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # prune_dir(ref_dir, oper_dir)

    # from hybrid.models.hybrid_res_unet_reg import ResUNet18
    # in_channels = 1
    # out_channels = 3
    # n_slices = 31
    # input_size = 96
    # unet = ResUNet18(in_channels, out_channels, n_slices=n_slices, input_size=input_size)
    # print(unet)
    # x = torch.FloatTensor(6, in_channels, n_slices, input_size, input_size)  # the smallest patch size is 12 * 12
    # output, reg = unet(x)
    # print("reg term: {}".format(reg))

    # x = np.arange(9).reshape(3, 3)
    # print(x)
    #
    # y = np.diagonal(np.rot90(x))
    # print(y)



    # canny_edge_detector()
    # debug_dataloader()
    # canny_sobel_comp()
    # colormap_test()
    # from image.models.deeplab_resnet import Res_Deeplab
    # model = Res_Deeplab(input_channels=1, output_channels=5, pretrain=True)
    #
    # x = torch.rand(4, 1, 224, 224)
    # y = model(x)
    # print(y.size())

    # show_train_dataloader()
    # from image.models.res_unet import resunet_debug
    # resunet_debug()

    # mpl_debug()
    # print(3/5)
    # debug_polylr()
    # calcium_score_check()
    # x = np.random.random(3, 4)
    # y = np.random.rand(4, 3, 4)
   # show_train_dataloader()
   #  x = [1,2,3,4]
   #  y = [2,3,4]
   #  for elex, eley in zip(x, y):
   #      print(elex, eley)
   #
   #  x = np.random.rand(10, 10)
   #  inxs = [2,3,4]
   #  print(x[inxs])
   #
    # data_dir = "/data/ugui0/antonio-t/CPR_all"
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180601/20180601"
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180713"
    # des_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # # des_dir = "/home/mil/huang/Dataset/CPR_multiview"
    # create_multiview_dataset_multi_preocess(create_multiview_dataset, data_dir, des_dir, num_workers=16)
    #
    # data_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # remove_redundant_slice_applicate_multi_preocess(remove_redundant_slice_applicate, data_dir, 16)

    # show_train_dataloader()
    # weight = np.load("./class_weight.npy")
    # print("current weight: {}".format(weight))

    # x = torch.rand(2, 2, requires_grad=True)
    # y = torch.ones(2, 2, requires_grad=False)
    # print("x: ", x)
    # print("y: ", y)
    #
    #
    # loss = (x-y).pow(2)
    # print("loss: ", loss)
    # loss = loss.clone()
    # filter = (loss <= 0.1)
    # loss[filter] = 0.0
    # loss = loss.mean()
    # loss.backward()
    # print("auto grad: ", x.grad)
    #
    #
    # # manually calculate the differential of loss w.r.t x
    # grad = 2.0 * (x - y)/ x.numel()
    # grad[filter] = 0
    #
    # print("manual grad: ", grad)
    #
    # x = torch.randint(0, 5, (3, 4)).long()
    # # print(x)
    # y = torch.zeros(5, 3, 4)
    # y.scatter_(0, x.unsqueeze(0), 1)
    # z = 0.5 * y
    # print(z)
    # flag = (z == 0)
    # print(flag)

    # x = torch.FloatTensor([0.0])
    # print(x * torch.log(x+ 1.0e-9))

    # grad_check()

    # x = torch.rand(5).float()
    # print(x)
    # y = x.repeat(2, 3, 3, 1).permute(0, 3, 1, 2)
    # print(y)
    # print(y.size())

    # # data_dir = "/data/ugui0/antonio-t/01"
    # # data_dir = "/data/ugui0/antonio-t/CPR_20180831"
    # data_dir = "/data/ugui0/antonio-t/CPR_20180918/20190918-plaque"
    # # des_dir = "/home/mil/huang/Dataset/CPR_multiview"
    # des_dir = "/data/ugui0/antonio-t/CPR_multiview"
    # create_multiview_dataset_multi_preocess(create_multiview_dataset, data_dir, des_dir, num_workers=1)

    ### debug
    # from sklearn.preprocessing import label_binarize
    # from sklearn.metrics import f1_score
    # n_classes = 2
    #
    # for _ in range(100):
    #     label = np.random.randint(0, 1, (96, 96)).flatten()
    #     label_binary = np.stack([1-label, label], axis=1)
    #
    #     y = np.random.rand(96, 96).flatten()
    #     y_binary = np.stack([1-y, y], axis=1)
    #
    #     result = np.zeros(n_classes, dtype=np.float32)
    #
    #     for i in range(n_classes):
    #         result[i] = f1_score(label_binary[:, i], y_binary[:, i])
    #
    #     print(result, result.mean())


    # x = torch.randint(0, 2, (10, 10))
    # mask = (x == 0)
    #
    # y = torch.rand(10, 10).float()
    #
    # z = torch.zeros(10, 10).float()
    # z[mask] = y[mask]
    # print(z)

    # x = np.random.randint(0, 10, (10, 10))
    # y = np.random.rand(10, 10)
    # z = y[x == 3].flatten()
    # print(z)
    # print(type(z))
    # x = [] * 3
    # print(x)
    # from functools import reduce
    # x = [1,1,1,2,3,4]
    # values, cnts = np.unique(np.array(x), return_counts=True)
    # print(values)
    # print(cnts)
    #
    #
    # a = [[11], [12], [13]]
    # b = [[22], [34], [67]]
    # for v1, v2 in zip(a, b):
    #     z = reduce(lambda x, y: x + y, [v1, v2])
    #     print(z)


    # x = [-3, -4, 6, 11]
    # ave = sum(x[-8:]) / len(x[-8:])
    # print(ave)

    # x = range(0, 5)
    # y = range(0, 10)
    #
    # z = np.array(np.meshgrid(x, y, indexing='ij'))
    #
    # print(z.shape)
    # print(z[0]) # column index
    # print(z[1]) # row index

    # x = np.random.rand(5)
    # y = np.tile(x, (20, 10, 10, 1)).transpose()
    #
    # print(x)
    # print(y.shape)

    # with open("./noncal_map_6/file_paths.txt", 'r') as reader, open("./noncal_map_6/flie_paths_rev.txt", 'w') as writer:
    #     for line in reader.readlines():
    #         inx, file_name = line.strip('\n').split(':')
    #         new_line = "{} : {}".format(int(inx) + 1, file_name)
    #         writer.write("{}\n".format(new_line))
    # show_train_dataloader()
    #
    # x = torch.rand(10, 5, 20, 20)
    # # x = x.permute(0, 2, 3, 1)
    # y = torch.rand(5)
    #
    # z = y * x
    # print(z)

    # max_iter = 200
    # x = np.arange(0, max_iter)
    # y = (1 - x/max_iter) ** 0.9 # Poly LR
    # z = 0.9 ** (x//10)  # Step LR
    # plt.figure()
    # plt.plot(x, y, 'r.', markersize=5, label="PolyLR")
    # plt.plot(x, z, 'b.', markersize=5, label='StepLR')
    # plt.legend()
    # # plt.show()
    # # plt.close()
    # plt.savefig("./LR_figure/lr_comp_{}.png".format(max_iter))

    # show_train_dataloader()
    #
    # x = torch.rand(3, 4)
    # z = (x >= 0.7) | (x<= 0.3)
    # bounds = torch.nonzero(z)
    # print(bounds)
    # print(x)
    # print(z)
    # weight = np.load("nlf_weight_all_3.npy")
    #
    # weight4 = [*weight, 0.0]
    # np.save("nlf_weight_all_bound_4.npy", weight4)
    # weight4 = np.load("nlf_weight_all_bound_4.npy")
    # print(weight4)
    #
    # weight5 = [*weight, 0.0, 0.0]
    # np.save("nlf_weight_all_bound_5.npy", weight5)
    # weight5 = np.load("nlf_weight_all_bound_5.npy")
    # print(weight5)

    # cal_mean_std_dataloader()
    # show_train_dataloader()

    # plt.figure()
    # x = np.arange(10)
    # y = x + 0.5
    # err = np.sqrt(x)
    # plt.errorbar(x, y, yerr=err, fmt='-o')
    # for i in range(len(x)):
    #     plt.text(x[i], y[i]+err[i] + 1, "{:d}".format(int(err[i])), color="red",
    #              horizontalalignment="center", verticalalignment="center")
    #
    # plt.show()


    # data_dir = "/Users/AlbertHuang/Downloads/101_ObjectCategories"
    # sample_dict = {}
    # for file in listdir(data_dir):
    #     if not file.startswith('.') and osp.isdir(osp.join(data_dir, file)):
    #         sample_dict[file] = len(listdir(osp.join(data_dir, file)))
    #
    # sample_dict_x = sorted(sample_dict.items(), key=operator.itemgetter(1))
    #
    # keys, values = [[t[i] for t in sample_dict_x] for i in range(len(sample_dict_x[0]))]
    #
    # names = keys[:10] + keys[-10:]
    # cnts = values[:10] + values[-10:]
    # max_cnts = values[-10:]
    # print("top 10 cover {:.4f}% of the whole data".format(100 * sum(max_cnts)/sum(values)))
    # print(names)
    # print(cnts)
    # plt.figure()
    # plt.bar(range(len(names)), cnts, align="center")
    # plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("./class_distribution.jpg")

    # from skimage import transform
    # gupta_file = "/Users/AlbertHuang/Downloads/gupta.png"
    # antonio_file = "/Users/AlbertHuang/Downloads/antonio.png"
    # gupta = transform.resize(io.imread(gupta_file), (400, 300))
    # antonio = transform.resize(io.imread(antonio_file), (400, 300))
    # ratio = 0.5
    #
    # print(gupta.shape)
    # print(gupta.max(), gupta.min())
    # print(antonio.shape)
    #
    # gupta_antonio = ratio*gupta + (1.0 - ratio)*antonio
    #
    # plt.figure()
    # plt.imshow(gupta_antonio)
    # plt.savefig("/Users/AlbertHuang/Downloads/gupta_antonio.png")
    # # plt.show()

    # x = np.random.randint(-5, 5, (20, 30)).flatten()
    # inxes = np.array(np.where(x != 0)).min()
    # print(inxes)
    # height, width = 10, 10
    # all_points = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
    # print(all_points[0].size(), all_points[1].size())

    # from hybrid.models.hybrid_res_unet import ResUNet23 as ResUNet
    # from hybrid.models.utils import count_parameters
    # in_channels = 1
    # out_channels = 2
    # unet = ResUNet(in_channels, out_channels, p=0.2)
    # # print(unet)
    # x = torch.FloatTensor(6, 1, 31, 96, 96)  # the smallest patch size is 16 * 16
    # y = unet(x)
    #
    # print("number of parameters: {}".format(count_parameters(unet)))

    # show_plot_dataloader()