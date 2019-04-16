# _*_ coding: utf-8 _*_

""" Often used functions for data loading and visualisation """

import matplotlib as mpl
mpl.use('Agg')

import warnings
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy import ndimage

import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

def count_parameters(model):
    """ count number of parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rgb2gray(rgb):
    """ convert rgb image to grayscale one """
    if rgb.ndim == 2:
        img_gray = rgb
    elif rgb.ndim == 3:
        img_gray = np.dot(rgb, [0.299, 0.587, 0.114])

    return img_gray.astype(np.uint8)

def denormalize(image, v=182.7666473388672, m=-4.676876544952393):
    """ de-normalize image into original HU range """
    return (image * v  + m).astype(np.int16)


def rgb2mask(rgb):
    """ convert rgb image into mask
        red - (255, 0, 0) : low-density plaque --> 4
        black - (0, 0, 0) : background --> 0
        orange - (255, 128, 0) : calcification --> 3
        white - (255, 255, 255) : Border of the artery (small in healthy patients) --> 2
        blue - (0, 0, 255) : inside of the artery --> 1
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[np.all(rgb == [255, 0, 0], axis=2)] = 4
    mask[np.all(rgb == [255, 128, 0], axis=2)] = 3
    mask[np.all(rgb == [255, 255, 255], axis=2)] = 2
    mask[np.all(rgb == [0, 0, 255], axis=2)] = 1

    return mask


def gray2rgb(gray):
    """ convert grayscale rgb for some discrete values """
    h, w = gray.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[gray == 29] = [0, 0, 255]
    rgb[gray == 255] = [255, 255, 255]
    rgb[gray == 151] = [255, 128, 0]
    rgb[gray == 76] = [255, 0, 0]
    rgb[gray == 226] = [255, 255, 0]
    rgb[gray == 150] = [0, 255, 0]

    return rgb

def mask2rgb(mask):
    """ convert mask into RGB image """
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = [0, 0, 255]
    rgb[mask == 2] = [255, 255, 255]
    rgb[mask == 3] = [255, 128, 0]
    rgb[mask == 4] = [255, 0, 0]

    return rgb

def mask2gray(mask):
    """ convert mask to gray """
    h, w = mask.shape[:2]
    gray = np.zeros((h, w), dtype=np.uint8)
    gray[mask == 1] = 29
    gray[mask == 2] = 255
    gray[mask == 3] = 151
    gray[mask == 4] = 76

    return gray

def gray2mask(gray):
    """ convert gray-scale image to 2D mask
        red - 76 : low-density plaque --> 4
        black - 0 : background --> 0
        orange - 151 : calcification --> 3
        white - 255 : Border of the artery (small in healthy patients) --> 2
        blue - 29 : inside of the artery --> 1
    """
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray == 76] = 4
    mask[gray == 151] = 3
    mask[gray == 255] = 2
    mask[gray == 29] = 1

    return mask

def central_crop(image, patch_size):
    """ centre crop the given image
    Args:
        im: numpy ndarray, input image
        new_size: tuple, new image size
    """

    assert isinstance(patch_size, (int, tuple)), "size must be int or tuple"
    if isinstance(patch_size, int):
        size = (patch_size, patch_size)
    else:
        size = patch_size

    h, w = image.shape[:2]
    assert (h - size[0]) % 2 == 0 and (w - size[1]) % 2 == 0, \
        "new image size must match with the input image size"
    h_low, w_low = (h - size[0]) // 2, (w - size[1]) // 2
    h_high, w_high = (h + size[0]) // 2, (w + size[1]) // 2

    new_image = image[h_low:h_high, w_low:w_high]

    return new_image

def dcm2hu(dcm):
    """ convert dicom image into Hounsfield (HU) value """

    image = dcm.pixel_array
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    image = slope * image + intercept
    return np.array(image, dtype=np.int16)

def hu2gray(image, hu_max=1640.0, hu_min=-1024.0):
    scale = float(255) / (hu_max - hu_min)
    image =  (image - hu_min) * scale

    return image

def gray2m11range(image):
    """ convert grayscale to -1~1 """
    return 2.0 * image / 255.0 - 1.0


def hu2lut(data, window, level):
    lut = np.piecewise(data, [data <= (level - 0.5 - (window - 1) / 2),
                              data > (level - 0.5 + (window - 1) / 2)],
                       [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])

    return lut.astype(np.float32)

def hu2norm(image, hu_max=1440.0, hu_min=-1024.0):
    """ scale into (0.0 1.0) """
    scale = 1.0 / (hu_max - hu_min)
    image =  (image - hu_min) * scale

    return image

def shuffle_backward(l, order):
    """ shuffle back to original """
    l_out = np.zeros_like(l)
    for i, j in enumerate(order):
        l_out[j] = l[i]
    return l_out


def gray2dboulebound(gray, width=2):
    """ convert mask with grayscale value to inner bounds and outer bounds respectively """

    h, w = gray.shape[:2]
    gray[gray == 76] = 255
    gray[gray == 151] = 255

    label = gray2mask(gray)
    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    inner_bound, outer_bound = bound_binary[:, :, 1], bound_binary[:, :, 0]

    return inner_bound, outer_bound

def gray2bound(gray, n_classes=3, width=2):
    """ convert mask with grayscale value to inner bound, outer bound, cal and noncal bound """

    h, w = gray.shape[:2]
    if n_classes <= 3: # if n_classes less than 3, cal and noncal are not considered
        gray[gray == 76] = 255
        gray[gray == 151] = 255

    label = gray2mask(gray)
    label_binary = label_binarize(label.flatten(), classes=range(0, n_classes))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(n_classes):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound = np.any(bound_binary, axis=2).astype(np.uint8)

    return bound

def gray2triplewithbound(gray, n_classes=4, width=1):
    """ convert gray to triple seg with bounds """
    h, w = gray.shape[:2]
    gray[gray == 76] = 255
    gray[gray == 151] = 255

    label = gray2mask(gray)
    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound = np.any(bound_binary, axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)

    mask[gray == 255] = 2
    mask[gray == 29] = 1

    if n_classes == 4:
        mask[bound] = 3
    elif n_classes == 5:
        mask[bound_binary[:, :, 1] == 1] = 3  # inner boundary
        mask[bound_binary[:, :, 0] == 1] = 4  # outer boundary

    return mask

def gray2innerbound(gray, width):
    """ convert grayscale annotation to inner bound """
    h, w = gray.shape[:2]
    gray[gray == 76] = 255
    gray[gray == 151] = 255

    label = gray2mask(gray)
    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))

    tmp = ndimage.distance_transform_cdt(label_binary[:, :, 1], 'taxicab')
    inner_bound = np.logical_and(tmp >= 1, tmp <= width).astype(np.uint8)

    return inner_bound

def gray2outerbound(gray, width):
    """ convert grayscale annotation to outer bound """
    h, w = gray.shape[:2]
    gray[gray == 76] = 255
    gray[gray == 151] = 255

    label = gray2mask(gray)
    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))

    tmp = ndimage.distance_transform_cdt(label_binary[:, :, 0], 'taxicab')
    outer_bound = np.logical_and(tmp >= 1, tmp <= width).astype(np.uint8)

    return outer_bound

# def gray2innerouterbound(gray, width=1):
#     """ convert mask with grayscale value to inner and outer boundaries
#         where inner and outer boundaries are treated as different classes
#     """
#
#     h, w = gray.shape[:2]
#     gray[gray == 76] = 255
#     gray[gray == 151] = 255
#     bound = np.zeros_like(gray, dtype=np.uint8)
#
#     bound = np.zeros_like(gray, dtype=np.uint8)
#     label = gray2mask(gray)
#     label_binary = label_binarize(label.flatten(), classes=range(0, 3))
#     label_binary = np.reshape(label_binary, (h, w, -1))
#     bound_binary = np.zeros_like(label_binary)
#
#     for i in range(3):  # use Sobel edge detector
#         horizontal = ndimage.sobel(label_binary[:, :, i], axis=0)
#         vertical = ndimage.sobel(label_binary[:, :, i], axis=1)
#         tmp = np.logical_or(horizontal != 0, vertical != 0)
#         bound_binary[:, :, i] = tmp
#
#     bound[bound_binary[:, :, 0] != 0] = 2  # outer bound marked as 2
#     bound[bound_binary[:, :, 1] != 0] = 1  # inner bound marked as 1
#
#     return bound

def gray2innerouterbound(gray, width):
    """ convert mask annotation into inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    h, w = gray.shape[:2]
    gray_cp = gray.copy()
    gray_cp[gray == 76] = 255
    gray_cp[gray == 151] = 255
    bound = np.zeros_like(gray, dtype=np.uint8)
    label = gray2mask(gray_cp)

    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound[bound_binary[:, :, 0] != 0] = 2  # outer bound marked as 2
    bound[bound_binary[:, :, 1] != 0] = 1  # inner bound marked as 1

    return bound

def mask2innerouterbound(mask, width):
    """ convert mask annotation into inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    h, w = mask.shape[:2]
    mask_np = mask.copy()
    mask_np[mask == 3] = 2
    mask_np[mask == 4] = 2
    
    bound = np.zeros_like(mask_np, dtype=np.uint8)
    label_binary = label_binarize(mask_np.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound[bound_binary[:, :, 0] != 0] = 2 # outer bound marked as 2
    bound[bound_binary[:, :, 1] != 0] = 1 # inner bound marked as 1

    return bound

def innerouterbound2mask(innerouter, n_classes=3):
    """ transform innerouter bound to mask segmentation, cv2.drawContours is used for transformation
    :param innerouter: ndarray of size [H, W], 1 - inner, 2 - outer
    :param n_classes: int, number of classes
    :return: mask: ndarray of size [H, W]
    """
    # only apply to situation with n_classes = 3
    ls = np.zeros(innerouter.shape[:2], np.uint8)
    for c_inx in range(n_classes-1, 0, -1):
        points = np.array(np.where(innerouter == c_inx)).transpose() # [N, 2]
        points = np.expand_dims(np.flip(points, axis=1), axis=1) # [N, 2] --> [N, 1, 2]
        ls = cv2.drawContours(ls, [points], -1, c_inx, thickness=cv2.FILLED)

    return ls

def mask2outerbound(mask, width):
    """ convert mask annotation into inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    h, w = mask.shape[:2]
    mask[mask == 3] = 2
    mask[mask == 4] = 2

    bound = np.zeros_like(mask, dtype=np.uint8)
    label_binary = label_binarize(mask.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound[bound_binary[:, :, 0] != 0] = 1  # outer bound marked as 2

    return bound

def mask2bounds(mask, width=2, n_classes=3):
    """ convert mask (with value range from 0 to n_classes-1) to bounds
        this operation is similar to gray2bounds
    """
    h, w = mask.shape[:2]

    label_binary = label_binarize(mask.flatten(), classes=range(0, n_classes))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(n_classes):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bounds = np.any(bound_binary, axis=2).astype(np.uint8)

    return bounds

def ls2bound(ls, width=1):
    """ convert morphological snake result into boundary """
    tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
    bound = np.logical_and(tmp >= 1, tmp <= width)

    return bound


def lslist2bound(ls_list):
    """ convet ls list to boundary """

    h, w = ls_list[0].shape
    bound = np.zeros((h, w), dtype=np.uint8)
    for inx, ls in enumerate(ls_list):
        bound[ls2bound(ls, width=1)] = inx + 1

    return bound


if __name__ == "__main__":
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    # gray = img_as_float(gray)
    rgb = gray2rgb(gray)
    print(rgb.shape)
    print(rgb.dtype)
    plt.figure()
    plt.imshow(rgb)
    plt.savefig("test.png")



# def read_loss_acc_from_txt(file_name, fig_name= None):
#     """ read loss and acc from txt file for both train and test data
#         and return the result as a dictionary
#     """
#     train_loss, train_acc = [], []
#     test_loss, test_acc = [], []
#     with open(file_name, 'rb') as f:
#         for line in f.readlines():
#             line = line.strip()
#             if line.startswith('train'):
#                 loss, acc = float(line.split(' ')[2]), float(line.split(' ')[4])
#                 train_loss.append(loss)
#                 train_acc.append(acc)
#             if line.startswith('test'):
#                 loss, acc = float(line.split(' ')[2]), float(line.split(' ')[4])
#                 test_loss.append(loss)
#                 test_acc.append(acc)
#
#     data =  {'train': (train_loss, train_acc),
#             'test': (test_loss, test_acc)}
#     # plot accuracy
#     plt.figure()
#     for phase, value in data.iteritems():
#         loss, acc = value
#         plt.plot(acc, label=phase)
#     plt.title("pixel-wise segmentation accuracy")
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy (%)")
#     plt.legend()
#     if fig_name:
#         plt.savefig(fig_name+'_acc.png')
#     plt.close()
#
#     # plot loss
#     plt.figure()
#     for phase, value in data.iteritems():
#         loss, acc = value
#         plt.plot(loss, label=phase)
#     plt.title("pixel-wise segmentation loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend()
#     if fig_name:
#         plt.savefig(fig_name+'_loss.png')
#     plt.close()