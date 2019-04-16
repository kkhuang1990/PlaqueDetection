# _*_ coding: utf-8 _*_

""" different types of transforms """

from skimage import transform
import torch
import random
import warnings
import cv2
from os import listdir
import os
import os.path as osp
from skimage.transform import rotate
from skimage import io
import numpy as np
import shutil
from utils import hu2lut, gray2mask, central_crop, hu2lut, hu2gray
from utils import gray2innerouterbound, gray2bound, gray2mask, rgb2mask, gray2triplewithbound
from utils import gray2innerbound, gray2outerbound
from scipy import ndimage
from sklearn.preprocessing import label_binarize
from torchvision.transforms import ToTensor

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

class RandomRotation(object):
    """ random rotation (angle is randomly set as a multiplier of given angle) """
    def __init__(self, angle=90, prob=0.8):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample
        rand_angle = random.randrange(0, 360, self.angle)
        x = random.uniform(0, 1)
        if x <= self.prob:
            image = rotate(image, rand_angle, mode='reflect', preserve_range=True)
            mask = rotate(mask, rand_angle, mode='reflect', preserve_range=True, order=0)

        return (image, mask)

class RandomFlip(object):
    """ random horizontal or vertical flip """
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample
        x = random.uniform(0, 1)
        if x < self.prob:
            phase = random.randint(0, 1)
            image = np.flip(image, phase)
            mask = np.flip(mask, phase)

        return (image, mask)


class Normalize(object):
    def __init__(self, m=-4.676876544952393, v=182.7666473388672):
        self.m = m
        self.v = v

    def __call__(self, sample):
        image, mask = sample
        image = (image - self.m) / self.v

        return (image, mask)

class Rescale(object):
    """Rescale the image in a sample to a given size with range preserved
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h // w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w // h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w), mode= 'reflect', preserve_range=True)
        mask = transform.resize(mask, (new_h, new_w), mode= 'reflect', preserve_range=True, order=0)

        return image, mask

class RandomCrop(object):
    """ Randomly crop the image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]

        return image, mask

class RGB2Mask(object):
    """ convert rgb image to 2D mask """
    def __call__(self, sample):
        image, rgb = sample
        mask = rgb2mask(rgb)

        return image, mask

class Gray2Mask(object):
    """ convert gray-scale image to 2D mask """
    def __call__(self, sample):
        image, gray = sample
        mask = gray2mask(gray)

        return image, mask

class Intercept(object):
    def __init__(self, intercept=-450):
        self.intercept = intercept

    def __call__(self, sample):
        hu, gray = sample
        h, w = gray.shape[:2]
        hu[hu<=self.intercept] = np.random.randint(self.intercept, -200, (h, w), dtype=hu.dtype)[hu<=self.intercept]

        return hu, gray

class Gray2TripleWithBound(object):
    """ convert grayscale value into triple segmentation + bound """
    def __init__(self, n_classes=4, width=1):  # number of classes after conversion
        self.n_classes = n_classes
        self.width = width

    def __call__(self, sample):
        image, gray = sample
        mask = gray2triplewithbound(gray, self.n_classes, self.width)

        return image, mask

class Gray2Bound(object):
    """ convert mask with grayscale value to boundary """
    def __init__(self, n_classes=3, width=2):
        self.width = width  # boundary width
        self.n_classes = n_classes

    def __call__(self, sample):
        image, gray = sample
        bound = gray2bound(gray, self.n_classes, self.width)

        return image, bound

class Gray2InnerOuterBound(object):
    """ convert mask with grayscale value to inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    def __init__(self, width=2):
        self.width = width  # boundary width

    def __call__(self, sample):
        image, gray = sample
        bound = gray2innerouterbound(gray, self.width)

        return image, bound

class Gray2InnerBound(object):
    """ convert mask with grayscale value to inner bound
        this is to test whether the WHD (weighted Hausdorff Distance) loss works well or not
    """

    def __init__(self, width=1):
        self.width = width  # boundary width

    def __call__(self, sample):
        image, gray = sample
        inner_bound = gray2innerbound(gray, self.width)

        return image, inner_bound


class Gray2OuterBound(object):
    """ convert mask with grayscale value to outer bound
        this is to test whether the WHD (weighted Hausdorff Distance) loss works well or not
    """
    def __init__(self, width=1):
        self.width = width  # boundary width

    def __call__(self, sample):
        image, gray = sample
        outer_bound = gray2outerbound(gray, self.width)

        return image, outer_bound


class Gray2Triple(object):
    """ convert gray-scale image to mask which only marks central part, outline and background """
    def __call__(self, sample):
        image, gray = sample
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray == 76] = 2
        mask[gray == 151] = 2
        mask[gray == 255] = 2
        mask[gray == 29] = 1

        return image, mask

class Mask2Gray(object):
    """ convert 2D image to gray-scale image
        red - 76 : low-density plaque --> 4
        black - 0 : background --> 0
        orange - 151 : calcification --> 3
        white - 255 : Border of the artery (small in healthy patients) --> 2
        blue - 29 : inside of the artery --> 1
    """
    def __call__(self, sample):
        image, mask = sample
        h, w = image.shape[:2]
        gray = np.zeros((h, w), dtype=np.uint8)
        gray[mask == 4] = 76
        gray[mask == 3] = 151
        gray[mask == 2] = 255
        gray[mask == 1] = 29

        return image, gray

class Gray2Binary(object):
    """ convert gray-scale annotation to binary mask """
    def __call__(self, sample):
        image, gray = sample
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray == 151] = 1
        mask[gray == 76] = 1

        return image, mask

class HU2Gray(object):
    def __init__(self, hu_max=1640.0, hu_min=-1024.0):
        self.hu_max = hu_max
        self.hu_min = hu_min
        self.scale = float(255) / (self.hu_max - self.hu_min)

    def __call__(self, sample):
        """ convert HU value to gray scale [0, 255] """
        image, mask = sample
        image = hu2gray(image, self.hu_max, self.hu_min)

        return image, mask

class HU2LUT(object):
    def __init__(self, window, level):
        self.window = window
        self.level = level

    def __call__(self, sample):
        image, mask = sample
        new_image = hu2lut(image, self.window, self.level)

        return new_image, mask

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, sample):
        image, mask = sample
        # swap color axis because (1) numpy image: H x W x C (2) torch image: C X H X W
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[np.newaxis, :, :]

        image = torch.from_numpy(image).float()
        if self.norm:
            image = image / 255.0

        mask = torch.from_numpy(mask).long()

        return image, mask

class HU2GrayMultiStreamToTensor(object):
    """ convert HU value to grayscale for different windows + ToTensor """
    def __init__(self, w_widths = [500.0, 100.0], w_centers = [250.0, 50.0], norm=True):
        self.w_widths = w_widths
        self.w_centers = w_centers
        self.norm = norm

    def __call__(self, sample):
        image, mask = sample
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[np.newaxis, :, :]
        sample_img = []
        for w_w, w_c in zip(self.w_widths, self.w_centers):
            stream = hu2lut(image, w_w, w_c)
            stream = torch.from_numpy(stream).float()
            if self.norm:
                stream = stream / 255.0
            sample_img.append(stream)

        sample_mask = torch.from_numpy(mask).long()

        return sample_img, sample_mask

class Identical(object):
    def __call__(self, sample):
        return sample

class GaussianCrop(object):
    """ crop patches with central pixel position (x, y) obeying Guassian distribution """
    def __init__(self, size, sigma=0.1):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.sigma = sigma

    def __call__(self, sample):
        """ centre crop the given image
        Args:
            sample : (image, mask)
        """
        image, mask = sample
        h, w = image.shape[:2]
        p_h, p_w = self.size

        c_h = int(random.normalvariate(0.5, self.sigma) * h)
        if c_h < p_h // 2:
            c_h = p_h // 2
        elif c_h > h - p_h // 2:
            c_h = h - p_h // 2

        c_w = int(random.normalvariate(0.5, self.sigma) * w)
        if c_w < p_w // 2 :
            c_w = p_w // 2
        elif c_w > w - p_w // 2:
            c_w = w - p_w // 2

        new_image = image[c_h - p_h // 2:c_h + p_h // 2, c_w - p_w // 2:c_w + p_w // 2]
        new_mask = mask[c_h - p_h // 2:c_h + p_h // 2, c_w - p_w // 2:c_w + p_w // 2]

        return new_image, new_mask

class CentralCrop(object):
    """ central crop """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        """ centre crop the given image
        :param sample: (image, mask)
        :param size: int or tuple, new image size
        """
        image, mask = sample
        new_image = central_crop(image, self.size)
        new_mask = central_crop(mask, self.size)

        return new_image, new_mask

class RandomCentralCrop(object):
    """ randomly central crop with given size options """
    def __init__(self, lower_size=192, upper_size=256, step=4):
        assert lower_size%2 ==0 and upper_size%2==0, "both lower and upper size should be even number"
        self.lower_size = lower_size
        self.upper_size = upper_size
        self.step = step

    def __call__(self, sample):
        x = random.randrange(self.lower_size, self.upper_size, self.step)
        return CentralCrop(x)(sample)

class AddNoise(object):
    """ add Gaussian noise to given sample """
    def __init__(self, loc=0.0, scale=1.0, prob=0.5):
        self.loc = loc
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample
        x = random.uniform(0, 1)
        if x <= self.prob:
            noise = np.random.normal(self.loc, self.scale, image.shape)
            image += noise

        return image, mask

class RandomTranslate(object):
    """ random translate the given image """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample
        [H, W] = image.shape[:2]
        x = random.uniform(0, 1)
        if x <= self.prob:
            right = random.randint(int(-W/4), int(W/4))
            down = random.randint(int(H/4), int(H/4))
            M = np.float32([[1, 0, right], [0, 1, down]])
            image = cv2.warpAffine(image, M, (W, H))
            mask = cv2.warpAffine(mask, M, (W, H))

        return image, mask

# composed = transforms.Compose([CentralCrop((128, 128)),
#                                Rescale((388, 388)),
#                                MirrorPadding(((92, 92), (92, 92))),
#                                Gray2Mask(),
#                                ToTensor()])
#
# if __name__ == "__main__":
#     x = np.random.normal(0, 0.5, (100, 100))
#     plt.figure(1)
#     plt.imshow(x)
#
#     y, _ = RandomTranslate()((x, x))
#     plt.figure(2)
#     plt.imshow(y)
#
#     plt.show()