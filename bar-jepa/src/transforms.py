# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from math import floor

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    random_resize_crop=True,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
    preserve_aspect_ratio=True,
    max_patches=14*14,
    patch_size=16
):
    logger.info('making data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []

    if preserve_aspect_ratio:
        if random_resize_crop:
            transform_list.append(RandomResizedCropARP(scale=crop_scale))
        transform_list.append(
            ResizeToFixedPatches(max_patches=max_patches, patch_size=patch_size)
        )
    else:
        if random_resize_crop:
            transform_list.append(transforms.RandomResizedCrop(crop_size, scale=crop_scale))
        else:
            transform_list.append(transforms.Resize((crop_size, crop_size)))

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if color_distortion:
        transform_list.append(get_color_distortion(s=color_jitter))
    if gaussian_blur:
        transform_list.append(GaussianBlur(p=0.5))

    mean, std = normalization
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    """
    Apply Gaussian blur with a given probability.

    :param p: Probability of applying blur
    :param radius_min: Minimum blur radius
    :param radius_max: Maximum blur radius
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class ResizeToFixedPatches(object):
    """
    Resize an image so that its patch grid fits within max_patches.

    :param max_patches: Maximum number of patches (H*W)
    :param patch_size: Patch size in pixels
    """
    def __init__(self, max_patches=14*14, patch_size=16):
        self.max_patches = int(max_patches)
        self.patch_size = int(patch_size)

    def __call__(self, img):
        image_width, image_height = img.size

        scale = (self.max_patches * (self.patch_size / image_height) * (self.patch_size / image_width)) ** 0.5
        scale += 1e-6

        num_feasible_rows = max(min(floor(scale * image_height / self.patch_size), self.max_patches), 1)
        num_feasible_cols = max(min(floor(scale * image_width / self.patch_size), self.max_patches), 1)

        resized_height = max(int(num_feasible_rows * self.patch_size), 1)
        resized_width = max(int(num_feasible_cols * self.patch_size), 1)

        return F.resize(
            img,
            (resized_height, resized_width),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )


class RandomResizedCropARP(object):
    """
    Random resized crop that preserves the original aspect ratio.

    Uses torchvision's RandomResizedCrop sampling logic with ratio fixed
    to the input image's aspect ratio.

    :param scale: Range of area scaling for the crop
    """
    def __init__(self, scale=(0.3, 1.0)):
        self.scale = scale

    def __call__(self, img):
        image_width, image_height = img.size
        if image_width <= 0 or image_height <= 0:
            return img

        ratio = image_width / image_height
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img,
            scale=self.scale,
            ratio=(ratio, ratio)
        )
        return F.crop(img, i, j, h, w)
