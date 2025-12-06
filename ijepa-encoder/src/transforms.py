# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from math import floor

from PIL import ImageFilter
from PIL.Image import Image

import torch
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
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
        transform_list += [ResizeToFixedPatches(max_patches=max_patches, patch_size=patch_size)]
    else:
        transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
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
    def __init__(self, max_patches=14*14, patch_size=16):
        self.max_patches = int(max_patches)
        self.patch_size = int(patch_size)

    def __call__(self, img):
        image_height, image_width = img.size

        scale = (self.max_patches * (self.patch_size / image_height) * (self.patch_size / image_width)) ** 0.5

        num_feasible_rows = max(min(floor(scale * image_height / self.patch_size), self.max_patches), 1)
        num_feasible_cols = max(min(floor(scale * image_width / self.patch_size), self.max_patches), 1)

        resized_height = max(int(num_feasible_rows * self.patch_size), 1)
        resized_width = max(int(num_feasible_cols * self.patch_size), 1)

        return transforms.Resize(
            (resized_height, resized_width),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )(img)