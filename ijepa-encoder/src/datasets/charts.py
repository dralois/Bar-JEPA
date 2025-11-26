# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import numpy as np

from logging import getLogger

from PIL import Image

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_charts(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True
):
    dataset = Charts(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training)
    logger.info('Chart dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Chart unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class Charts(torchvision.datasets.DatasetFolder):

    def __init__(
        self,
        root,
        image_folder='data',
        transform=None,
        train=True
    ):
        """
        Chart dataset loader

        :param root: root network directory for chart data
        :param image_folder: path to images inside root network directory
        :param train: whether to load train data (or validation)
        """

        suffix = 'train/images' if train else 'train/images'
        data_path = os.path.join(root, image_folder, suffix)
        if not os.path.exists(data_path):
            suffix = ''
        data_path = os.path.join(root, image_folder, suffix)
        if not os.path.exists(data_path):
            raise ValueError(f'Datapath {data_path} does not exist')
        logger.info(f'data-path {data_path}')

        self.transform = transform
        self.image_paths = [
            os.path.join(data_path, fname)
            for fname in os.listdir(data_path)
            if fname.lower().endswith(['.png', '.jpg', '.jpeg'])
        ]
        logger.info(f'Loaded {len(self.image_paths)} images from {data_path}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0
