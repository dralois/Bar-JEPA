# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json

import numpy as np

from logging import getLogger

from PIL import Image

import torch
import torchvision

from src.utils.heatmap import cls_pts_to_map

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
    annotation_folder=None,
    training=True,
    drop_last=True
):
    dataset = Charts(
        root=root_path,
        image_folder=image_folder,
        annotation_folder=annotation_folder,
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
        root='data',
        image_folder='images',
        annotation_folder='annotations',
        transform=None,
        train=True
    ):
        """
        Chart dataset loader

        :param root: Root directory for dataset
        :param image_folder: Path to images inside root directory
        :param annotation_folder: Path to annotations inside root directory
        :param train: whether to load train or test data
        """

        suffix = 'train' if train else 'test'
        img_path = os.path.join(root, suffix, image_folder)
        ann_path = os.path.join(root, suffix, annotation_folder)
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            suffix = ''
        img_path = os.path.join(root, suffix, image_folder)
        ann_path = os.path.join(root, suffix, annotation_folder)
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            raise FileNotFoundError(f'Path {img_path} / {ann_path} does not exist')
        logger.info(f'Loading data from {img_path} / {ann_path}')

        try:
            self.data_paths = []
            for fname in os.listdir(img_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.splitext(fname)[0]
                    img_full_path = os.path.join(img_path, fname)
                    ann_full_path = os.path.join(ann_path, f"{base_name}.json")

                    if not os.path.exists(ann_full_path):
                        raise FileNotFoundError(f"Annotation file not found for image: {fname}")
                    self.data_paths.append((img_full_path, ann_full_path))

            self.transform = transform
            logger.info(f'Loaded {len(self.data_paths)} {"training" if train else "test"} images')
        except FileNotFoundError:
            raise ValueError(f'Number of images and annotations do not match')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx][0]
        ann_path = self.data_paths[idx][1]

        # -- Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # -- Annotations
        ann = json.load(open(ann_path))

        # Coordinate system origin is normalized
        size = torch.tensor([float(v) for v in ann['chart_metadata']['size']['bbox'][2:]])
        gt_org = torch.tensor([float(v) for v in ann['chart_metadata']['origin']['bbox'][:2]]) / size

        ticks = []
        # Ticks are normalized x,y coordinates of the tick location
        for tick in ann['data']['value_axis']['ticks']:
            ticks.append(torch.tensor([float(v) for v in tick['bbox'][:2]]) / size)

        bars = []
        # Bars are normalized x,y coordinates of a bar's top left corner
        for feature in ann['data']['features']:
            for bar in feature['data']:
                bars.append(torch.tensor([float(v) for v in bar['bbox'][:2]]) / size)

        # Map size depends on image size
        mapsize = torch.tensor(img.shape[1:3]) // 4
        # Generate class and regression maps
        gt_cls, gt_reg = cls_pts_to_map([bars, ticks], mapsize)

        return img, (gt_org, gt_cls, gt_reg)
