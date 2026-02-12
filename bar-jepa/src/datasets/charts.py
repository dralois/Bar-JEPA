# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json

from logging import getLogger

from PIL import Image

import torch
import torchvision

from torchvision.transforms import PILToTensor

from src.utils.heatmap import cls_pts_to_maps

_GLOBAL_SEED = 0
logger = getLogger()


def make_charts(
    transform,
    batch_size,
    patch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    val_train_split=True,
    decoder_training=True,
    training=True,
    drop_last=True,
    shuffle=False
):
    g = torch.Generator()
    g.manual_seed(_GLOBAL_SEED)

    dataset = Charts(
        patch_size=patch_size,
        root=root_path,
        transform=transform,
        training=training,
        decoder_training=decoder_training)
    logger.info('Chart dataset created')

    def create_sampler_loader(dataset):
        sampler = torch.utils.data.distributed.DistributedSampler( # type: ignore
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers)
        logger.info(f'Chart data loader for {len(dataset)} samples created')
        return loader, sampler

    if val_train_split:
        train, val = torch.utils.data.random_split(dataset, [0.8, 0.2], g)
        train_loader, train_sampler = create_sampler_loader(train)
        val_loader, val_sampler = create_sampler_loader(val)
        return train_loader, train_sampler, val_loader, val_sampler
    else:
        loader, sampler = create_sampler_loader(dataset)
        return loader, sampler


class Charts(torchvision.datasets.DatasetFolder):

    def __init__(
        self,
        patch_size,
        root='data',
        transform=None,
        training=True,
        decoder_training=True
    ):
        """
        Chart dataset loader

        :param root: Root directory for dataset
        :param training: whether to load train or test data
        :param decoder_training: whether to return annotations for decoder training
        """

        image_folder = 'images'
        annotation_folder = 'annotations'
        suffix = 'train' if training else 'test'
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
            self.patch_size = patch_size
            self.transform = transform if transform is not None else PILToTensor()
            self.decoder_training = decoder_training
            self.data_paths = []

            for fname in os.listdir(img_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.splitext(fname)[0]
                    img_full_path = os.path.join(img_path, fname)
                    ann_full_path = os.path.join(ann_path, f"{base_name}.json")

                    if self.decoder_training:
                        if not os.path.exists(ann_full_path):
                            raise FileNotFoundError(f"Annotation file not found for image: {fname}")
                        self.data_paths.append((img_full_path, ann_full_path))
                    else:
                        self.data_paths.append((img_full_path, ann_full_path if os.path.exists(ann_full_path) else None))

            logger.info(f'Loaded {len(self.data_paths)} {"training" if training else "test"} images')

        except FileNotFoundError:
            raise ValueError(f'Number of images and annotations do not match')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx][0]
        ann_path = self.data_paths[idx][1]

        # -- Image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # If not decoder training, ignore annotations
        if not self.decoder_training:
            return img, 0

        # -- Annotations
        ann = json.load(open(ann_path))

        # Coordinate system origin is normalized
        size = torch.tensor(ann['chart_metadata']['size']['bbox'][2:])
        org = (torch.tensor(ann['chart_metadata']['origin']['bbox'][:2]) / size).flip(-1)

        ticks = []
        # Ticks are normalized x,y coordinates of the tick location
        for tick in ann['data']['value_axis']['ticks']:
            ticks.append((torch.tensor(tick['bbox'][:2]) / size).flip(-1))

        bars = []
        # Bars are normalized x,y coordinates of a bar's top right corner
        for feature in ann['data']['features']:
            for bar in feature['data']:
                bars.append((torch.tensor([bar['bbox'][2], bar['bbox'][1]]) / size).flip(-1))

        # Map size depends on image size
        mapsize = (torch.tensor(img.shape[1:3]) // self.patch_size) * 4
        # Generate class and regression maps
        gt_org, gt_cls, gt_reg = cls_pts_to_maps([bars, ticks], org, mapsize)

        return img, (gt_org, gt_cls, gt_reg)
