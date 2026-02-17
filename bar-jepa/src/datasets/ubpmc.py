import os
import json
import numpy as np

from logging import getLogger

from PIL import Image

import torch
import torchvision

from torchvision.transforms import PILToTensor

from src.utils.heatmap import cls_pts_to_maps
from src.utils.numeric import extract_numeric_value

_GLOBAL_SEED = 0
logger = getLogger()


def make_ubpmc(
    transform,
    batch_size,
    patch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    split=None,
    training=True,
    eval_mode=False,
    val_train_split=True,
    drop_last=True,
    shuffle=False
):
    g = torch.Generator()
    g.manual_seed(_GLOBAL_SEED)

    dataset = UBPMCDataset(
        patch_size=patch_size,
        root=root_path,
        training=training,
        split=split,
        transform=transform,
        eval_mode=eval_mode)

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
        logger.info(f'Custom data loader for {len(dataset)} samples created')
        return loader, sampler

    if val_train_split:
        train, val = torch.utils.data.random_split(dataset, [0.8, 0.2], g)
        train_loader, train_sampler = create_sampler_loader(train)
        val_loader, val_sampler = create_sampler_loader(val)
        return train_loader, train_sampler, val_loader, val_sampler
    else:
        loader, sampler = create_sampler_loader(dataset)
        return loader, sampler


class UBPMCDataset(torchvision.datasets.DatasetFolder):

    def __init__(
        self,
        patch_size,
        root='UBPMC',
        training=True,
        split=None,
        transform=None,
        eval_mode=False,
    ):
        """
        UB PMC dataset loader

        :param root: Root directory for dataset
        :param training: whether to load train or test data
        :param split: in test mode: which test split to load
        :param decoder_training: whether to return annotations for decoder training
        """
        if training:
            image_folder = os.path.join('train', 'images', 'vertical_bar')
            annotation_folder = os.path.join('train', 'annotations_JSON', 'vertical_bar')
        else:
            split = split or 'split_4'
            image_folder = os.path.join('test', 'chart_images', split)
            annotation_folder = os.path.join('test', 'final_full_GT', split, 'annotations_JSON')

        img_path = os.path.join(root, image_folder)
        ann_path = os.path.join(root, annotation_folder)
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            raise FileNotFoundError(f'Path {img_path} / {ann_path} does not exist')
        logger.info(f'Loading data from {img_path} / {ann_path}')

        self.patch_size = patch_size
        self.transform = transform if transform is not None else PILToTensor()
        self.eval_mode = eval_mode
        self.data_paths = []

        # Filter out incompatible charts (missing gt)
        for fname in os.listdir(img_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(fname)[0]
                img_full_path = os.path.join(img_path, fname)
                ann_full_path = os.path.join(ann_path, f"{base_name}.json")

                if not os.path.exists(ann_full_path):
                    continue

                with open(ann_full_path, 'r') as f:
                    ann = json.load(f)

                if not isinstance(ann, dict):
                    continue

                is_vertical = (ann.get('task1') or {}).get('output', {}).get('chart_type') == 'vertical bar'
                y_axis = (ann.get('task4') or {}).get('output', {}).get('axes', {}).get('y-axis', [])
                bars = (ann.get('task6') or {}).get('output', {}).get('visual elements', {}).get('bars', [])

                if not is_vertical or not y_axis or not bars:
                    continue

                self.data_paths.append((img_full_path, ann_full_path))

        logger.info(f'Loaded {len(self.data_paths)} images')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        img_path = self.data_paths[idx][0]
        ann_path = self.data_paths[idx][1]

        img_pil = Image.open(img_path).convert('RGB')
        size = torch.tensor(img_pil.size)
        img = self.transform(img_pil)

        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # Extract ticks and bars
        axes = ann.get('task4', {}).get('output', {}).get('axes', {})
        bars_ann = ann.get('task6', {}).get('output', {}).get('visual elements', {}).get('bars', [])
        data_series = ann.get('task6', {}).get('output', {}).get('data series', [])

        # Flatten y-values from task6 data series.
        raw_bar_values = []
        for series in data_series:
            for point in (series.get('data') or []):
                y_val = point.get('y')
                raw_bar_values.append(
                    float(y_val) if isinstance(y_val, (int, float)) else float('nan')
                )

        # Determine origin tick from y-axis ticks
        origin_tick_pt = None
        text_blocks = ann.get('task2', {}).get('output', {}).get('text_blocks', [])
        role_by_id = {
            entry.get('id'): entry.get('role')
            for entry in ann.get('task3', {}).get('output', {}).get('text_roles', [])
            if entry.get('id') is not None
        }

        # Map block IDs to their parsed numeric values (for tick labels)
        tick_value_by_id = {}
        for block in text_blocks:
            block_id = block.get('id')
            if block_id is None:
                continue
            if role_by_id and role_by_id.get(block_id) != 'tick_label':
                continue
            value = extract_numeric_value(str(block.get('text', '')))
            if value is not None:
                tick_value_by_id[block_id] = value

        # Extract y-axis ticks with valid coordinates and numeric values
        y_axis_ticks = axes.get('y-axis', [])
        tick_candidates = [
            (tick_value_by_id[tick.get('id')], float(tick.get('tick_pt', {}).get('x')), float(tick.get('tick_pt', {}).get('y')))
            for tick in y_axis_ticks
            if (tick.get('id') in tick_value_by_id and
                isinstance(tick.get('tick_pt', {}).get('x'), (int, float)) and
                isinstance(tick.get('tick_pt', {}).get('y'), (int, float)))
        ]

        # Select origin tick point:
        # 1. Prefer a tick with value 0 (if exists)
        # 2. Otherwise, use the tick with the smallest absolute value
        if tick_candidates:
            zero_candidates = [entry for entry in tick_candidates if abs(entry[0]) < 1e-6]
            selected = zero_candidates[0] if zero_candidates else min(tick_candidates, key=lambda p: p[0])
            origin_tick_pt = (selected[1], selected[2])

        # Normalize axes tick points (sorted bottom -> top)
        ticks = []
        for tick in sorted(axes['y-axis'], key=lambda t: t['tick_pt']['y'], reverse=True):
            ticks.append(
                (torch.tensor([tick['tick_pt']['x'], tick['tick_pt']['y']]) / size).flip(-1)
            )

        # Normalize bars (and choose bottom right corner for negative bars)
        bar_entries = []
        for i, bar in enumerate(bars_ann):
            y = bar['y0']
            if origin_tick_pt is not None and y > origin_tick_pt[1]:
                y = bar['y0'] + bar['height']
            top_right = torch.tensor([bar['x0'] + bar['width'], y])
            point = (top_right / size).flip(-1).clamp(0.0, 1.0)
            bar_entries.append({
                'x0': bar['x0'],
                'point': point,
                'value': raw_bar_values[i] if i < len(raw_bar_values) else float('nan')
            })

        bar_entries.sort(key=lambda b: b['x0'])
        bars = [entry['point'] for entry in bar_entries]
        bar_values = [entry['value'] for entry in bar_entries]

        # Normalize coordinate origin directly from the resolved y-axis tick.
        org = (torch.tensor([origin_tick_pt[0], origin_tick_pt[1]]) / size).flip(-1) # type: ignore

        # For evaluation, don't convert to maps
        if self.eval_mode:
            gt_bars = torch.stack(bars)
            gt_ticks = torch.stack(ticks)
            gt_bar_values = torch.tensor(bar_values)
            # OCR expects uint8 RGB, HWC.
            full_img = np.array(img_pil, dtype=np.uint8, copy=True)
            return img, full_img, (org, gt_bars, gt_ticks, gt_bar_values)

        # Map size depends on image size
        mapsize = (torch.tensor(img.shape[1:3]) // self.patch_size) * 4
        mapsize = torch.clamp(mapsize, min=1)
        # Generate class and regression maps
        gt_org, gt_cls, gt_reg = cls_pts_to_maps([bars, ticks], org, mapsize)

        return img, (gt_org, gt_cls, gt_reg)
