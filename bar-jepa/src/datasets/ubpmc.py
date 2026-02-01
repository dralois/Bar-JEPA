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
        transform=transform)

    def create_sampler_loader(dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
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
        self.data_paths = []

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

                task1 = ann.get('task1') or {}
                task1_out = task1.get('output') or {}
                if task1_out.get('chart_type') != 'vertical bar':
                    continue

                if 'task6' not in ann or 'task4' not in ann:
                    continue

                task4 = ann.get('task4') or {}
                task4_out = task4.get('output') or {}
                axes = task4_out.get('axes') or {}
                y_axis = axes.get('y-axis', [])
                plot_bb = task4_out.get('_plot_bb')
                task6 = ann.get('task6') or {}
                task6_out = task6.get('output') or {}
                bars = (task6_out.get('visual elements') or {}).get('bars', [])

                if not plot_bb or not y_axis or not bars:
                    continue

                self.data_paths.append((img_full_path, ann_full_path))

        logger.info(f'Loaded {len(self.data_paths)} images')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        img_path = self.data_paths[idx][0]
        ann_path = self.data_paths[idx][1]

        img = Image.open(img_path).convert('RGB')
        size = torch.tensor(img.size)
        img = self.transform(img)

        with open(ann_path, 'r') as f:
            ann = json.load(f)
        if not isinstance(ann, dict):
            raise ValueError(f'Invalid annotation (not a dict): {ann_path}')

        # Extract ticks from task4
        task4 = ann.get('task4') or {}
        task4_out = task4.get('output') or {}
        axes = task4_out.get('axes') or {}
        plot_bb = task4_out.get('_plot_bb')

        # Extract bars from task6
        task6 = ann.get('task6') or {}
        task6_out = task6.get('output') or {}
        bars = (task6_out.get('visual elements') or {}).get('bars', [])

        # Find origin y using the '0' tick label, snapping to nearest y-axis tick
        origin_y = None
        task2 = ann.get('task2') or {}
        task2_out = task2.get('output') or {}
        text_blocks = task2_out.get('text_blocks', [])
        task3 = ann.get('task3') or {}
        task3_out = task3.get('output') or {}
        text_roles = task3_out.get('text_roles', [])
        role_by_id = {entry.get('id'): entry.get('role') for entry in text_roles}
        zero_blocks = []
        for block in text_blocks:
            if role_by_id.get(block.get('id')) != 'tick_label':
                continue
            text = (block.get('text') or '').strip()
            if text in {'0', '0.0', '0.00'}:
                zero_blocks.append(block)
        if zero_blocks and axes.get('y-axis'):
            block = zero_blocks[0]
            poly = block.get('polygon') or {}
            ys = [poly.get(k) for k in ('y0', 'y1', 'y2', 'y3')]
            ys = [y for y in ys if isinstance(y, (int, float))]
            if ys:
                label_y = sum(ys) / len(ys)
                origin_tick = min(
                    axes['y-axis'],
                    key=lambda t: abs(t.get('tick_pt', {}).get('y', label_y) - label_y)
                )
                origin_y = origin_tick.get('tick_pt', {}).get('y')
        if origin_y is None and axes.get('y-axis'):
            origin_y = max(t.get('tick_pt', {}).get('y', 0) for t in axes['y-axis'])

        # Normalize axes tick points (sorted bottom -> top)
        normalized_y_axis = []
        for tick in sorted(axes['y-axis'], key=lambda t: t['tick_pt']['y'], reverse=True):
            normalized_y_axis.append(
                (torch.tensor([tick['tick_pt']['x'], tick['tick_pt']['y']]) / size).flip(-1)
            )

        # Normalize bars
        normalized_bars = []
        for bar in sorted(bars, key=lambda b: b['x0']):
            y = bar['y0']
            if origin_y is not None and y > origin_y:
                y = bar['y0'] + bar['height']
            top_right = torch.tensor([bar['x0'] + bar['width'], y])
            normalized_bars.append((top_right / size).flip(-1))

        # Normalize plot bounding box origin
        org_y = origin_y if origin_y is not None else plot_bb['y0']
        org = (torch.tensor([plot_bb['x0'], org_y]) / size).flip(-1)

        # Prepare ticks and bars
        ticks = normalized_y_axis
        bars = normalized_bars
        org_t = org

        # Map size depends on image size
        mapsize = (torch.tensor(img.shape[1:3]) // self.patch_size) * 4
        mapsize = torch.clamp(mapsize, min=1)
        # Generate class and regression maps
        gt_org, gt_cls, gt_reg = cls_pts_to_maps([bars, ticks], org_t, mapsize)

        return img, (gt_org, gt_cls, gt_reg)
