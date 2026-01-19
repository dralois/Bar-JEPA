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

def make_custom_dataset(
    transform,
    batch_size,
    patch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    annotation_folder=None,
    drop_last=True,
    shuffle=False
):
    g = torch.Generator()
    g.manual_seed(_GLOBAL_SEED)

    dataset = UBPMCDataset(
        patch_size=patch_size,
        root=root_path,
        image_folder=image_folder,
        annotation_folder=annotation_folder,
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

    loader, sampler = create_sampler_loader(dataset)
    return loader, sampler

class UBPMCDataset(torchvision.datasets.DatasetFolder):

    def __init__(
        self,
        patch_size,
        root='data',
        image_folder='images',
        annotation_folder='annotations',
        transform=None,
    ):
        suffix = 'train'
        img_path = os.path.join(root, suffix, image_folder)
        ann_path = os.path.join(root, suffix, annotation_folder)
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            suffix = ''
        img_path = os.path.join(root, suffix, image_folder)
        ann_path = os.path.join(root, suffix, annotation_folder)
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

                if 'task6' not in ann:
                    continue

                self.data_paths.append((img_full_path, ann_full_path))

        logger.info(f'Loaded {len(self.data_paths)} images')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        img_path = self.data_paths[idx][0]
        ann_path = self.data_paths[idx][1]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        with open(ann_path, 'r') as f:
            ann = json.load(f)

        size = torch.tensor([img.shape[2], img.shape[1]])

        # Extract ticks from task4
        axes = ann['task4']['output']['axes']
        plot_bb = ann['task4']['output']['_plot_bb']

        # Extract bars from task6
        bars = ann['task6']['output']['visual elements']['bars']

        # Normalize coordinates
        def normalize_coords(coords, size):
            return (torch.tensor(coords) / size).flip(-1)

        # Normalize axes tick points
        normalized_y_axis = []
        for tick in axes['y-axis']:
            normalized_tick_pt = {
                'x': tick['tick_pt']['x'] / size[0],
                'y': tick['tick_pt']['y'] / size[1]
            }
            normalized_y_axis.append(normalized_tick_pt)

        # Normalize bars
        normalized_bars = []
        for bar in bars:
            normalized_bar = {
                'x0': bar['x0'] / size[0],
                'y0': bar['y0'] / size[1],
                'width': bar['width'] / size[0],
                'height': bar['height'] / size[1]
            }
            normalized_bars.append(normalized_bar)

        # Normalize plot bounding box origin
        org = {
            'x': plot_bb['x0'] / size[0],
            'y': plot_bb['y0'] / size[1]
        }

        # Prepare ticks and bars
        ticks = [torch.tensor([tick['x'], tick['y']]) for tick in normalized_y_axis]
        bars = [torch.tensor([bar['x0'] + bar['width'], bar['y0']]) for bar in normalized_bars]
        org_t = torch.tensor([org['x'], org['y']])

        # Map size depends on image size
        mapsize = (torch.tensor(img.shape[1:3]) // self.patch_size) * 4
        # Generate class and regression maps
        gt_org, gt_cls, gt_reg = cls_pts_to_maps([bars, ticks], org_t, mapsize)

        return img, (gt_org, gt_cls, gt_reg)

