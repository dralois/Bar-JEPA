import csv
import os

from logging import getLogger
from PIL import Image

import torch
from torchvision.transforms import PILToTensor

_GLOBAL_SEED = 0
logger = getLogger()


def make_ctt(
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
    drop_last=True,
    shuffle=False,
    include_multicol=True,
):
    g = torch.Generator()
    g.manual_seed(_GLOBAL_SEED)

    dataset = CTTDataset(
        patch_size=patch_size,
        root=root_path,
        transform=transform,
        include_multicol=include_multicol)
    logger.info('CTT dataset created')

    def create_sampler_loader(ds):
        sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)
        loader = torch.utils.data.DataLoader(
            ds,
            collate_fn=collator,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers)
        logger.info(f'CTT data loader for {len(ds)} samples created')
        return loader, sampler

    if val_train_split:
        train, val = torch.utils.data.random_split(dataset, [0.8, 0.2], g)
        train_loader, train_sampler = create_sampler_loader(train)
        val_loader, val_sampler = create_sampler_loader(val)
        return train_loader, train_sampler, val_loader, val_sampler
    else:
        loader, sampler = create_sampler_loader(dataset)
        return loader, sampler


class CTTDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        patch_size,
        root='CTT',
        transform=None,
        include_multicol=True,
    ):
        """
        Chart-to-Text dataset loader, combining Statista & Pew datasets

        :param root: root directory for dataset
        :param include_multicol: whether to include multi-column charts
        """

        if not os.path.exists(root):
            raise FileNotFoundError(f'CTT root not found: {root}')

        self.patch_size = patch_size
        self.transform = transform if transform is not None else PILToTensor()

        statista_imgs = self._collect_statista_imgs(root, include_multicol)
        pew_imgs = self._collect_pew_imgs(root, include_multicol)
        self.img_paths = statista_imgs + pew_imgs

        logger.info(
            f'CTT dataset: {len(statista_imgs)} Statista + {len(pew_imgs)} Pew '
            f'= {len(self.img_paths)} images total'
        )

    @staticmethod
    def _collect_statista_imgs(
        root: str,
        include_multicol: bool
    ) -> list[str]:
        """
        Returns paths to bar-type Statista chart images, filtered via metadata.csv.
        """
        statista_root = os.path.join(root, 'statista_dataset', 'dataset')
        folders = [statista_root]
        if include_multicol:
            folders.append(os.path.join(statista_root, 'multiColumn'))

        paths = []
        for folder in folders:
            meta_path = os.path.join(folder, 'metadata.csv')
            imgs_dir = os.path.join(folder, 'imgs')
            if not os.path.exists(meta_path) or not os.path.exists(imgs_dir):
                continue
            with open(meta_path, newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    if row.get('chartType', '').strip().lower() != 'bar':
                        continue
                    img_name = os.path.basename(row.get('imgPath', '').strip())
                    img_path = os.path.join(imgs_dir, img_name)
                    if os.path.exists(img_path):
                        paths.append(img_path)

        return paths

    @staticmethod
    def _collect_pew_imgs(
        root: str,
        include_multicol: bool
    ) -> list[str]:
        """
        Returns paths to bar-type Pew chart images, filtered via metadata.csv.
        """
        pew_root = os.path.join(root, 'pew_dataset', 'dataset')
        paths = []

        meta_files = [os.path.join(pew_root, 'metadata.csv')]
        if include_multicol:
            meta_files.append(os.path.join(pew_root, 'multiColumn', 'metadata.csv'))

        for meta_path in meta_files:
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    if row.get('chartType', '').strip().lower() != 'bar':
                        continue
                    img_path = os.path.join(pew_root, row.get('imgPath', '').strip())
                    if os.path.exists(img_path):
                        paths.append(img_path)

        return paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_pil = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img_pil)
        return img, 0
