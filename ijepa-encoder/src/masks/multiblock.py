# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        patch_size=16,
        patch_count=14*14,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        input_size,
        generator,
        scale,
        aspect_ratio_scale
    ):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        input_h, input_w = input_size
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(input_h * input_w * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        block_h = int(round(math.sqrt(max_keep * aspect_ratio)))
        block_w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while block_h > input_h:
            block_h -= 1
        while block_w > input_w:
            block_w -= 1

        return (block_h, block_w)

    def _sample_block_mask(
        self,
        input_size,
        block_size,
        acceptable_regions=None
    ):
        block_h, block_w = block_size
        input_h, input_w = input_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, input_h - block_h + 1, (1,))
            left = torch.randint(0, input_w - block_w + 1, (1,))
            mask = torch.zeros((input_h, input_w), dtype=torch.int32)
            mask[top:top+block_h, left:left+block_w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((input_h, input_w), dtype=torch.int32)
        mask_complement[top:top+block_h, left:left+block_w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        #collated_batch = torch.utils.data.default_collate(batch)
        collated_batch = [img for img, _ in batch]

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = float('inf')
        min_keep_enc = float('inf')

        for img, _ in batch:
            # B, C, W, H
            img_s = (img.shape[-2] // self.patch_size, img.shape[-1] // self.patch_size)

            p_size = self._sample_block_size(
                input_size=img_s,
                generator=g,
                scale=self.pred_mask_scale,
                aspect_ratio_scale=self.aspect_ratio)
            e_size = self._sample_block_size(
                input_size=img_s,
                generator=g,
                scale=self.enc_mask_scale,
                aspect_ratio_scale=(1., 1.))

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(img_s, p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e, masks_C_e = [], []
            for _ in range(self.nenc):
                mask, mask_C = self._sample_block_mask(img_s, e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                masks_C_e.append(mask_C)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # TODO: What happens here?
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

