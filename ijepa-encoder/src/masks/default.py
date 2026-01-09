# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


class DefaultCollator(object):

    def __call__(self, batch):
        # [[img, [org, cls, reg]]]
        imgs = [img for img, _ in batch]
        gt_org = [t[0] for _, t in batch]
        gt_cls = [t[1] for _, t in batch]
        gt_reg = [t[2] for _, t in batch]
        return imgs, (gt_org, gt_cls, gt_reg)
