from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


class ChartsCollator(object):

    def __call__(self, batch):
        # [[img, [org, cls, reg, ...]]]
        imgs = [item[0] for item in batch]
        targets = tuple(
            [item[1][i] for item in batch]
            for i in range(len(batch[0][1]))
        )
        return imgs, targets


class EvalCollator(object):

    def __call__(self, batch):
        # [[img, full_img, [org, bars, ticks, bar_values]]]
        imgs = [item[0] for item in batch]
        full_imgs = [item[1] for item in batch]
        if len(batch[0][2]) != 4:
            raise ValueError('EvalCollator expects targets as (org, bars, ticks, bar_values).')
        gt_org = [item[2][0] for item in batch]
        gt_bars = [item[2][1] for item in batch]
        gt_ticks = [item[2][2] for item in batch]
        gt_bar_values = [item[2][3] for item in batch]

        return imgs, full_imgs, (gt_org, gt_bars, gt_ticks, gt_bar_values)
