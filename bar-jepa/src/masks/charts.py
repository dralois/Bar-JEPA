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
        # [[img, full_img, [org, bar_yxv, tick_yxv]]]
        imgs = [item[0] for item in batch]
        full_imgs = [item[1] for item in batch]
        gt_org = [item[2][0] for item in batch]
        gt_bar_yxv = [item[2][1] for item in batch]
        gt_tick_yxv = [item[2][2] for item in batch]
        return imgs, full_imgs, (gt_org, gt_bar_yxv, gt_tick_yxv)
