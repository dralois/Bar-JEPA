from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


class ChartsCollator(object):

    def __call__(self, batch):
        # [[img, [org, cls, reg]]]
        imgs = [img for img, _ in batch]
        gt_org = [t[0] for _, t in batch]
        gt_cls = [t[1] for _, t in batch]
        gt_reg = [t[2] for _, t in batch]
        return imgs, (gt_org, gt_cls, gt_reg)


class UBPMCCollator(object):

    def __call__(self, batch):
        # [[img, [org, cls, reg]]]
        imgs = [img for img, _ in batch]
        gt_org = [t[0] for _, t in batch]
        gt_cls = [t[1] for _, t in batch]
        gt_reg = [t[2] for _, t in batch]
        return imgs, (gt_org, gt_cls, gt_reg)
