from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


class ChartsCollator(object):

    def __call__(self, batch):
        # [[img, full_img, [org, cls, reg]]]
        imgs = [item[0] for item in batch]
        full_imgs = [item[1] for item in batch]
        gt_org = [item[2][0] for item in batch]
        gt_cls = [item[2][1] for item in batch]
        gt_reg = [item[2][2] for item in batch]
        return imgs, full_imgs, (gt_org, gt_cls, gt_reg)


class UBPMCCollator(object):

    def __call__(self, batch):
        # [[img, full_img, [org, cls, reg]]]
        imgs = [item[0] for item in batch]
        full_imgs = [item[1] for item in batch]
        gt_org = [item[2][0] for item in batch]
        gt_cls = [item[2][1] for item in batch]
        gt_reg = [item[2][2] for item in batch]
        return imgs, full_imgs, (gt_org, gt_cls, gt_reg)
