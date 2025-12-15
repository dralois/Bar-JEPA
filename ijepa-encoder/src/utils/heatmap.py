import math
import torch
import torch.nn as nn
import torch.nn.functional as F

id_to_name = {0: 'None', 1: 'bar', 2: 'tick'}
name_to_id = {'None': 0, 'bar': 1, 'tick': 2}

def pts_map_to_lists(pts_cls_map, pts_reg_map):
    bars = [[] for im in range(pts_cls_map.shape[0])]
    ticks = [[] for im in range(pts_cls_map.shape[0])]

    classes = pts_cls_map
    pts_im, pts_x, pts_y = torch.nonzero(classes, as_tuple=True)

    for im, x, y in zip(pts_im, pts_x, pts_y):
        cls = classes[im, x, y]
        pos_x = (x.float() * 2 + 1) / (classes.shape[2] * 2)
        pos_y = (y.float() * 2 + 1) / (classes.shape[3] * 2)

        reg = pts_reg_map[im, :, x, y]
        pos_x += reg[0] / classes.shape[2]
        pos_y += reg[1] / classes.shape[3]

        if cls == name_to_id['bar']:
            bars[im].append((pos_x, pos_y))
        elif cls == name_to_id['tick']:
            ticks[im].append((pos_x, pos_y))

    return bars, ticks

def get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh=0.8, conf_thresh=0.5):
    bars = [[] for im in range(pred_cls_map.shape[0])]
    ticks = [[] for im in range(pred_cls_map.shape[0])]

    pts_mask = torch.sigmoid(pred_cls_map[:, 0]).lt(pt_thresh)

    masked_bars = (torch.sigmoid(pred_cls_map[:, 1]) * pts_mask).gt(conf_thresh)
    b_im, b_x, b_y = torch.nonzero(masked_bars, as_tuple=True)
    for im, x, y in zip(b_im, b_x, b_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[3] * 2)
        reg = pred_reg_map[im, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[2] * 2)
        pos_y += reg[1] / (pts_mask.shape[3] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 1, x, y])
        bars[im].append((pos_y, pos_x, conf))

    masked_ticks = (torch.sigmoid(pred_cls_map[:, 2]) * pts_mask).gt(conf_thresh)
    t_im, t_x, t_y = torch.nonzero(masked_ticks, as_tuple=True)
    for im, x, y in zip(t_im, t_x, t_y):
        pos_x = (x.float() * 2 + 1) / (pts_mask.shape[2] * 2)
        pos_y = (y.float() * 2 + 1) / (pts_mask.shape[3] * 2)
        reg = pred_reg_map[im, :, x, y]
        pos_x += reg[0] / (pts_mask.shape[2] * 2)
        pos_y += reg[1] / (pts_mask.shape[3] * 2)
        conf = torch.sigmoid(pred_cls_map[im, 2, x, y])
        ticks[im].append((pos_y, pos_x, conf))

    return bars, ticks

def nms(bars, ticks, thresh=1.5 / 56):
    nms_bars = [[] for im in range(len(bars))]
    nms_ticks = [[] for im in range(len(ticks))]

    for im, im_pts in enumerate(bars):
        sorted_impts = sorted(im_pts, key=lambda p: -p[2])
        checked_pts = []
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i, p in enumerate(sorted_impts)
                             if abs(p[0] - pt[0]) + abs(p[1] - pt[1]) / 5 < thresh
                             if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_bars[im].append(pt)

    for im, im_pts in enumerate(ticks):
        sorted_impts = sorted(im_pts, key=lambda p: -p[2])
        checked_pts = []
        for k, pt in enumerate(sorted_impts):
            if k in checked_pts:
                continue
            neighbors_inds = [i for i, p in enumerate(sorted_impts)
                             if abs(p[0] - pt[0]) + abs(p[1] - pt[1]) < thresh
                             if i not in checked_pts]
            checked_pts.extend(neighbors_inds)
            nms_ticks[im].append(pt)

    return nms_bars, nms_ticks
