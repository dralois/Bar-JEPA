import torch
import numpy as np

def cls_pts_to_map(cls_pts_lists, mapsize):
    """
    Converts a list of points of different classes
    to maps of point classes and regression values.

    :param cls_pts_lists: List of lists containing normalized points for each class.
    :param mapsize: (h, w) size of the output maps.
    :return: Tuple containing map with class IDs and x and y regression values (offsets).
    """

    # Make maps of size [H, W], [2, H, W]
    cls_map = torch.zeros((mapsize[0], mapsize[1]), dtype=torch.int32)
    reg_map = torch.zeros((2, mapsize[0], mapsize[1]))

    # [bars, ticks] -> Class 1, 2
    for cls_id, cls_list in enumerate(cls_pts_lists):
        for point in cls_list:
            # Compute map coordinates and regression values
            pos = torch.floor(point.flip(-1) * mapsize).type(torch.int32)
            reg = point.flip(-1) * mapsize - pos - 0.5

            # Store in maps
            cls_map[pos[0], pos[1]] = cls_id + 1
            reg_map[:, pos[0], pos[1]] = reg

    return cls_map, reg_map

# TODO
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

        match cls:
            case 0: # Background
                pass
            case 1: # Bar
                bars[im].append((pos_x, pos_y))
            case 2: # Tick
                ticks[im].append((pos_x, pos_y))
            case _:
                raise ValueError(f"Unknown class {cls}")

    return bars, ticks

# TODO
def get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh, conf_thresh):
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

# TODO
def nms(bars, ticks, thresh):
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

# TODO
def evaluate_pts(gt_bars, gt_ticks, pred_bars, pred_ticks, dist_thresh):
    bar_matches = [[((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) if
                    ((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_bars] for gb in gt_bars]
    for i in range(len(gt_bars)):
        min_dist = min([bm for bm in bar_matches[i] if bm > 0], default = 0.)
        bar_matches[i] = [m if m <= min_dist else 0 for m in bar_matches[i]]

    num_matches = np.count_nonzero(bar_matches)

    bar_precision = (num_matches / len(pred_bars)) if len(pred_bars) != 0 else 0
    bar_recall = (num_matches / len(gt_bars)) if len(gt_bars) != 0 else 0

    tick_matches = [[((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) if
                     ((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in pred_ticks] for gt in gt_ticks]
    for i in range(len(gt_ticks)):
        min_dist = min([tm for tm in tick_matches[i] if tm > 0], default = 0.)
        tick_matches[i] = [m if m <= min_dist else 0 for m in tick_matches[i]]

    num_matches = np.count_nonzero(tick_matches)

    tick_precision = (num_matches / len(pred_ticks)) if len(pred_ticks) != 0 else 0
    tick_recall = (num_matches / len(gt_ticks)) if len(gt_ticks) != 0 else 0

    return bar_precision, bar_recall, tick_precision, tick_recall

# TODO
def evaluate_pts_err(gt_bars, gt_ticks, p_bars, p_ticks, dist_thresh):
    bar_matches = [[((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) if
                    ((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in p_bars] for gb in gt_bars]
    min_bar_dists = [min([bm for bm in row if bm > 0], default = 0.) for row in bar_matches]
    
    tick_matches = [[((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) if
                     ((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in p_ticks] for gt in gt_ticks]
    min_tick_dists = [min([tm for tm in row if tm > 0], default = 0.) for row in tick_matches]
    
    return sum(min_bar_dists) + sum(min_tick_dists)


def f1(precision, recall):
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
