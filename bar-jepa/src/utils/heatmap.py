import sys
import logging

import torch
import torch.nn.functional as F

from typing import List, Tuple

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def cls_pts_to_maps(
    cls_pts_lists: List[torch.Tensor],
    origin: torch.Tensor,
    mapsize: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of points of different classes
    to maps of point classes and regression values.

    :param cls_pts_lists: list of lists containing normalized points for each class
    :param mapsize: size of the output maps, (H, W)
    :return: tuple containing:

        - map with class IDs, shape [H, W]
        - map with origin, shape [H, W]
        - map with regression values (offsets), [y, x], shape [2, H, W]
    """

    # Make maps of size [H, W], [2, H, W]
    cls_map = torch.zeros(size=(mapsize[0], mapsize[1]), dtype=torch.long)
    org_map = torch.zeros(size=(mapsize[0], mapsize[1]))
    reg_map = torch.zeros(size=(2, mapsize[0], mapsize[1]))

    # [bars, ticks] -> Class 1, 2
    for cls_id, cls_list in enumerate(cls_pts_lists):
        for point in cls_list:
            # Compute map coordinates and regression values
            pos = torch.floor(point * mapsize).type(torch.int32)
            reg = (point * mapsize) - pos - 0.5

            # Store in maps
            cls_map[pos[0], pos[1]] = cls_id + 1
            reg_map[:, pos[0], pos[1]] = reg

    # Store origin
    org_pos = torch.floor(origin * mapsize).type(torch.int32)
    org_reg = (origin * mapsize) - org_pos - 0.5
    org_map[org_pos[0], org_pos[1]] = 1.0
    reg_map[:, org_pos[0], org_pos[1]] = org_reg

    return org_map, cls_map, reg_map


def build_slot_heatmaps(
    origin: torch.Tensor,
    ticks: torch.Tensor,
    bars: torch.Tensor,
    mapsize: torch.Tensor,
    num_hm_slots: int,
    num_tick_slots: int = 15,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Builds slot-based heatmaps for origin, ticks, and bars.

    Slot layout:
    - 0: origin
    - [1, num_tick_slots]: ticks (sorted bottom->top)
    - [num_tick_slots + 1, num_hm_slots]: bars (sorted left->right)

    :param origin: origin points, shape [1, 2] (normalized [y, x])
    :param ticks: tick points, shape [T, 2] (normalized [y, x])
    :param bars: bar points, shape [B, 2] (normalized [y, x])
    :param mapsize: map size, shape [2] (H, W)
    :param num_hm_slots: total number of slots (K)
    :param num_tick_slots: number of tick slots
    :param sigma: gaussian sigma in heatmap pixels
    :return: heatmaps tensor, shape [K, H, W]
    """

    def draw_heatmap(
        heatmap: torch.Tensor,
        center: torch.Tensor,
        gaussian: torch.Tensor,
        radius: int
    ) -> None:
        H, W = heatmap.shape
        cy = float(center[0])
        cx = float(center[1])
        if not (0.0 <= cy <= 1.0 and 0.0 <= cx <= 1.0):
            return

        mu_y = int(cy * H)
        mu_x = int(cx * W)
        if mu_x < 0 or mu_x >= W or mu_y < 0 or mu_y >= H:
            return

        ul = (mu_x - radius, mu_y - radius)
        br = (mu_x + radius + 1, mu_y + radius + 1)

        if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
            return

        g_x0 = max(0, -ul[0])
        g_x1 = min(br[0], W) - ul[0]
        g_y0 = max(0, -ul[1])
        g_y1 = min(br[1], H) - ul[1]

        img_x0 = max(0, ul[0])
        img_x1 = min(br[0], W)
        img_y0 = max(0, ul[1])
        img_y1 = min(br[1], H)

        heatmap[img_y0:img_y1, img_x0:img_x1] = gaussian[g_y0:g_y1, g_x0:g_x1]

    H, W = int(mapsize[0].item()), int(mapsize[1].item())
    device = origin.device
    heatmaps = torch.zeros((num_hm_slots, H, W), device=device, dtype=torch.float32)

    # Make kernel
    radius = int(sigma * 3)
    size = 2 * radius + 1
    xs = torch.arange(0, size, device=device, dtype=heatmaps.dtype)
    ys = xs[:, None]
    x0 = y0 = size // 2
    gaussian = torch.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))

    # Origin slot
    if origin.numel() >= 2:
        draw_heatmap(heatmaps[0], origin[0], gaussian, radius)

    # Ticks: bottom -> top (y descending, since origin is top-left)
    if ticks.numel() > 0:
        ticks_sorted = ticks[torch.argsort(ticks[:, 0], descending=True)]
    else:
        ticks_sorted = ticks

    max_ticks = min(num_tick_slots, ticks_sorted.size(0))
    for i in range(max_ticks):
        draw_heatmap(heatmaps[1 + i], ticks_sorted[i], gaussian, radius)

    # Bars: left -> right (x ascending)
    bar_slots = num_hm_slots - 1 - num_tick_slots
    if bars.numel() > 0:
        bars_sorted = bars[torch.argsort(bars[:, 1])]
    else:
        bars_sorted = bars

    max_bars = min(bar_slots, bars_sorted.size(0))
    for i in range(max_bars):
        draw_heatmap(heatmaps[1 + num_tick_slots + i], bars_sorted[i], gaussian, radius)

    return heatmaps


def gt_maps_to_cls_lists(
    gt_org: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_reg: torch.Tensor,
    size: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts ground truth class and regression maps
    to lists of bar and tick positions.

    :param gt_org: ground truth origin map, shape [H, W]
    :param gt_cls: ground truth class map, shape [H, W]
    :param gt_reg: ground truth regression map, shape [2, H, W]
    :param size: size of the map, shape [2]
    :return: Tuple containing:

        - origin, [y, x] in image space
        - bars, [N, 2] -> [y, x] in image space
        - ticks, [M, 2] -> [y, x] in image space
    """
    # Get all non-background points
    points = torch.nonzero(gt_cls > 0)
    org_point = torch.nonzero(gt_org > 0)

    # Pixel midpoints in image space
    pos = (points * 2 + 1) / (size * 2)
    org = (org_point * 2 + 1) / (size * 2)
    # Apply regression offsets
    pos += gt_reg[:, points[:, 0], points[:, 1]].T / size
    org += gt_reg[:, org_point[:, 0], org_point[:, 1]].T / size

    # Class labels for each point
    cls = gt_cls[points[:, 0], points[:, 1]]

    # Split into bars and ticks
    bars  = pos[cls == 1]
    ticks = pos[cls == 2]

    return org, bars, ticks


def p_maps_to_cls_lists(
    p_cls: torch.Tensor,
    p_reg: torch.Tensor,
    size: torch.Tensor,
    bg_conf_thresh: float,
    cls_conf_thresh: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Extracts and processes predicted bars, ticks & origin
    from classification and regression maps.

    :param p_cls: classification map [4, H, W] (bg, bar, tick, origin)
    :param p_reg: regression map [2, H, W]
    :param size: size of the map, shape [2]
    :param bg_conf_thresh: background confidence threshold
    :param cls_conf_thresh: classification confidence threshold
    :return: tuple containing:

        - lists of [[y, x], confidence] for predicted bars
        - lists of [[y, x]], confidence] for predicted ticks
        - [y, x] for predicted coordinate system origin
    """

    bars = []
    ticks = []

    # Mask out parts that very likely are background
    pts_mask = torch.sigmoid(p_cls[0]).lt(bg_conf_thresh)

    # Select high confidence candidates from background-masked bars heatmap
    masked_bars = (torch.sigmoid(p_cls[1]) * pts_mask).gt(cls_conf_thresh)
    for point in torch.nonzero(masked_bars):
        # Pixel midpoint in image space
        pos = (point * 2 + 1) / (size * 2)
        # Offset pixel midpoint by regression value
        pos += p_reg[:, point[0], point[1]] / size
        # Store image space prediction and confidence for bar
        conf = torch.sigmoid(p_cls[1, point[0], point[1]])
        bars.append((pos, conf))

    # Select high confidence candidates from background-masked ticks heatmap
    masked_ticks = (torch.sigmoid(p_cls[2]) * pts_mask).gt(cls_conf_thresh)
    for point in torch.nonzero(masked_ticks):
        # Pixel midpoint in image space
        pos = (point * 2 + 1) / (size * 2)
        # Offset pixel midpoint by regression value
        pos += p_reg[:, point[0], point[1]] / size
        # Store image space prediction and confidence for tick
        conf = torch.sigmoid(p_cls[2, point[0], point[1]])
        ticks.append((pos, conf))

    # Select highest confidence candidate from background-masked origin heatmap
    org_idx = torch.argmax(torch.sigmoid(p_cls[3]) * pts_mask)
    org_point = torch.stack(torch.unravel_index(org_idx, p_cls[3].shape))
    org_pos = (org_point * 2 + 1) / (size * 2)
    # Offset pixel midpoint by regression value
    org_pos += p_reg[:, org_point[0], org_point[1]] / size

    return bars, ticks, org_pos


def nms(
    p_bars: List[Tuple[torch.Tensor, torch.Tensor]],
    p_ticks: List[Tuple[torch.Tensor, torch.Tensor]],
    radius_thresh: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Performs non-maximum suppression on predicted bars and ticks.

    :param p_bars: list of (position, shape: [2], confidence) for predicted bars
    :param p_ticks: list of (position, shape: [2], confidence) for predicted ticks
    :param radius_thresh: threshold distance for considering points as neighbors
    :return: tuple containing:

        - list of nms filtered bars
        - list of nms filtered ticks
    """
    nms_bars = []
    nms_ticks = []

    # Sort bars by confidence
    if len(p_bars) > 0:
        sorted_bar_pts = sorted(p_bars, key=lambda p: -p[1])
        all_bar_pts = torch.stack([p for p, _ in sorted_bar_pts])
        checked_bar_pts = []
        # Iteratively remove low confidence bars within radius
        for k, (pt, conf) in enumerate(sorted_bar_pts):
            if k in checked_bar_pts:
                continue
            # Anisotropic radius for bars, more reach in width direction
            distances = torch.abs(all_bar_pts[:, 0] - pt[0]) / 5. + torch.abs(all_bar_pts[:, 1] - pt[1])
            neighbor_idx = torch.nonzero(distances < radius_thresh).squeeze(1).tolist()
            # Update checked points and maximum points
            checked_bar_pts.extend(neighbor_idx)
            nms_bars.append(pt)

    # Sort ticks by confidence
    if len(p_ticks) > 0:
        sorted_tick_pts = sorted(p_ticks, key=lambda p: -p[1])
        all_tick_pts = torch.stack([p for p, _ in sorted_tick_pts])
        checked_tick_pts = []
        # Iteratively remove low confidence ticks within radius
        for k, (pt, conf) in enumerate(sorted_tick_pts):
            if k in checked_tick_pts:
                continue
            # Equal radius for ticks
            distances = torch.abs(all_tick_pts[:, 0] - pt[0]) + torch.abs(all_tick_pts[:, 1] - pt[1])
            neighbor_idx = torch.nonzero(distances < radius_thresh).squeeze(1).tolist()
            # Update checked points and maximum points
            checked_tick_pts.extend(neighbor_idx)
            nms_ticks.append(pt)

    return nms_bars, nms_ticks


def evaluate_gt_p_match(
    gt_bars: torch.Tensor,
    gt_ticks: torch.Tensor,
    p_bars: List[torch.Tensor],
    p_ticks: List[torch.Tensor],
    dist_thresh: float
) -> Tuple[float, float, float, float]:
    """
    Evaluate predicted bars and ticks against ground truth.

    :param gt_bars: ground truth bar positions, shape [n, 2]
    :param gt_ticks: ground truth tick positions, shape [m, 2]
    :param p_bars: list of predicted bar positions, shape [2]
    :param p_ticks: list of predicted tick positions, shape [2]
    :param dist_thresh: distance threshold for considering a match
    :param eval: flag indicating which metrics to compute
    :return: Tuple containing:

        - bar precision & recall
        - tick precision & recall
    """

    # Helper function, calculates distance to closest p for each gt, or 0 if none exists
    def closest_match(gt_elems, p_elems):
        if len(p_elems) > 0:
            # Calculate pairwise distances
            g_comp = gt_elems.unsqueeze(1)                              # [n_gt, 1, 2]
            p_comp = torch.stack(p_elems).unsqueeze(0)                  # [1, n_p, 2]
            distances = torch.sqrt(((g_comp - p_comp) ** 2).sum(dim=2)) # [n_gt, n_p]

            # Create matches mask
            matches = (distances < dist_thresh)

            # Calculate distances for each ground truth element (inf mask non-matches)
            distances = distances.masked_fill(~matches, float('inf'))
            min_dists, _ = torch.min(distances, dim=1)

            # Set min distance to 0 if no matches were found
            min_dists = torch.where(matches.any(dim=1), min_dists, torch.zeros_like(min_dists))

            # Return number of matching elements
            return matches.sum().item()
        else:
            # No predictions, no matches
            return 0

    # Calculate matches for ticks & bars
    bar_matches = closest_match(gt_bars, p_bars)
    tick_matches = closest_match(gt_ticks, p_ticks)

    # Calculate precision and recall for bars
    bar_precision = (bar_matches / len(p_bars)) if len(p_bars) > 0 else 0.
    bar_recall = (bar_matches / len(gt_bars)) if len(gt_bars) > 0 else 0.

    # Calculate precision and recall for ticks
    tick_precision = (tick_matches / len(p_ticks)) if len(p_ticks) > 0 else 0.
    tick_recall = (tick_matches / len(gt_ticks)) if len(gt_ticks) > 0 else 0.

    # Return all metrics
    return bar_precision, bar_recall, tick_precision, tick_recall


def f1(
    precision: float,
    recall: float
) -> float:
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
