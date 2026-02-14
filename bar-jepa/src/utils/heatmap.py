import torch
import torch.nn.functional as F

from typing import List, Tuple


def adaptive_wing_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 2.1,
    omega: float = 14.0,
    epsilon: float = 1.0,
    theta: float = 0.5
) -> torch.Tensor:
    """
    Computes the adaptive wing loss between predictions and ground truth.

    :param pred: predicted heatmap, values in [0, 1]
    :param gt: ground truth heatmap, values in [0, 1]
    :param use_weight_map: whether to apply a weighted loss map
    :return: Mean loss across all elements
    """
    delta = (gt - pred).abs()

    # Loss function for large errors
    A = omega * (
        1 / (1 + torch.pow(theta / epsilon, alpha - gt))
    ) * (alpha - gt) * (
        torch.pow(theta / epsilon, alpha - gt - 1)
    ) * (1 / epsilon)

    # Ensures continuity at the threshold point
    C = theta * A - omega * torch.log(
        1 + torch.pow(theta / epsilon, alpha - gt)
    )

    # For small errors (delta < theta): use logarithmic loss
    # For large errors (delta >= theta): use linear loss
    losses = torch.where(
        delta < theta,
        omega * torch.log(1 + torch.pow(delta / epsilon, alpha - gt)),
        A * delta - C
    )

    return losses.mean()


def udp_encode_point(
    point: torch.Tensor,
    size: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts normalized coordinates to discrete index + sub-pixel offset.

    :param point: normalized [y, x]
    :param size: map size [H, W]
    :return: Tuple containing:

        - map index [y, x]
        - regression offset [y, x]
    """
    coord = point * (size.to(dtype=torch.float32) - 1).clamp(min=1.0)
    idx = torch.floor(coord).type(torch.int32)
    reg = coord - idx - 0.5
    return idx, reg


def udp_decode_point(
    idx: torch.Tensor,
    reg: torch.Tensor,
    size: torch.Tensor
) -> torch.Tensor:
    """
    Converts discrete index + sub-pixel offset to normalized coordinates.

    :param index: map index [y, x] or [N, [y, x]]
    :param reg: regression offset [y, x] or [N, [y, x]]
    :param size: map size [H, W]
    :return: normalized point [y, x]
    """
    scale = (size.to(dtype=torch.float32) - 1).clamp(min=1.0)
    return (idx.to(dtype=torch.float32) + reg + 0.5) / scale


def draw_heatmap(
    heatmap: torch.Tensor,
    center: torch.Tensor,
    sigma: float
) -> None:
    """
    Draws a gaussian heatmap centered at a normalized coordinate.

    :param heatmap: target heatmap [H, W]
    :param center: normalized [y, x] coordinate
    :param sigma: gaussian sigma in heatmap pixels
    """
    H, W = heatmap.shape
    cy = float(center[0])
    cx = float(center[1])

    mu_y = cy * (H - 1)
    mu_x = cx * (W - 1)
    radius = int(sigma * 3)

    ul = (int(mu_x) - radius, int(mu_y) - radius)
    br = (int(mu_x) + radius + 1, int(mu_y) + radius + 1)

    x0 = max(0, ul[0])
    x1 = min(br[0], W)
    y0 = max(0, ul[1])
    y1 = min(br[1], H)

    xs = torch.arange(x0, x1, device=heatmap.device, dtype=heatmap.dtype)
    ys = torch.arange(y0, y1, device=heatmap.device, dtype=heatmap.dtype)
    ys = ys[:, None]
    gaussian = torch.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2 * sigma ** 2))

    heatmap[y0:y1, x0:x1] = gaussian


def cls_pts_to_maps(
    cls_pts_lists: List[torch.Tensor],
    origin: torch.Tensor,
    mapsize: torch.Tensor,
    origin_sigma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of points of different classes
    to maps of point classes and regression values.

    :param cls_pts_lists: list of lists containing normalized points for each class
    :param mapsize: size of the output maps, (H, W)
    :param origin_sigma: gaussian sigma for origin target in heatmap pixels
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
            pos, reg = udp_encode_point(point, mapsize)

            # Store in maps
            cls_map[pos[0], pos[1]] = cls_id + 1
            reg_map[:, pos[0], pos[1]] = reg

    # Origin heatmap can be either one hot or gaussian target
    if origin_sigma > 0:
        draw_heatmap(org_map, origin, origin_sigma)
        # Also store offset at the argmax pixel to match decoding.
        org_argmax = torch.argmax(org_map)
        org_point = torch.stack(torch.unravel_index(org_argmax, org_map.shape))
        scale = (mapsize.to(dtype=torch.float32) - 1).clamp(min=1.0)
        org_coord = origin * scale
        reg_map[:, org_point[0], org_point[1]] = (
            org_coord - org_point.to(dtype=torch.float32) - 0.5
        )
    else:
        org_idx, org_reg = udp_encode_point(origin, mapsize)
        org_map[org_idx[0], org_idx[1]] = 1.0
        reg_map[:, org_idx[0], org_idx[1]] = org_reg

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
    H, W = int(mapsize[0].item()), int(mapsize[1].item())
    device = origin.device
    heatmaps = torch.zeros((num_hm_slots, H, W), device=device, dtype=torch.float32)

    # Origin slot
    if origin.numel() >= 2:
        draw_heatmap(heatmaps[0], origin[0], sigma)

    # Ticks: bottom -> top (y descending, since origin is top-left)
    if ticks.numel() > 0:
        ticks_sorted = ticks[torch.argsort(ticks[:, 0], descending=True)]
    else:
        ticks_sorted = ticks

    max_ticks = min(num_tick_slots, ticks_sorted.size(0))
    for i in range(max_ticks):
        draw_heatmap(heatmaps[1 + i], ticks_sorted[i], sigma)

    # Bars: left -> right (x ascending)
    bar_slots = num_hm_slots - 1 - num_tick_slots
    if bars.numel() > 0:
        bars_sorted = bars[torch.argsort(bars[:, 1])]
    else:
        bars_sorted = bars

    max_bars = min(bar_slots, bars_sorted.size(0))
    for i in range(max_bars):
        draw_heatmap(heatmaps[1 + num_tick_slots + i], bars_sorted[i], sigma)

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
    org_idx = torch.argmax(gt_org)
    org_point = torch.stack(torch.unravel_index(org_idx, gt_org.shape)).unsqueeze(0)

    # UDP: use pixel index + offset + 0.5, then normalize by (size - 1)
    pos = udp_decode_point(
        points,
        gt_reg[:, points[:, 0], points[:, 1]].T,
        size
    )
    org = udp_decode_point(
        org_point,
        gt_reg[:, org_point[:, 0], org_point[:, 1]].T,
        size
    )

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts and processes predicted bars, ticks & origin
    from classification and regression maps.

    :param p_cls: classification map [4, H, W] (bg, bar, tick, origin)
    :param p_reg: regression map [2, H, W]
    :param size: size of the map, shape [2]
    :param bg_conf_thresh: background confidence threshold
    :param cls_conf_thresh: classification confidence threshold
    :return: tuple containing:

        - predicted bars [y, x, conf], shape [N, 3]
        - predicted ticks [y, x, conf], shape [M, 3]
        - [y, x] for predicted coordinate system origin
    """

    # Mask out parts that very likely are background
    pts_mask = torch.sigmoid(p_cls[0]).lt(bg_conf_thresh)
    bar_scores = torch.sigmoid(p_cls[1])
    tick_scores = torch.sigmoid(p_cls[2])

    # Select high confidence candidates from background-masked bars heatmap
    bar_idx = torch.nonzero(pts_mask & bar_scores.gt(cls_conf_thresh))
    if bar_idx.numel() > 0:
        bar_reg = p_reg[:, bar_idx[:, 0], bar_idx[:, 1]].T
        bar_pos = udp_decode_point(bar_idx, bar_reg, size)
        bar_conf = bar_scores[bar_idx[:, 0], bar_idx[:, 1]]
        bars = torch.cat((bar_pos, bar_conf.unsqueeze(1)), dim=1)
    else:
        bars = p_reg.new_empty((0, 3))

    # Select high confidence candidates from background-masked ticks heatmap
    tick_idx = torch.nonzero(pts_mask & tick_scores.gt(cls_conf_thresh))
    if tick_idx.numel() > 0:
        tick_reg = p_reg[:, tick_idx[:, 0], tick_idx[:, 1]].T
        tick_pos = udp_decode_point(tick_idx, tick_reg, size)
        tick_conf = tick_scores[tick_idx[:, 0], tick_idx[:, 1]]
        ticks = torch.cat((tick_pos, tick_conf.unsqueeze(1)), dim=1)
    else:
        ticks = p_reg.new_empty((0, 3))

    # Select highest confidence candidate from background-masked origin heatmap
    org_idx = torch.argmax(torch.sigmoid(p_cls[3]) * pts_mask)
    org_point = torch.stack(torch.unravel_index(org_idx, p_cls[3].shape))
    org_pos = udp_decode_point(org_point, p_reg[:, org_point[0], org_point[1]], size)

    return bars, ticks, org_pos


def nms(
    p_bars: torch.Tensor,
    p_ticks: torch.Tensor,
    radius_thresh: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs non-maximum suppression on predicted bars and ticks.

    :param p_bars: predicted bars [y, x, conf], shape [N, 3]
    :param p_ticks: predicted ticks [y, x, conf], shape [M, 3]
    :param radius_thresh: threshold distance for considering points as neighbors
    :return: tuple containing:

        - nms filtered bars [y, x, conf], shape [N, 3]
        - nms filtered ticks [y, x, conf], shape [M, 3]
    """
    bar_out = p_bars.new_empty((0, 3))
    tick_out = p_ticks.new_empty((0, 3))

    # Sort bars by confidence
    if p_bars.size(0) > 0:
        sorted_bars = p_bars[torch.argsort(p_bars[:, 2], descending=True)]
        checked_bars = torch.zeros(sorted_bars.size(0), dtype=torch.bool, device=sorted_bars.device)
        keep_bars = torch.zeros_like(checked_bars)

        # Iteratively remove low confidence bars within radius
        for k in range(sorted_bars.size(0)):
            if checked_bars[k]:
                continue
            pt = sorted_bars[k]
            # Anisotropic radius for bars, more reach in width direction
            distances = torch.abs(sorted_bars[:, 0] - pt[0]) + (torch.abs(sorted_bars[:, 1] - pt[1]) / 5.)
            checked_bars |= distances.lt(radius_thresh)
            keep_bars[k] = True

        bar_out = sorted_bars[keep_bars]

    # Sort ticks by confidence
    if p_ticks.size(0) > 0:
        sorted_ticks = p_ticks[torch.argsort(p_ticks[:, 2], descending=True)]
        checked_ticks = torch.zeros(sorted_ticks.size(0), dtype=torch.bool, device=sorted_ticks.device)
        keep_ticks = torch.zeros_like(checked_ticks)

        # Iteratively remove low confidence ticks within radius
        for k in range(sorted_ticks.size(0)):
            if checked_ticks[k]:
                continue
            pt = sorted_ticks[k]
            # Equal radius for ticks
            distances = torch.abs(sorted_ticks[:, 0] - pt[0]) + torch.abs(sorted_ticks[:, 1] - pt[1])
            checked_ticks |= distances.lt(radius_thresh)
            keep_ticks[k] = True

        tick_out = sorted_ticks[keep_ticks]

    return bar_out, tick_out


def evaluate_gt_p_match(
    gt_bars: torch.Tensor,
    gt_ticks: torch.Tensor,
    p_bars: torch.Tensor,
    p_ticks: torch.Tensor,
    dist_thresh: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate predicted bars and ticks against ground truth.

    :param gt_bars: ground truth bar positions, shape [N, 2]
    :param gt_ticks: ground truth tick positions, shape [M, 2]
    :param p_bars: predicted bars [y, x, conf], shape [N, 3]
    :param p_ticks: predicted ticks [y, x, conf], shape [M, 3]
    :param dist_thresh: max. distance threshold when considering matches
    :return: Tuple containing:

        - bar precision, recall, f1
        - tick precision, recall, f1
    """

    def closest_match(gt_elems: torch.Tensor, p_elems: torch.Tensor) -> int:
        if gt_elems.size(0) == 0 or p_elems.size(0) == 0:
            return 0

        # Pairwise distances between gt and predicted coords
        diffs = gt_elems.unsqueeze(1) - p_elems[:, :2].unsqueeze(0)
        distances = torch.sqrt((diffs * diffs).sum(dim=2))

        # Distances above threshold are not valid matches
        invalid = distances.gt(dist_thresh)
        if invalid.all():
            return 0

        # Coordinates in [0, 1], so max distance is sqrt(2)
        large_cost = (2.0 ** 0.5) + 1e-06
        cost = distances.masked_fill(invalid, large_cost)

        # Perform Hungarian matching
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
        if len(row_idx) == 0:
            return 0

        # Keep only assignments that are within the distance threshold
        row_idx_t = torch.as_tensor(row_idx, device=distances.device, dtype=torch.long)
        col_idx_t = torch.as_tensor(col_idx, device=distances.device, dtype=torch.long)
        return int(distances[row_idx_t, col_idx_t].le(dist_thresh).sum().item())

    # Calculate matches for ticks & bars
    bar_matches = closest_match(gt_bars, p_bars)
    tick_matches = closest_match(gt_ticks, p_ticks)

    # Calculate precision and recall for bars
    bar_precision = (bar_matches / p_bars.size(0)) if p_bars.size(0) > 0 else 0.
    bar_recall = (bar_matches / gt_bars.size(0)) if gt_bars.size(0) > 0 else 0.
    bar_f1 = f1(bar_precision, bar_recall)

    # Calculate precision and recall for ticks
    tick_precision = (tick_matches / p_ticks.size(0)) if p_ticks.size(0) > 0 else 0.
    tick_recall = (tick_matches / gt_ticks.size(0)) if gt_ticks.size(0) > 0 else 0.
    tick_f1 = f1(tick_precision, tick_recall)

    # Return all metrics
    return bar_precision, bar_recall, bar_f1, tick_precision, tick_recall, tick_f1


def f1(
    precision: float,
    recall: float
) -> float:
    """
    Computes the F1 score from precision and recall.

    :param precision: precision value
    :param recall: recall value
    :return: F1 score
    """
    denominator = precision + recall
    if denominator <= 0:
        return 0.0

    numerator = 2.0 * precision * recall
    return numerator / denominator
