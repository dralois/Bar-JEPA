import torch
import torch.nn.functional as F

from typing import List, Tuple


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
    cls_conf_thresh: float,
    score_norm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts and processes predicted bars, ticks & origin
    from classification and regression maps.

    :param p_cls: classification map [4, H, W] (bg, bar, tick, origin)
    :param p_reg: regression map [2, H, W]
    :param size: size of the map, shape [2]
    :param bg_conf_thresh: background confidence threshold
    :param cls_conf_thresh: classification confidence threshold
    :param score_norm: if ``True``, use softmax normalization over bg/bar/tick
        channels (mutually exclusive classes).\n
        If ``False``, use legacy independent sigmoid scores.
    :return: tuple containing:

        - predicted bars [y, x, conf], shape [N, 3]
        - predicted ticks [y, x, conf], shape [M, 3]
        - [y, x] for predicted coordinate system origin
    """

    if score_norm:
        # Normalize bg/bar/tick per pixel
        cls_scores = torch.softmax(p_cls[:3], dim=0)
        pts_mask = cls_scores[0].lt(bg_conf_thresh)
        bar_scores = cls_scores[1]
        tick_scores = cls_scores[2]
    else:
        # Legacy independent per-channel probabilities.
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
