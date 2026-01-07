import torch

from typing import List, Tuple

def cls_pts_to_map(
    cls_pts_lists: List[torch.Tensor],
    mapsize: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a list of points of different classes
    to maps of point classes and regression values.

    :param cls_pts_lists: list of lists containing normalized points for each class
    :param mapsize: size of the output maps, (H, W)
    :return: tuple containing:

        - map with class IDs, shape [H, W]
        - map with regression values (offsets), [y, x], shape [2, H, W]
    """

    # Make maps of size [H, W], [2, H, W]
    cls_map = torch.zeros(size=(mapsize[0], mapsize[1]), dtype=torch.long)
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

    return cls_map, reg_map


def gt_maps_to_cls_lists(
    gt_cls: torch.Tensor,
    gt_reg: torch.Tensor,
    size: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Converts ground truth class and regression maps
    to lists of bar and tick positions.

    :param gt_cls: ground truth class map, shape [H, W]
    :param gt_reg: ground truth regression map, shape [2, H, W]
    :param size: size of the map, shape [2]
    :return: Tuple containing:

        - list of bars, [y, x] in image space
        - list of ticks, [y, x] in image space
    """
    bars = []
    ticks = []

    for point in torch.nonzero(gt_cls):
        # Pixel midpoint in image space
        pos = (point * 2 + 1) / (size * 2)
        # Offset pixel midpoint by regression value
        pos += gt_reg[:, point[0], point[1]] / size

        # Add to corresponding class's list
        match gt_cls[point[0], point[1]]:
            case 0: # Background
                pass
            case 1: # Bar
                bars.append(pos)
            case 2: # Tick
                ticks.append(pos)
            case _:
                raise ValueError(f"Unknown class {gt_cls[point[0], point[1]]}")

    return bars, ticks


def gt_maps_to_cls_lists_v2(
    gt_cls: torch.Tensor,  # Shape: [H, W]
    gt_reg: torch.Tensor,  # Shape: [2, H, W]
    size: torch.Tensor     # Shape: [2]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts ground truth class and regression maps to tensors of bar and tick positions.
    Returns tensors of shape [N, 2] and [M, 2] for bars and ticks, respectively.
    """
    # Get all non-background points
    points = torch.nonzero(gt_cls > 0)

    # Pixel midpoints in image space
    pos = (points * 2 + 1) / (size * 2)
    # Apply regression offsets
    pos += gt_reg[:, points[:, 0], points[:, 1]].T / size

    # Class labels for each point
    cls = gt_cls[points[:, 0], points[:, 1]]

    # Mask for bars and ticks
    bar_mask = cls == 1
    tick_mask = cls == 2

    # Extract bars and ticks
    bars = pos[bar_mask]
    ticks = pos[tick_mask]

    return bars, ticks


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

    :param p_cls: classification map [3, H, W]
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


def p_maps_to_cls_lists_v2(
    p_cls: torch.Tensor,
    p_reg: torch.Tensor,
    size: torch.Tensor,
    bg_conf_thresh: float,
    cls_conf_thresh: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts and processes predicted bars, ticks & origin from classification and regression maps.
    Returns tensors instead of lists for bars, ticks, and origin.
    """
    # Precompute sigmoid once
    p_cls_sig = torch.sigmoid(p_cls)

    # Background mask
    pts_mask = p_cls_sig[0] < bg_conf_thresh

    def extract_points(cls_map: torch.Tensor) -> torch.Tensor:
        # Confidence mask
        cls_conf = cls_map * pts_mask
        mask = cls_conf > cls_conf_thresh

        # Get coordinates and confidences
        points = torch.nonzero(mask)
        if points.numel() == 0:
            return torch.empty((0, 3), device=p_cls.device)  # [y, x, conf]

        # Pixel midpoint in image space
        pos = ((points * 2 + 1) / (size * 2)).to(cls_map.dtype)

        # Apply regression offsets
        pos += p_reg[:, points[:, 0], points[:, 1]].T / size
        conf = cls_map[points[:, 0], points[:, 1]]
        return torch.cat([pos, conf.unsqueeze(1)], dim=1)

    # Extract bars and ticks
    bars = extract_points(p_cls_sig[1])
    ticks = extract_points(p_cls_sig[2])

    # Origin: highest confidence point
    org_conf = p_cls_sig[3] * pts_mask
    org_idx = torch.argmax(org_conf)
    org_point = torch.tensor(torch.unravel_index(org_idx, p_cls[3].shape), device=p_cls.device)
    org_pos = (org_point * 2 + 1) / (size * 2) + p_reg[:, org_point[0], org_point[1]] / size

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


def nms_v2(
    p_bars: torch.Tensor,  # Shape: [N, 3] (x, y, conf)
    p_ticks: torch.Tensor,  # Shape: [M, 3] (x, y, conf)
    radius_thresh: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs non-maximum suppression on predicted bars and ticks.
    Returns tensors of suppressed coordinates (without confidence) for bars and ticks.
    """
    def suppress_points(points: torch.Tensor, anisotropic: bool = False) -> torch.Tensor:
        if points.numel() == 0:
            return points[:, :2]  # Return empty tensor with correct shape

        # Sort by confidence (descending)
        points = points.detach()
        sorted_indices = points[:, 2].argsort(descending=True)
        points = points[sorted_indices]

        # Coordinates
        coords = points[:, :2]

        # Pairwise distances
        dx = torch.abs(coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0))
        dy = torch.abs(coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0))

        if anisotropic:
            distances = dx / 5. + dy
        else:
            distances = dx + dy

        # NMS logic
        keep = torch.ones(points.shape[0], device=points.device).bool()
        for i in range(points.shape[0]):
            if keep[i]:
                # Suppress neighbors within radius
                neighbors = distances[i] < radius_thresh
                neighbors[i] = False  # Keep current point
                keep[neighbors] = False

        return coords[keep]

    # Suppress bars (anisotropic) and return coordinates only
    nms_bars = suppress_points(p_bars, anisotropic=True)

    # Suppress ticks (isotropic) and return coordinates only
    nms_ticks = suppress_points(p_ticks, anisotropic=False)

    return nms_bars, nms_ticks


def evaluate_gt_p_match(
    gt_bars: List[torch.Tensor],
    gt_ticks: List[torch.Tensor],
    p_bars: List[torch.Tensor],
    p_ticks: List[torch.Tensor],
    dist_thresh: float,
    eval: bool=False
) -> torch.Tensor | Tuple[float, float, float, float]:
    """
    Evaluate predicted bars and ticks against ground truth.

    :param gt_bars: list of ground truth bar positions, shape [2]
    :param gt_ticks: list of ground truth tick positions, shape [2]
    :param p_bars: list of predicted bar positions, shape [2]
    :param p_ticks: list of predicted tick positions, shape [2]
    :param dist_thresh: distance threshold for considering a match
    :param eval: flag indicating which metrics to compute
    :return: depending on if train / eval:

        - train: total error distance
        - eval: bar precision & recall, tick precision & recall, origin precision & recall
    """

    # Helper function, calculates distance to closest p for each gt, or 0 if none exists
    def closest_match(gt_elems, p_elems):
        if len(p_elems) > 0:
            # Calculate pairwise distances
            g_comp = torch.stack(gt_elems).unsqueeze(1)                 # [n_gt, 1, 2]
            p_comp = torch.stack(p_elems).unsqueeze(0)                  # [1, n_p, 2]
            distances = torch.sqrt(((g_comp - p_comp) ** 2).sum(dim=2)) # [n_gt, n_p]

            # Create matches mask
            matches = (distances < dist_thresh)

            # Calculate distances for each ground truth element (inf mask non-matches)
            distances = distances.masked_fill(~matches, float('inf'))
            min_dists, _ = torch.min(distances, dim=1)

            # Set min distance to 0 if no matches were found
            min_dists = torch.where(matches.any(dim=1), min_dists, torch.zeros_like(min_dists))

            # Return number of matching elements and total distance error
            return matches.sum().item(), min_dists.sum()
        else:
            # No predictions, no matches
            return 0, torch.zeros_like(gt_elems[0][0])

    # Calculate matches for ticks & bars
    bar_matches, min_bar_dists = closest_match(gt_bars, p_bars)
    tick_matches, min_tick_dists = closest_match(gt_ticks, p_ticks)

    if eval:
        # Calculate precision and recall for bars
        bar_precision = (bar_matches / len(p_bars))
        bar_recall = (bar_matches / len(gt_bars))

        # Calculate precision and recall for ticks
        tick_precision = (tick_matches / len(p_ticks))
        tick_recall = (tick_matches / len(gt_ticks))

        # Return all metrics
        return bar_precision, bar_recall, tick_precision, tick_recall
    else:
        # If training, return total distance error
        return min_bar_dists + min_tick_dists


def evaluate_gt_p_match_v2(
    gt_bars: torch.Tensor,   # Shape: [N, 2]
    gt_ticks: torch.Tensor,  # Shape: [M, 2]
    p_bars: torch.Tensor,    # Shape: [K, 2]
    p_ticks: torch.Tensor,   # Shape: [L, 2]
    dist_thresh: float,
    eval: bool = False
) -> torch.Tensor | Tuple[float, float, float, float]:
    """
    Evaluate predicted bars and ticks against ground truth.
    Inputs are tensors of coordinates.
    """
    def closest_match(gt_elems: torch.Tensor, p_elems: torch.Tensor) -> Tuple[int, torch.Tensor]:
        if p_elems.numel() == 0:
            return 0, torch.tensor(0., device=gt_elems.device)

        # Pairwise distance using broadcasting: [n_gt, n_p, 2] -> [n_gt, n_p]
        diff = gt_elems.unsqueeze(1) - p_elems.unsqueeze(0)
        distances = (diff ** 2).sum(dim=2).sqrt()
        matches = distances < dist_thresh  # bool

        # Min distances per gt, inf for non-matches
        min_dists = distances.min(dim=1).values
        min_dists = torch.where(matches.any(dim=1), min_dists, torch.zeros_like(min_dists))

        return matches.sum().item(), min_dists.sum()

    bar_matches, min_bar_dists = closest_match(gt_bars, p_bars)
    tick_matches, min_tick_dists = closest_match(gt_ticks, p_ticks)

    if eval:
        bar_precision = bar_matches / max(len(p_bars), 1)
        bar_recall = bar_matches / max(len(gt_bars), 1)
        tick_precision = tick_matches / max(len(p_ticks), 1)
        tick_recall = tick_matches / max(len(gt_ticks), 1)
        return bar_precision, bar_recall, tick_precision, tick_recall
    else:
        return min_bar_dists + min_tick_dists


def f1(
    precision: float,
    recall: float
) -> float:
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
