import torch


def nms(
    p_bars: torch.Tensor,
    p_ticks: torch.Tensor,
    radius_thresh: float
) -> tuple[torch.Tensor, torch.Tensor]:
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
            # Keep current bar, neighbors within radius are removed
            distances = torch.abs(sorted_bars[:, 0] - pt[0]) + torch.abs(sorted_bars[:, 1] - pt[1])
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
            # Keep current tick, neighbors within radius are removed
            distances = torch.abs(sorted_ticks[:, 0] - pt[0]) + torch.abs(sorted_ticks[:, 1] - pt[1])
            checked_ticks |= distances.lt(radius_thresh)
            keep_ticks[k] = True

        tick_out = sorted_ticks[keep_ticks]

    return bar_out, tick_out


def hungarian_match(
    gt_points: torch.Tensor,
    p_points: torch.Tensor,
    dist_thresh: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matches GT and predictions with Hungarian matching under a distance threshold.

    :param gt_points: ground truth points [N, 2]
    :param p_points: predicted points [M, 2]
    :param dist_thresh: max distance for a valid match
    :return: tuple containing:

        - matched GT indices [K]
        - matched prediction indices [K]
    """
    device = gt_points.device
    empty = torch.empty((0,), dtype=torch.long, device=device)

    # Nothing to match if either side is empty
    if gt_points.size(0) == 0 or p_points.size(0) == 0:
        return empty, empty

    # Pairwise distances between gt and predicted coords
    diffs = gt_points.unsqueeze(1) - p_points.unsqueeze(0)
    distances = torch.sqrt((diffs * diffs).sum(dim=2))

    # Distances above threshold are not valid candidate matches
    invalid = distances.gt(dist_thresh)
    if bool(invalid.all()):
        return empty, empty

    # Coordinates in [0, 1], so max distance is sqrt(2)
    large_cost = (2.0 ** 0.5) + 1e-6
    cost = distances.masked_fill(invalid, large_cost)

    # Global one-to-one assignment minimizing total matching cost
    from scipy.optimize import linear_sum_assignment
    row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())

    row_idx_t = torch.as_tensor(row_idx, device=device, dtype=torch.long)
    col_idx_t = torch.as_tensor(col_idx, device=device, dtype=torch.long)

    # Keep only assignments that truly satisfy the threshold
    valid_matches = distances[row_idx_t, col_idx_t].le(dist_thresh)
    return row_idx_t[valid_matches], col_idx_t[valid_matches]


def evaluate_gt_p_match(
    gt_bars: torch.Tensor,
    gt_ticks: torch.Tensor,
    p_bars: torch.Tensor,
    p_ticks: torch.Tensor,
    dist_thresh: float
) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate predicted bars and ticks against ground truth.

    :param gt_bars: ground truth bars [y, x, value], shape [N, 3]
    :param gt_ticks: ground truth ticks [y, x, value], shape [M, 3]
    :param p_bars: predicted bars [y, x, conf], shape [N, 3]
    :param p_ticks: predicted ticks [y, x, conf], shape [M, 3]
    :param dist_thresh: max. distance threshold when considering matches
    :return: Tuple containing:

        - bar precision, recall, f1
        - tick precision, recall, f1
    """
    bar_rows, _ = hungarian_match(gt_bars[:, :2], p_bars[:, :2], dist_thresh)
    tick_rows, _ = hungarian_match(gt_ticks[:, :2], p_ticks[:, :2], dist_thresh)
    bar_matches = int(bar_rows.numel())
    tick_matches = int(tick_rows.numel())

    # Calculate precision and recall for bars
    bar_precision = (bar_matches / p_bars.size(0)) if p_bars.size(0) > 0 else 0.
    bar_recall = (bar_matches / gt_bars.size(0)) if gt_bars.size(0) > 0 else 0.
    bar_f1 = f1(bar_precision, bar_recall)

    # Calculate precision and recall for ticks
    tick_precision = (tick_matches / p_ticks.size(0)) if p_ticks.size(0) > 0 else 0.
    tick_recall = (tick_matches / gt_ticks.size(0)) if gt_ticks.size(0) > 0 else 0.
    tick_f1 = f1(tick_precision, tick_recall)

    return bar_precision, bar_recall, bar_f1, tick_precision, tick_recall, tick_f1


def evaluate_value_accuracy(
    gt_bar_yxv: torch.Tensor,
    p_bar_yxv: torch.Tensor,
    dist_thresh: float,
    hard_eps: float = 0.02,
    relaxed_eps: float = 0.05
) -> tuple[float, float]:
    """
    Evaluate bar-value accuracy with padded-count denominator.

    :param gt_bar_yxv: ground truth bars [N, 3] as [y, x, value]
    :param p_bar_yxv: predicted bars [M, 3] as [y, x, value]
    :param dist_thresh: max. spatial distance for matching bars
    :param hard_eps: strict relative error threshold
    :param relaxed_eps: relaxed relative error threshold
    :return: tuple containing:

        - hard-threshold accuracy
        - relaxed-threshold accuracy
    """
    gt_bars = gt_bar_yxv[:, :2]
    gt_values = gt_bar_yxv[:, 2]
    p_bars = p_bar_yxv[:, :2]
    p_values = p_bar_yxv[:, 2]

    # Ignore invalid gt values
    valid_gt_mask = torch.isfinite(gt_values)
    if not bool(valid_gt_mask.any()):
        return 0.0, 0.0

    p_count = int(min(p_bars.size(0), p_values.numel()))
    total = max(int(gt_bars[valid_gt_mask].size(0)), p_count)
    if total == 0 or p_count == 0:
        return 0.0, 0.0

    # Match predicted and gt bars using Hungarian assignment
    gt_idx, pred_idx = hungarian_match(
        gt_bars[valid_gt_mask],
        p_bars[:p_count],
        dist_thresh
    )
    if gt_idx.numel() == 0:
        return 0.0, 0.0

    # Criterion = abs(h_g - h_p) / h_g <= eps.
    gt_matched = gt_values[valid_gt_mask][gt_idx]
    p_matched = p_values[:p_count][pred_idx]
    denom = gt_matched.abs().clamp(min=1e-8)
    rel_err = (gt_matched - p_matched).abs() / denom

    # Defaults: hard = 0.02, relaxed = 0.05
    hard_correct = int(rel_err.le(hard_eps).sum().item())
    relaxed_correct = int(rel_err.le(relaxed_eps).sum().item())
    hard_acc = float(hard_correct / total)
    relaxed_acc = float(relaxed_correct / total)
    return hard_acc, relaxed_acc


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
