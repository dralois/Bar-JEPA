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


def evaluate_confusion(
    gt_bars: torch.Tensor,
    gt_ticks: torch.Tensor,
    p_bars: torch.Tensor,
    p_ticks: torch.Tensor,
    dist_thresh: float
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Returns a full confusion matrix for bar/tick keypoint detection.

                     p_bar       p_tick       p_none
        gt_bar       tp_bar      bar_as_tick  fn_bar
        gt_tick      tick_as_bar tp_tick      fn_tick
        gt_none      fp_bar      fp_tick      –

    :param gt_bars: ground truth bars [y, x, value], shape [N, 3]
    :param gt_ticks: ground truth ticks [y, x, value], shape [M, 3]
    :param p_bars: predicted bars [y, x, conf], shape [P, 3]
    :param p_ticks: predicted ticks [y, x, conf], shape [Q, 3]
    :param dist_thresh: max. distance threshold for matching
    :return: tuple containing:

        - tp_bar: GT bars matched with predicted bars
        - bar_as_tick: GT bars matched with predicted ticks (cross-class)
        - fn_bar: GT bars with no match
        - fp_bar: predicted bars with no GT match
        - tp_tick: GT ticks matched with predicted ticks
        - tick_as_bar: GT ticks matched with predicted bars (cross-class)
        - fn_tick: GT ticks with no match
        - fp_tick: predicted ticks with no GT match
    """
    device = gt_bars.device
    n_gt_bars = gt_bars.size(0)
    n_gt_ticks = gt_ticks.size(0)
    n_p_bars = p_bars.size(0)
    n_p_ticks = p_ticks.size(0)

    empty = torch.empty(0, dtype=torch.long, device=device)

    # --- Pass 1: within-class matching ---
    if n_gt_bars > 0 and n_p_bars > 0:
        gt_bar_idx, p_bar_idx = hungarian_match(gt_bars[:, :2], p_bars[:, :2], dist_thresh)
    else:
        gt_bar_idx, p_bar_idx = empty, empty

    if n_gt_ticks > 0 and n_p_ticks > 0:
        gt_tick_idx, p_tick_idx = hungarian_match(gt_ticks[:, :2], p_ticks[:, :2], dist_thresh)
    else:
        gt_tick_idx, p_tick_idx = empty, empty

    tp_bar = int(gt_bar_idx.numel())
    tp_tick = int(gt_tick_idx.numel())

    # Build masks of unmatched points after pass 1
    unmatched_gt_bar = torch.ones(n_gt_bars, dtype=torch.bool, device=device)
    unmatched_p_bar = torch.ones(n_p_bars, dtype=torch.bool, device=device)
    unmatched_gt_tick = torch.ones(n_gt_ticks, dtype=torch.bool, device=device)
    unmatched_p_tick = torch.ones(n_p_ticks, dtype=torch.bool, device=device)

    if gt_bar_idx.numel() > 0:
        unmatched_gt_bar[gt_bar_idx] = False
        unmatched_p_bar[p_bar_idx] = False
    if gt_tick_idx.numel() > 0:
        unmatched_gt_tick[gt_tick_idx] = False
        unmatched_p_tick[p_tick_idx] = False

    # --- Pass 2: cross-class matching on remaining unmatched points ---
    ub_gt_bar = gt_bars[unmatched_gt_bar, :2]
    ub_p_tick = p_ticks[unmatched_p_tick, :2]
    ub_gt_tick = gt_ticks[unmatched_gt_tick, :2]
    ub_p_bar = p_bars[unmatched_p_bar, :2]

    # Unmatched GT bars vs unmatched pred ticks
    if ub_gt_bar.size(0) > 0 and ub_p_tick.size(0) > 0:
        xm_gt_bar, xm_p_tick = hungarian_match(ub_gt_bar, ub_p_tick, dist_thresh)
    else:
        xm_gt_bar, xm_p_tick = empty, empty

    bar_as_tick = int(xm_gt_bar.numel())

    # Unmatched GT ticks vs unmatched pred bars
    if ub_gt_tick.size(0) > 0 and ub_p_bar.size(0) > 0:
        xm_gt_tick, xm_p_bar = hungarian_match(ub_gt_tick, ub_p_bar, dist_thresh)
    else:
        xm_gt_tick, xm_p_bar = empty, empty

    tick_as_bar = int(xm_gt_tick.numel())

    # Update masks based on cross-class matches
    unmatched_gt_bar_idx = unmatched_gt_bar.nonzero(as_tuple=False).squeeze(1)
    unmatched_p_tick_idx = unmatched_p_tick.nonzero(as_tuple=False).squeeze(1)
    unmatched_gt_tick_idx = unmatched_gt_tick.nonzero(as_tuple=False).squeeze(1)
    unmatched_p_bar_idx = unmatched_p_bar.nonzero(as_tuple=False).squeeze(1)

    if xm_gt_bar.numel() > 0:
        unmatched_gt_bar[unmatched_gt_bar_idx[xm_gt_bar]] = False
        unmatched_p_tick[unmatched_p_tick_idx[xm_p_tick]] = False
    if xm_gt_tick.numel() > 0:
        unmatched_gt_tick[unmatched_gt_tick_idx[xm_gt_tick]] = False
        unmatched_p_bar[unmatched_p_bar_idx[xm_p_bar]] = False

    fn_bar = int(unmatched_gt_bar.sum())
    fn_tick = int(unmatched_gt_tick.sum())
    fp_bar = int(unmatched_p_bar.sum())
    fp_tick = int(unmatched_p_tick.sum())

    return tp_bar, bar_as_tick, fn_bar, fp_bar, tp_tick, tick_as_bar, fn_tick, fp_tick


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
