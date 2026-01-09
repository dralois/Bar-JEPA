import torch
import torch.nn.functional as F

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts ground truth class and regression maps
    to lists of bar and tick positions.

    :param gt_cls: ground truth class map, shape [H, W]
    :param gt_reg: ground truth regression map, shape [2, H, W]
    :param size: size of the map, shape [2]
    :return: Tuple containing:

        - bars, [N, 2] -> [y, x] in image space
        - ticks, [M, 2] -> [y, x] in image space
    """
    # Get all non-background points
    points = torch.nonzero(gt_cls > 0)

    # Pixel midpoints in image space
    pos = (points * 2 + 1) / (size * 2)
    # Apply regression offsets
    pos += gt_reg[:, points[:, 0], points[:, 1]].T / size

    # Class labels for each point
    cls = gt_cls[points[:, 0], points[:, 1]]

    # Split into bars and ticks
    bars  = pos[cls == 1]
    ticks = pos[cls == 2]

    return bars, ticks


def keypoint_sets(
    gt: torch.Tensor,          # [N, 2]
    slot_coords: torch.Tensor, # [K, 2]
    slot_conf: torch.Tensor,   # [K] class probability
    tau=0.05,
    lambda_missing=1.0,
    lambda_extra=0.5,
) -> torch.Tensor:
    """
    Set loss with:
      - GT → slot matching (coverage)
      - slot → GT repulsion (false positives)
    """

    # --------------------------------------------------
    # Pairwise distances [N, K]
    # --------------------------------------------------
    diff = gt[:, None, :] - slot_coords[None, :, :]   # [N, K, 2]
    dists = torch.sqrt((diff * diff).sum(dim=2) + 1e-8)

    # --------------------------------------------------
    # GT → slot (coverage, missing penalty)
    # --------------------------------------------------
    weights_gt = torch.softmax(-dists / tau, dim=1)
    matched_dist = (weights_gt * dists).sum(dim=1)

    coverage = weights_gt.sum(dim=1).clamp(max=1.0)
    missing_loss = (1.0 - coverage).mean()

    # --------------------------------------------------
    # slot → GT (extra points penalty)
    # --------------------------------------------------
    weights_slot = torch.softmax(-dists / tau, dim=0)
    slot_to_gt_dist = (weights_slot * dists).sum(dim=0)

    extra_loss = (slot_conf * slot_to_gt_dist).mean()

    return (
        matched_dist.mean()
        + lambda_missing * missing_loss
        + lambda_extra * extra_loss
    )


def f1(
    precision: float,
    recall: float
) -> float:
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
