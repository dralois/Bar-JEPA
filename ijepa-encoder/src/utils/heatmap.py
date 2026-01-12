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


def keypoint_sets(
    gt_coords: torch.Tensor,
    kp_coords: torch.Tensor,
    kp_logits: torch.Tensor,
    cls_id: int,
    # Hyperparameters
    tau: float = 0.04,
    sigma: float = 0.12,
    lambda_missing: float = 1.0,
    lambda_claim: float = 1.0,
    lambda_bg: float = 0.5
) -> torch.Tensor:
    """
    _summary_

    :param gt_coords: _description_
    :param kp_coords: _description_
    :param kp_logits: _description_
    :param cls_id: _description_
    :param tau: Soft assignment temperature
    :param sigma: _description_, defaults to 0.12
    :param lambda_missing: _description_, defaults to 1.0
    :param lambda_claim: _description_, defaults to 1.0
    :param lambda_bg: _description_, defaults to 0.5
    :return: _description_
    """
    # Calculate pairwise distances
    diff = gt_coords.unsqueeze(1) - kp_coords
    dists = torch.sqrt((diff * diff).sum(dim=2) + 1e-8)

    # Soft assign keypoints to ground truths
    assign_kp_to_gt = F.softmin(dists / tau, dim=1)
    soft_dist_gt = (assign_kp_to_gt * dists).sum(dim=1)
    # Punish ground truths missing from predictions
    coverage = torch.exp(-(soft_dist_gt / sigma) ** 2)
    missing_loss = (1.0 - coverage).mean()

    # Punish predictions being far away from ground truth
    dist_loss = soft_dist_gt.mean()

    # Punish ground truths not being claimed by predictions
    cls_prob = torch.softmax(kp_logits[:, :3], dim=1)[:, cls_id]
    gt_cls_conf = (assign_kp_to_gt * cls_prob).sum(dim=1)
    claim_loss = (1.0 - gt_cls_conf).mean()

    # Punish keypoints that should be background having the wrong class
    assign_gt_to_kp = F.softmin(dists / tau, dim=0)
    soft_dist_kp = (assign_gt_to_kp * dists).sum(dim=0)
    bg_target = torch.sigmoid((soft_dist_kp - sigma) / (0.25 * sigma))
    bg_loss = F.binary_cross_entropy_with_logits(
        kp_logits[:, 0],
        bg_target.detach()
    )

    logger.info('Loss %s:\t[%.3f + %.3f + %.3f + %.3f]',
                'bars' if cls_id == 1 else 'ticks', dist_loss,
                lambda_missing * missing_loss,
                lambda_claim * claim_loss,
                lambda_bg * bg_loss)

    return (
        dist_loss
        + lambda_missing * missing_loss
        + lambda_claim * claim_loss
        + lambda_bg * bg_loss)


def f1(
    precision: float,
    recall: float
) -> float:
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.
