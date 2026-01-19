import os
import time

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import sys
import yaml
import wandb

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.utils.distributed import (
    init_distributed,
    AllReduce)

from src.utils.heatmap import (
    gt_maps_to_cls_lists,
    p_maps_to_cls_lists,
    evaluate_gt_p_match,
    nms,
    f1
)

from src.utils.logging import (
    CSVLogger,
    gpu_timer)

from src.datasets.charts import make_charts

from src.masks.charts import UBPMCCollator

from src.helper import (
    load_decoder_checkpoint,
    init_decoder_model,
    init_decoder_opt)

from src.transforms import make_transforms

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# Initialize logging and distributed training
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- MODEL
    model_name = args['meta']['model_name']
    decoder_type = args['meta']['decoder_type']
    r_file = args['meta']['read_checkpoint']
    use_bfloat16 = args['meta']['use_bfloat16']

    # -- DATA
    preserve_aspect_ratio = args['data']['preserve_aspect_ratio']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    annotation_folder = args['data']['annotation_folder']
    crop_size = args['data']['crop_size']
    patch_size = args['data']['patch_size']
    patch_count = int((crop_size // patch_size) ** 2.)

    # -- KEYPOINT DETECTION
    max_keypoints = args['keypoint']['max_keypoints']
    pnt_thresh = args['keypoint']['pnt_detect_thresh']
    cls_thresh = args['keypoint']['cls_conf_thresh']
    eval_thresh = args['keypoint']['eval_thresh']

    logger.info(f'Python version: {sys.version}, PyTorch version: {torch.__version__}')
    # ----------------------------------------------------------------------- #

    # -- create device
    if not torch.cuda.is_available():
        try:
            device = torch.device('cpu') if not torch.mps.is_available() else torch.device('mps')
        except AttributeError:
            device = torch.device('cpu')
        logger.warning(f'Falling back to {device.type}.')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- check bfloat16 support, otherwise fallback to float16
    try:
        autocast_dtype = torch.bfloat16 if (
            use_bfloat16 and (
                torch.cuda.is_bf16_supported() or
                torch.cpu._is_avx512_bf16_supported() or
                device.type == 'mps'
            )
        ) else (torch.float16 if use_bfloat16 else torch.float32)
    except Exception as e:
        logger.warning(f'Error checking bfloat16 support: {e}. Falling back to float16.')
        autocast_dtype = torch.float16 if use_bfloat16 else torch.float32

    # -- init model
    encoder, decoder = init_decoder_model(
        device=device,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=crop_size,
        max_keypoints=max_keypoints,
        decoder_type=decoder_type)

    # -- make data transforms
    transform = make_transforms(
        crop_size=crop_size,
        random_resize_crop=False,
        horizontal_flip=False,
        color_distortion=False,
        gaussian_blur=False,
        preserve_aspect_ratio=preserve_aspect_ratio,
        max_patches=patch_count,
        patch_size=patch_size)

    # -- init data-loaders/samplers
    collator = UBPMCCollator()
    train_loader, train_sampler, val_loader, val_sampler = make_charts( # type: ignore
            transform=transform,
            batch_size=batch_size,
            patch_size=patch_size,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            root_path=root_path,
            image_folder=image_folder,
            annotation_folder=annotation_folder,
            val_train_split=True,
            training=True,
            drop_last=False,
            shuffle=True)
    ipe = len(train_loader)

    # -- load training checkpoint
    encoder, decoder, _, _, _ = load_decoder_checkpoint(
        world_size=1,
        do_finetune=False,
        r_path=r_file,
        encoder=encoder,
        decoder=decoder,
        opt=None,
        scaler=None)

    # -- TEST
    for epoch in range(1):
        # Update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        def load_charts(data, targets):
            gt_orgs, gt_cls, gt_reg = targets
            imgs = [img.to(device, non_blocking=True) for img in data]
            grids = [(img.shape[1] // patch_size, img.shape[2] // patch_size) for img in imgs]
            gt_orgs = [gt.to(device, non_blocking=True) for gt in gt_orgs]
            gt_cls = [gt.to(device, non_blocking=True) for gt in gt_cls]
            gt_reg = [gt.to(device, non_blocking=True) for gt in gt_reg]
            return (imgs, grids, gt_orgs, gt_cls, gt_reg)

        def step(imgs, grids, gt_orgs, gt_cls, gt_reg):

            def forward():
                # Charts -> Embeddings
                h = encoder(imgs, grids)
                # Embeddings -> Predictions
                p_cls, p_reg, p_kps = decoder(h, grids)
                return p_cls, p_reg, p_kps

            # Forward
            with torch.amp.autocast(device.type, dtype=autocast_dtype, enabled=use_bfloat16):
                sizes = torch.tensor([c.shape for c in gt_cls], device=device)

                with torch.no_grad():
                    p_cls, p_reg, p_kps = forward()

                # For each chart in batch individually
                for i in range(len(p_cls)):
                    # Extract ground truth labels
                    gt_org, gt_bars, gt_ticks = gt_maps_to_cls_lists(
                        gt_orgs[i], gt_cls[i], gt_reg[i], sizes[i]
                    )

                    # Keypoint coordinates & probabilities
                    kp_coords = p_kps[i][:, :2]
                    kp_logits = p_kps[i][:, 2:]

                    # Origin coordinate loss
                    origin_probs = torch.sigmoid(kp_logits[:, 3])
                    origin_weights = origin_probs / (origin_probs.sum() + 1e-8)
                    p_org_w = (origin_weights.unsqueeze(1) * kp_coords).sum(dim=0)

                    # Radius for nms and l_pts is based on max(H, W)
                    radius_thresh = eval_thresh / sizes[i].max()
                    # Convert maps to lists of bars & ticks + origin
                    p_bars, p_ticks, p_org = p_maps_to_cls_lists(p_cls[i], p_reg[i], sizes[i], pnt_thresh, cls_thresh)
                    # Filter predictions using nms
                    p_bars, p_ticks = nms(p_bars, p_ticks, radius_thresh)

                    # Evalutate
                    b_p, b_r, t_p, t_r = evaluate_gt_p_match(
                        gt_bars, gt_ticks,
                        p_bars, p_ticks,
                        radius_thresh) # type: ignore

        decoder.eval()
        for _, (data, targets) in enumerate(val_loader):
            imgs, grids, gt_orgs, gt_cls, gt_reg = load_charts(data, targets)
            step(imgs, grids, gt_orgs, gt_cls, gt_reg)

if __name__ == '__main__':
    main({})
