import os

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
from torch.utils.data import random_split

from src.utils.distributed import (
    init_distributed,
    AllReduce)

from src.utils.heatmap import (
    gt_maps_to_cls_lists,
    p_maps_to_cls_lists,
    nms,
    evaluate_gt_p_match,
    f1
)

from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    AverageMeter)

from src.datasets.charts import make_charts

from src.masks.default import DefaultCollator

from src.helper import (
    load_decoder_checkpoint,
    init_decoder_model,
    init_decoder_opt)

from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

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

    # TODO
    # -- MODEL
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    decoder_type = args['meta']['decoder_type']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    ckpt_epoch = args['meta']['checkpoint_epoch']
    do_finetune = args['meta']['do_finetune']
    r_file = args['meta']['read_checkpoint']

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
    cls_weights = args['keypoint']['class_weights']

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-decoder.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    # -- create device
    if not torch.cuda.is_available():
        device = torch.device('cpu') if not torch.mps.is_available() else torch.device('mps')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- check bfloat16 support, otherwise fallback to float16
    try:
        autocast_dtype = torch.bfloat16 if (
            use_bfloat16 and (
                torch.cuda.is_bf16_supported() or
                torch.cpu._is_avx512_bf16_supported()
            )
        ) else (torch.float16 if use_bfloat16 else torch.float32)
    except Exception as e:
        logger.warning(f'Error checking bfloat16 support: {e}. Falling back to float16')
        autocast_dtype = torch.float16 if use_bfloat16 else torch.float32

    # -- dataloader
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%d', 'time (ms)'))

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
    collator = DefaultCollator()
    train_loader, train_sampler, val_loader, val_sampler = make_charts(
            transform=transform,
            batch_size=batch_size,
            patch_size=patch_size,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            annotation_folder=annotation_folder,
            val_train_split=True,
            training=True,
            drop_last=True)
    ipe = len(train_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_decoder_opt(
        decoder=decoder,
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16,
        ipe_scale=ipe_scale)

    if world_size != 1:
        decoder = DistributedDataParallel(decoder)
        encoder = DistributedDataParallel(encoder)

    # Encoder is frozen for decoder training
    for p in encoder.parameters():
        p.requires_grad = False

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, decoder, optimizer, scaler, start_epoch = load_decoder_checkpoint(
            world_size=world_size,
            do_finetune=do_finetune,
            r_path=load_path,
            encoder=encoder,
            decoder=decoder,
            opt=optimizer,
            scaler=scaler)
        if do_finetune:
            start_epoch -= (ckpt_epoch - 1)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_train_m.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # Class weights: [None, bar, tick]
    cls_weights = torch.tensor(cls_weights).to(device)

    # Initialize wandb
    run = wandb.init(
        entity='bar-ijepa',
        project='bar-ijepa-detector',
        mode='offline',
        config={
            'learning-rate': lr,
            'epochs': num_epochs
        }
    )

    # -- TRAIN / VAL LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}')

        # Update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # Loss tracking
        loss_train_m, loss_val_m = AverageMeter(), AverageMeter()
        org_loss_train_m, org_loss_val_m = AverageMeter(), AverageMeter()
        cls_loss_train_m, cls_loss_val_m = AverageMeter(), AverageMeter()
        reg_loss_train_m, reg_loss_val_m = AverageMeter(), AverageMeter()
        pts_loss_train_m, pts_loss_val_m = AverageMeter(), AverageMeter()
        alg_loss_train_m, alg_loss_val_m = AverageMeter(), AverageMeter()
        time_meter = AverageMeter()

        def load_charts(data, targets):
            gt_org, gt_cls, gt_reg = targets
            imgs = [img.to(device, non_blocking=True) for img in data]
            grids = [(img.shape[1] // patch_size, img.shape[2] // patch_size) for img in imgs]
            gt_org = [gt.to(device, non_blocking=True) for gt in gt_org]
            gt_cls = [gt.to(device, non_blocking=True) for gt in gt_cls]
            gt_reg = [gt.to(device, non_blocking=True) for gt in gt_reg]
            return (imgs, grids, gt_org, gt_cls, gt_reg)

        def log_stats(epoch, itr, loss, etime, train):
            csv_logger.log(epoch + 1, itr, loss, etime)
            if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                logger.info('%s: [%d, %5d] loss: %.3f '
                            '[%.3f + %.3f + %.3f + %.3f + %.3f] '
                            '[mem: %.2e] '
                            '(%.1f ms)'
                            % ('Train' if train else 'Val', epoch + 1, itr,
                                loss_train_m.avg if train else loss_val_m.avg,
                                org_loss_train_m.avg if train else org_loss_val_m.avg,
                                cls_loss_train_m.avg if train else cls_loss_val_m.avg,
                                reg_loss_train_m.avg if train else reg_loss_val_m.avg,
                                pts_loss_train_m.avg if train else pts_loss_val_m.avg,
                                alg_loss_train_m.avg if train else alg_loss_val_m.avg,
                                torch.cuda.max_memory_allocated() / 1024.**2,
                                time_meter.avg))

        def log_wandb(epoch, train):
            run.log({
                'epoch': epoch + 1,
                f'loss-{"train" if train else "val"}/total': loss_train_m.avg if train else loss_val_m.avg,
                f'loss-{"train" if train else "val"}/origin': org_loss_train_m.avg if train else org_loss_val_m.avg,
                f'loss-{"train" if train else "val"}/classification': cls_loss_train_m.avg if train else cls_loss_val_m.avg,
                f'loss-{"train" if train else "val"}/regression': reg_loss_train_m.avg if train else reg_loss_val_m.avg,
                f'loss-{"train" if train else "val"}/points': pts_loss_train_m.avg if train else pts_loss_val_m.avg,
                f'loss-{"train" if train else "val"}/align': alg_loss_train_m.avg if train else alg_loss_val_m.avg,
                'gpu-mem': torch.cuda.max_memory_allocated() / 1024.**2
            })

        def step(imgs, grids, gt_org, gt_cls, gt_reg):
            # Update learning rate
            if decoder.training:
                scheduler.step()
                wd_scheduler.step()

            def forward():
                # Charts -> Embeddings
                h = encoder(imgs, grids)
                # Embeddings -> Predictions
                p_org, p_cls, p_reg = decoder(h, grids)
                return p_org, p_cls, p_reg

            def loss_fn(p_org, gt_org, p_cls, gt_cls, p_reg, gt_reg):
                l_org = torch.tensor(0., device=device)
                l_cls = torch.tensor(0., device=device)
                l_reg = torch.tensor(0., device=device)
                l_pts = torch.tensor(0., device=device)
                l_align = torch.tensor(0., device=device)
                sizes = torch.tensor([c.shape for c in gt_cls], device=device)

                # For each chart in batch individually
                for i in range(len(p_org)):
                    # Coordinate system origin loss
                    l_org += F.smooth_l1_loss(p_org[i], gt_org[i])

                    # Weighted classification loss
                    l_cls += F.cross_entropy(p_cls[i].unsqueeze(0), gt_cls[i].unsqueeze(0), weight=cls_weights, reduction='mean')

                    # Regression loss only on non-background samples
                    gt_reg_masked = torch.masked_select(gt_reg[i], gt_cls[i].gt(0))
                    p_reg_masked = torch.masked_select(p_reg[i], gt_cls[i].gt(0))
                    l_reg += F.mse_loss(p_reg_masked, gt_reg_masked)

                    # Radius for nms and l_pts is based on max(H, W)
                    radius_thresh = eval_thresh / sizes[i].max()
                    # Convert maps to lists of bars & ticks
                    gt_bars, gt_ticks = gt_maps_to_cls_lists(gt_cls[i], gt_reg[i], sizes[i])
                    p_bars, p_ticks = p_maps_to_cls_lists(p_cls[i], p_reg[i], sizes[i], pnt_thresh, cls_thresh)
                    # Filter predictions using nms
                    p_bars, p_ticks = nms(p_bars, p_ticks, radius_thresh)

                    # Within class point loss
                    l_pts += evaluate_gt_p_match(gt_bars, gt_ticks, p_bars, p_ticks, radius_thresh, True)

                    # Chart specific loss: Aligns tick x coordinates
                    if len(p_ticks) > 0:
                        l_align += (torch.stack(p_ticks)[:,1] - p_org[i][1]).abs().sum()

                # Weight losses according to scaling factors
                l_pts /= 10.
                l_align /= 1000.
                loss = l_org + l_cls + l_reg + l_pts + l_align
                loss = AllReduce.apply(loss)

                return loss, (float(loss.item()), l_org.item(), l_cls.item(), l_reg.item(), l_pts.item(), l_align.item())

            # Forward
            with torch.amp.autocast(device.type, dtype=autocast_dtype, enabled=use_bfloat16):
                if decoder.training:
                    p_org, p_cls, p_reg = forward()
                    loss, all_losses = loss_fn(p_org, gt_org, p_cls, gt_cls, p_reg, gt_reg)
                else:
                    with torch.no_grad():
                        p_org, p_cls, p_reg = forward()
                        loss, all_losses = loss_fn(p_org, gt_org, p_cls, gt_cls, p_reg, gt_reg)

            # Backward & step
            if decoder.training:
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

            return all_losses

        def update_meters(losses, train):
            loss, l_org, l_cls, l_reg, l_pts, l_alg = losses
            (loss_train_m if train else loss_val_m).update(loss)
            (org_loss_train_m if train else org_loss_val_m).update(l_org)
            (cls_loss_train_m if train else cls_loss_val_m).update(l_cls)
            (reg_loss_train_m if train else reg_loss_val_m).update(l_reg)
            (pts_loss_train_m if train else pts_loss_val_m).update(l_pts)
            (alg_loss_train_m if train else alg_loss_val_m).update(l_alg)
            return loss

        # Train loop
        decoder.train()
        for itr, (data, targets) in enumerate(train_loader):

            imgs, grids, gt_org, gt_cls, gt_reg = load_charts(data, targets)

            losses, etime = gpu_timer(
                step,
                imgs=imgs,
                grids=grids,
                gt_org=gt_org,
                gt_cls=gt_cls,
                gt_reg=gt_reg
            )

            loss = update_meters(losses, True)
            time_meter.update(etime)

            log_stats(epoch, itr, loss, etime, True)
            log_wandb(epoch, True)

            assert not np.isnan(loss), 'loss is nan'

        # Validation loop
        decoder.eval()
        for itr, (data, targets) in enumerate(val_loader):

            imgs, grids, gt_org, gt_cls, gt_reg = load_charts(data, targets)

            losses, etime = gpu_timer(
                step,
                imgs=imgs,
                grids=grids,
                gt_org=gt_org,
                gt_cls=gt_cls,
                gt_reg=gt_reg
            )

            loss = update_meters(losses, False)
            time_meter.update(etime)

            log_stats(epoch, itr, loss, etime, False)
            log_wandb(epoch, False)

        # TODO only save if lower validation loss?
        # -- Save Checkpoint after every epoch
        # save_checkpoint(epoch+1)

    run.finish()

if __name__ == '__main__':
    main()
