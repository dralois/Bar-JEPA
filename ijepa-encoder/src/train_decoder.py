import os
import time
import logging
import sys
import yaml
import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# Initialize logging and distributed training
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # Model and data parameters
    batch_size = args['data']['batch_size']
    num_epochs = args['optimization']['epochs']
    learning_rate = args['optimization']['lr']
    checkpoint_dir = args['logging']['folder']
    tag = args['logging']['write_tag']

    # Initialize model
    model = CombinedKeypointDetector(in_channels=1280, num_keypoints=64, num_classes=3, decoder_type='simple')
    model = model.to(device)
    if world_size != 1:
        model = DistributedDataParallel(model, static_graph=True)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Load checkpoint if needed
    start_epoch = 0
    if args['meta']['load_checkpoint']:
        load_path = os.path.join(checkpoint_dir, args['meta']['read_checkpoint'])
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']

    # Dataloaders
    train_dataloader = get_dataloaders(args['data'], training=True)
    val_dataloader = get_dataloaders(args['data'], training=False)

    # Initialize wandb
    run = wandb.init(
        entity='bar-ijepa',
        mode='offline',
        config={
            'learning-rate': learning_rate,
            'epochs': num_epochs
        }
    )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}')

        # Set model to train mode
        model.train()
        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (img, targets) in enumerate(train_dataloader):
            img = img.to(device, non_blocking=True)
            gt_cls_map, gt_reg_map = targets
            gt_cls_map = gt_cls_map.to(device, non_blocking=True)
            gt_reg_map = gt_reg_map.to(device, non_blocking=True)

            def train_step():
                def forward():
                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        pts_cls_pred, pts_reg_pred = model(img)

                        # Compute losses
                        class_weights = torch.tensor([0.05, 1., 1.]).to(device)
                        pts_cls_loss = F.cross_entropy(pts_cls_pred, gt_cls_map.long(), weight=class_weights)

                        # Masked regression loss
                        gt_reg_list = torch.masked_select(gt_reg_map.permute(1, 0, 2, 3), gt_cls_map.gt(0))
                        pred_reg_list = torch.masked_select(pts_reg_pred.permute(1, 0, 2, 3), gt_cls_map.gt(0))
                        pts_reg_loss = F.mse_loss(pred_reg_list, gt_reg_list)

                        total_loss = pts_cls_loss + pts_reg_loss
                        return total_loss, pts_cls_loss, pts_reg_loss

                # Forward pass
                total_loss, pts_cls_loss, pts_reg_loss = forward()

                # Backward pass
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                return total_loss.item(), pts_cls_loss.item(), pts_reg_loss.item()

            loss, cls_loss, reg_loss = train_step()

            # Update meters
            loss_meter.update(loss)
            cls_loss_meter.update(cls_loss)
            reg_loss_meter.update(reg_loss)

            # Log stats
            if itr % args['logging']['log_freq'] == 0:
                logger.info(f'[{epoch + 1}, {itr}] loss: {loss_meter.avg:.3f} '
                            f'cls_loss: {cls_loss_meter.avg:.3f} '
                            f'reg_loss: {reg_loss_meter.avg:.3f} '
                            f'[mem: {torch.cuda.max_memory_allocated() / 1024.**2:.2e}]')

                run.log({
                    'epoch': epoch + 1,
                    'train-loss': loss_meter.avg,
                    'cls-loss': cls_loss_meter.avg,
                    'reg-loss': reg_loss_meter.avg,
                    'gpu-mem': torch.cuda.max_memory_allocated() / 1024.**2
                })

        # Validation loop
        model.eval()
        val_loss_meter = AverageMeter()
        val_cls_loss_meter = AverageMeter()
        val_reg_loss_meter = AverageMeter()

        with torch.no_grad():
            for itr, (img, targets) in enumerate(val_dataloader):
                img = img.to(device, non_blocking=True)
                gt_cls_map, gt_reg_map = targets
                gt_cls_map = gt_cls_map.to(device, non_blocking=True)
                gt_reg_map = gt_reg_map.to(device, non_blocking=True)

                pts_cls_pred, pts_reg_pred = model(img)

                # Compute losses
                class_weights = torch.tensor([0.05, 1., 1.]).to(device)
                pts_cls_loss = F.cross_entropy(pts_cls_pred, gt_cls_map.long(), weight=class_weights)

                # Masked regression loss
                gt_reg_list = torch.masked_select(gt_reg_map.permute(1, 0, 2, 3), gt_cls_map.gt(0))
                pred_reg_list = torch.masked_select(pts_reg_pred.permute(1, 0, 2, 3), gt_cls_map.gt(0))
                pts_reg_loss = F.mse_loss(pred_reg_list, gt_reg_list)

                total_loss = pts_cls_loss + pts_reg_loss

                # Update meters
                val_loss_meter.update(total_loss.item())
                val_cls_loss_meter.update(pts_cls_loss.item())
                val_reg_loss_meter.update(pts_reg_loss.item())

        # Log validation stats
        logger.info(f'Validation - loss: {val_loss_meter.avg:.3f} '
                    f'cls_loss: {val_cls_loss_meter.avg:.3f} '
                    f'reg_loss: {val_reg_loss_meter.avg:.3f}')

        run.log({
            'val-loss': val_loss_meter.avg,
            'val-cls-loss': val_cls_loss_meter.avg,
            'val-reg-loss': val_reg_loss_meter.avg
        })

        # Save checkpoint
        if (epoch + 1) % args['logging']['checkpoint_freq'] == 0 and rank == 0:
            checkpoint_name = f'ppn_chk_epoch_{epoch+1:04}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': loss_meter.avg
            }, checkpoint_path)

    run.finish()

if __name__ == '__main__':
    main()
