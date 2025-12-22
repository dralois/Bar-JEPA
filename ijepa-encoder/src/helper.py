# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

from collections import OrderedDict

import torch

import src.models.vision_transformer as vit
from src.models.decoders import KeypointDetector
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def convert_ddp(checkpoint_dict, do_convert):
    if not do_convert:
        return checkpoint_dict

    checkpoint_cpy = OrderedDict()
    for k, v in checkpoint_dict.items():
        name = k[len('module:'):]  # remove 'module.' of DataParallel/DistributedDataParallel
        checkpoint_cpy[name] = v
    return checkpoint_cpy


def load_ijepa_checkpoint(
    world_size,
    do_finetune,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        print(list(checkpoint.keys()))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = convert_ddp(checkpoint['encoder'], world_size == 1)
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        if predictor is not None:
            pretrained_dict = convert_ddp(checkpoint['predictor'], world_size == 1)
            msg = predictor.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            pretrained_dict = convert_ddp(checkpoint['target_encoder'], world_size == 1)
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None and not do_finetune:
            pretrained_dict = checkpoint['opt']
            msg = opt.load_state_dict(pretrained_dict)
            logger.info(f'loaded optimizer from epoch {epoch} with msg: {msg}')

        # -- loading scaler
        if scaler is not None and not do_finetune:
            pretrained_dict = checkpoint['scaler']
            msg = scaler.load_state_dict(pretrained_dict)
            logger.info(f'loaded scaler from epoch {epoch} with msg: {msg}')

        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def load_decoder_checkpoint(
    world_size,
    do_finetune,
    r_path,
    encoder,
    decoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        print(list(checkpoint.keys()))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = convert_ddp(checkpoint['encoder'], world_size == 1)
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading decoder
        pretrained_dict = convert_ddp(checkpoint['decoder'], world_size == 1)
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained decoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None and not do_finetune:
            pretrained_dict = checkpoint['opt']
            msg = opt.load_state_dict(pretrained_dict)
            logger.info(f'loaded optimizer from epoch {epoch} with msg: {msg}')

        # -- loading scaler
        if scaler is not None and not do_finetune:
            pretrained_dict = checkpoint['scaler']
            msg = scaler.load_state_dict(pretrained_dict)
            logger.info(f'loaded scaler from epoch {epoch} with msg: {msg}')

        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, decoder, opt, scaler, epoch


def init_ijepa_model(
    device,
    model_name='vit_base',
    patch_size=16,
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    max_patches = (crop_size // patch_size) ** 2
    encoder = vit.__dict__[model_name](
        max_patches=max_patches,
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        max_patches=max_patches,
        patch_size=patch_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_decoder_model(
    device,
    model_name='vit_base',
    patch_size=16,
    crop_size=224,
    max_keypoints=64,
    decoder_type='simple'
):
    max_patches = (crop_size // patch_size) ** 2
    encoder = vit.__dict__[model_name](
        max_patches=max_patches,
        patch_size=patch_size)
    decoder = KeypointDetector(
        max_patches=max_patches,
        in_channels=encoder.num_features,
        num_keypoints=max_keypoints,
        num_classes=3,
        decoder_type=decoder_type)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in decoder.modules():
        init_weights(m)

    encoder.to(device)
    decoder.to(device)
    logger.info(encoder)
    logger.info(decoder)
    return encoder, decoder


def init_ijepa_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_decoder_opt(
    decoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in decoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in decoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
