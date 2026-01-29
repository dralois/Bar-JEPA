# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch,
    pack_by_masks
)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of the grid height and width
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim] or [1+grid_size[0]*grid_size[1], embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attn = attn.masked_fill(~attention_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attention_mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), attention_mask=attention_mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, num_patches=14*14, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        C, H, W = x.shape
        # e.g. [3, 224, 224] -> [768, 14, 14] -> [768, 196] -> [196, 768] -> [1, 196, 768]
        x = self.proj(x).flatten(1).transpose(0, 1).unsqueeze(0)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        max_patches,
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        predictor_embed_dim=384,
        **kwargs
    ):
        super().__init__()
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, max_patches, predictor_embed_dim),
            requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            (int(max_patches**.5), int(max_patches**.5)),
            cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, grids, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        if len(masks_x) == 0 or len(masks) == 0:
            raise ValueError('masks_x and masks must be non-empty')
        if not isinstance(masks_x[0], list) or not isinstance(masks[0], list):
            raise ValueError('masks_x and masks must be [B x nmasks] lists')
        if len(masks_x) != len(masks):
            raise ValueError('masks_x and masks must have the same batch size')

        nenc = len(masks_x[0])
        npred = len(masks[0])

        # -- Context encoder embeddings [B*nenc, N, D-enc] to predictor [B*nenc, N, D-pred] embeddings
        x = self.predictor_embed(x)

        B_ctx = len(masks_x)
        B_total, _, D = x.size()
        if B_ctx * nenc != B_total:
            raise ValueError('masks_x does not match batch size of x')

        # -- add positional embedding to context tokens
        pos_embed_full = self.interpolate_pos_encoding(
            torch.zeros((B_ctx, self.max_patches, D), device=x.device),
            self.predictor_pos_embed,
            grids
        )
        pos_ctx, ctx_attn = pack_by_masks(pos_embed_full, masks_x)
        x = x + pos_ctx

        L_ctx = x.size(1)

        # -- Create position embeddings for prediction tokens
        pos_pred, pred_attn = pack_by_masks(pos_embed_full, masks)
        pos_pred = repeat_interleave_batch(pos_pred, B_ctx, repeat=nenc)
        pred_attn = repeat_interleave_batch(pred_attn.unsqueeze(-1), B_ctx, repeat=nenc).squeeze(-1)

        # -- Create prediction tokens, add position embeddings
        pred_tokens = self.mask_token.repeat(pos_pred.size(0), pos_pred.size(1), 1)
        pred_tokens += pos_pred
        pred_tokens *= pred_attn.unsqueeze(-1)

        # -- Repeat context for each prediction mask
        x = x.repeat(npred, 1, 1)
        ctx_attn = ctx_attn.repeat(npred, 1)

        # -- Concatenate context and prediction tokens
        x = torch.cat([x, pred_tokens], dim=1)
        attention_mask = torch.cat([ctx_attn, pred_attn], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, attention_mask=attention_mask)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x * attention_mask.unsqueeze(-1)
        x = x[:, L_ctx:]
        x = self.predictor_proj(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed, grid_sizes):
        B, N, D = x.shape
        grid_h = grid_w = int(math.sqrt(pos_embed.shape[1]))

        # Position embeddings [D, N] -> [1, D, sqrt(N), sqrt(N)]
        pos_embed_2d = pos_embed.reshape(1, D, grid_h, grid_w)
        output = torch.zeros(B, N, D, device=x.device)

        # Interpolate embeddings for each patch grid size
        for i in range(B):
            h, w = grid_sizes[i]
            pos_embed_interp = nn.functional.interpolate(
                pos_embed_2d,
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
            # [1, D, H, W] -> [1, H * W, D]
            pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, h * w, D)
            output[i, :h*w, :] = pos_embed_interp

        return output


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        max_patches,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        in_chans=3,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            num_patches=max_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # --
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_patches, embed_dim),
            requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (int(max_patches**.5), int(max_patches**.5)),
            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, grids, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x [B, C, W, H] -> [B, N, D]
        x = [self.patch_embed(curr) for curr in x]
        B, N, D = len(x), self.patch_embed.num_patches, self.embed_dim

        # -- Pad x to max_patches if needed
        attention_mask = torch.ones(B, N, device=self.pos_embed.device).bool()
        for i, img in enumerate(x):
            if img.shape[1] < N:
                padding = torch.zeros(1, N - img.shape[1], D, device=self.pos_embed.device)
                x[i] = torch.cat([img, padding], dim=1)
                attention_mask[i, -padding.shape[1]:] = 0

        # [B, N, D]
        x = torch.cat(x)

        # -- Interpolate positional encoding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed, grids)

        # -- Add positional embedding to x
        x = x + pos_embed

        # -- Mask x (if masks are provided)
        if masks is not None:
            if len(masks) == 0 or not isinstance(masks[0], list):
                raise ValueError('masks must be a non-empty [B x nmasks] list')
            if len(masks) != B:
                raise ValueError('masks batch size does not match inputs')

            x, attention_mask = pack_by_masks(x, masks)
        else:
            attention_mask = attention_mask

        # -- Forward through blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, attention_mask=attention_mask)

        if self.norm is not None:
            x = self.norm(x)

        # Mask output embeddings (should not contribute to loss)
        return x * attention_mask.unsqueeze(-1)

    def interpolate_pos_encoding(self, x, pos_embed, grid_sizes):
        B, N, D = x.shape
        grid_h = grid_w = int(math.sqrt(pos_embed.shape[1]))
        # Position embeddings [D, N] -> [1, D, sqrt(N), sqrt(N)]
        pos_embed_2d = pos_embed.reshape(1, D, grid_h, grid_w)
        output = torch.zeros(B, N, D, device=self.pos_embed.device)

        for i in range(B):
            h, w = grid_sizes[i]
            pos_embed_interp = nn.functional.interpolate(
                pos_embed_2d,
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
            # [1, D, h, w] -> [1, h*w, D]
            pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, h * w, D)
            output[i, :h*w, :] = pos_embed_interp

        return output


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}
