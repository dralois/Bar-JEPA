from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensors import trunc_normal_


class ClassicDecoder(nn.Module):

    def __init__(self, in_channels=1280, out_channels=64):
        """
        Classic decoder for keypoint detection.

        :param in_channels: number of input channels from the backbone
        :param out_channels: number of output channels (number of keypoints / heatmaps)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Deconvolution layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Final convolution layer
        self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
        nn.init.normal_(self.final_layer.weight, std=0.001, mean=0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the decoder.

        :param x: input feature map from the backbone, shape: [B, C_in, H, W]
        :return: estimated heatmap for keypoint, shape: [B, C_out, H*4, W*4]
        """
        # [1, C_in, H, W] -> [1, C_in, H*4, W*4]
        x = self.deconv_layers(x)
        # [1, C_in, H*4, W*4] -> [C_out, H*4, W*4]
        return self.final_layer(x)


class SimpleDecoder(nn.Module):

    def __init__(self, in_channels=1280, out_channels=64):
        """
        Simple decoder for keypoint detection.

        :param in_channels: number of input channels from the backbone
        :param out_channels: number of output channels (number of keypoints / heatmaps)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Final convolution layer
        self.final_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.final_layer.weight, std=0.001, mean=0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the decoder.

        :param x: input feature map from the backbone, shape: [B, C_in, H, W]
        :return: estimated heatmap for keypoint, shape: [B, C_out, H*4, W*4]
        """
        # [1, C_in, H, W] -> [1, C_in, H*4, W*4]
        x = F.interpolate(F.relu_(x), scale_factor=4, mode='bilinear', align_corners=False)
        # [1, C_in, H*4, W*4] -> [C_out, H*4, W*4]
        return self.final_layer(x)


class KeypointDetector(nn.Module):

    def __init__(
        self,
        max_patches,
        in_channels=1280,
        num_hm_slots=64,
        num_classes=3,
        decoder_type='simple',
        init_std=0.02,
        use_aux_heads=True
    ):
        """
        Combined keypoint detector with classification and regression heads.

        :param max_patches: max. number of patches (H*W).
        :param in_channels: number of input channels from the backbone.
        :param num_hm_slots: max. number of keypoints (-> heatmap slots).
        :param num_classes: number of keypoint classes (e.g. background, tick, bar).
        :param decoder_type: type of decoder ('simple' or 'classic').
        :param use_aux_heads: whether to use auxiliary head or not
        """
        super().__init__()

        self.max_patches = max_patches
        self.in_channels = in_channels
        self.num_hm_slots = num_hm_slots
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.use_aux_heads = use_aux_heads

        # ViTPose-inspired decoders
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(self.in_channels, self.num_hm_slots)
        elif decoder_type == 'classic':
            self.decoder = ClassicDecoder(self.in_channels, self.num_hm_slots)
        else:
            raise ValueError(f'Unknown decoder type {self.decoder_type}')

        if self.use_aux_heads:
            self.fc_cls = nn.Conv2d(self.num_hm_slots, self.num_classes, 1, 1, 0)   # Predicts class probabilities
            self.fc_reg = nn.Conv2d(self.num_hm_slots, 2, 1, 1, 0)                  # Predicts (dx, dy) offsets
            self.fc_org = nn.Conv2d(self.num_hm_slots, 1, 1, 1, 0)                  # Predicts origin probability

        self.drop_layer = nn.Dropout(p=0.3)

        self.init_std = init_std
        self.apply(self._init_weights)

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

    def forward(
        self,
        x: torch.Tensor,
        grids: List[Tuple[int, int]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the keypoint detector.

        :param x: input feature maps, shape: [B, N, D]
        :param grids: list of grid shapes: B x [H, W]
        :return: tuple containing:

            - predicted class logits, shape: B x [ncls, H x 4, W x 4]
            - predicted (dy, dx) offsets, shape: B x [2, H x 4, W x 4]
            - predicted keypoint heatmaps, shape: B x [K, H x 4, W x 4]
        """
        batch_size = x.size(0)
        cls_preds = [None] * batch_size
        reg_preds = [None] * batch_size
        hm_preds = [None] * batch_size

        # Group indices by (H, W) to batch same-size grids through the decoder
        grid_to_indices = {}
        for i, (H, W) in enumerate(grids):
            grid_to_indices.setdefault((H, W), []).append(i)

        # Batched forward pass for each group of grids with the same shape
        for (H, W), idxs in grid_to_indices.items():
            num_patches = H * W

            # [B_g, N, C_in] -> [B_g, C_in, H, W], considering only valid patches
            valid_x = x[idxs, :num_patches]
            valid_x = valid_x.permute(0, 2, 1).reshape(-1, x.size(2), H, W)

            # First dropout before decoder
            valid_x = self.drop_layer(valid_x)
            # Get feature map from the decoder [B_g, C_out, H*4, W*4]
            feat = self.decoder(valid_x)

            # Scatter heatmaps back to the original batch order
            for j, idx in enumerate(idxs):
                hm_preds[idx] = feat[j]

            if self.use_aux_heads:
                # Second dropout for heads
                feat_heads = self.drop_layer(feat)

                # Decode class logits and offsets from latent features
                cls_logits = self.fc_cls(feat_heads)
                org_logits = self.fc_org(feat_heads)
                reg = self.fc_reg(feat_heads)

                # Scatter dense outputs back to the original batch order
                cls_cat = torch.cat([cls_logits, org_logits], dim=1)
                for j, idx in enumerate(idxs):
                    cls_preds[idx] = cls_cat[j] # type: ignore
                    reg_preds[idx] = reg[j]

        return cls_preds, reg_preds, hm_preds # type: ignore
