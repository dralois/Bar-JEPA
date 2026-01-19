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
        :param out_channels: number of output channels (number of keypoints)
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

        :param x: input feature map from the backbone, shape: [1, C_in, H, W]
        :return: estimated heatmap for keypoint, shape: [C_out, H*4, W*4]
        """
        # [1, C_in, H, W] -> [1, C_in, H*4, W*4]
        x = self.deconv_layers(x)
        # [1, C_in, H*4, W*4] -> [C_out, H*4, W*4]
        return self.final_layer(x).squeeze(0)


class SimpleDecoder(nn.Module):

    def __init__(self, in_channels=1280, out_channels=64):
        """
        Simple decoder for keypoint detection.

        :param in_channels: number of input channels from the backbone
        :param out_channels: number of output channels (number of keypoints)
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

        :param x: input feature map from the backbone, shape: [1, C_in, H, W]
        :return: estimated heatmap for keypoint, shape: [C_out, H*4, W*4]
        """
        # [1, C_in, H, W] -> [1, C_in, H*4, W*4]
        x = F.interpolate(F.relu_(x), scale_factor=4, mode='bilinear', align_corners=False)
        # [1, C_in, H*4, W*4] -> [C_out, H*4, W*4]
        return self.final_layer(x).squeeze(0)


class KeypointDetector(nn.Module):

    def __init__(
        self,
        max_patches,
        in_channels=1280,
        num_keypoints=64,
        num_classes=3,
        decoder_type='simple',
        init_std=0.02,
    ):
        """
        Combined keypoint detector with classification and regression heads.

        :param max_patches: max. number of patches (H*W).
        :param in_channels: number of input channels from the backbone.
        :param num_keypoints: max. number of keypoints.
        :param num_classes: number of keypoint classes (e.g. background, tick, bar).
        :param decoder_type: type of decoder ('simple' or 'classic').
        """
        super().__init__()

        self.max_patches = max_patches
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.decoder_type = decoder_type

        # ViTPose-inspired decoders
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(self.in_channels, self.num_keypoints)
        elif decoder_type == 'classic':
            self.decoder = ClassicDecoder(self.in_channels, self.num_keypoints)
        else:
            raise ValueError(f'Unknown decoder type {self.decoder_type}')

        self.fc_cls = nn.Sequential(
            nn.Conv2d(self.num_keypoints, self.num_classes, 1, 1, 0)
        ) # Predicts class probabilities
        self.fc_org = nn.Sequential(
            nn.Conv2d(self.num_keypoints, 1, 1, 1, 0)
        ) # Predicts origin probability
        self.fc_reg = nn.Sequential(
            nn.Conv2d(self.num_keypoints, 2, 1, 1, 0),
            nn.Tanh()
        ) # Predicts (dx, dy) offsets

        self.drop_layer = nn.Dropout(p=0.5)

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

    def _predict_keypoint(
        self,
        kp_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        org_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Directly predicts K keypoint coordinates and the
        corresponding class logits from the latent space.

        :param kp_logits: Latent space predictions, shape [K, H x 4, W x 4]
        :param cls_logits: Class predictions, shape [ncls, H x 4, W x 4]
        :param org_logits: Origin prediction, shape [1, H x 4, W x 4]
        :return: Predicted keypoints, shape [K, 2 + ncls] (x, y, logits)
        """
        _, H, W = kp_logits.shape
        # Create grids [H, W, 2]
        ys, xs = torch.meshgrid(
            (torch.arange(H, device=kp_logits.device) + 0.5) / H,
            (torch.arange(W, device=kp_logits.device) + 0.5) / W,
            indexing="ij",
        )
        # Coordinates grid in image space [H*W, 2]
        coord_grid = torch.stack([ys, xs], dim=-1).view(-1, 2)
        kp_logits = kp_logits.view(self.num_keypoints, -1)
        cls_logits = cls_logits.view(self.num_classes, -1).T
        org_logits = org_logits.view(1, -1).T

        # Spatial softmax per slot -> [K, H*W]
        weights = torch.softmax(kp_logits, dim=1)

        # Predict coordinates -> [K, 2]
        kp_coords = weights @ coord_grid

        # Class logits per slot [H*W, C] -> [K, C]
        kp_cls_logits = weights @ cls_logits

        # Origin logits per slot -> [K, 1]
        kp_org_logits = weights @ org_logits

        return torch.cat(
            [kp_coords, kp_cls_logits, kp_org_logits],
            dim=1
        )

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
            - predicted keypoints, shape: B x [K, 2 + ncls + 1] (x, y, logits)
        """
        cls_preds = []
        reg_preds = []
        kp_preds = []

        for i in range(x.size(0)):
            H, W = grids[i]
            num_patches = H * W

            # [1, N, C_in] -> [1, C_in, H, W], considering only valid patches
            valid_x = x[i, :num_patches]
            valid_x = valid_x.permute(1, 0).view(1, -1, H, W)

            # Get feature map from the decoder [C_out, H*4, W*4]
            valid_x = self.decoder(valid_x)
            valid_x = self.drop_layer(valid_x)

            # Decode class logits and offsets from latent features
            cls_logits = self.fc_cls(valid_x)
            org_logits = self.fc_org(valid_x)
            reg = self.fc_reg(valid_x)

            # Store dense outputs
            cls_preds.append(torch.cat([cls_logits, org_logits], dim=0))
            reg_preds.append(reg)

            # Decode keypoints from latent features directly
            kp_preds.append(
                self._predict_keypoint(
                    valid_x,
                    cls_logits,
                    org_logits,
                )
            )

        return cls_preds, reg_preds, kp_preds
