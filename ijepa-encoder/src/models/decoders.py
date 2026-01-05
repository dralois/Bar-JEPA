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
        num_classes=4,
        decoder_type='simple',
        init_std=0.02,
    ):
        """
        Combined keypoint detector with classification and regression heads.

        :param max_patches: max. number of patches (H*W).
        :param in_channels: number of input channels from the backbone.
        :param num_keypoints: max. number of keypoints.
        :param num_classes: number of keypoint classes (e.g., tick, bar, origin, background).
        :param decoder_type: type of decoder ('simple' or 'classic').
        """
        super().__init__()
        self.max_patches = max_patches
        self.h_w = int(max_patches ** 0.5)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.decoder_type = decoder_type

        # ViTPose-inspired decoders
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(in_channels, num_keypoints)
        elif decoder_type == 'classic':
            self.decoder = ClassicDecoder(in_channels, num_keypoints)
        else:
            raise ValueError(f'Unknown decoder type {decoder_type}')

        self.fc_cls = nn.Sequential(
            nn.Conv2d(num_keypoints, num_classes, 1, 1, 0)
        ) # Predicts class probabilities

        self.fc_reg = nn.Sequential(
            nn.Conv2d(num_keypoints, 2, 1, 1, 0),
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

    def forward(self, x, grids):
        """
        Forward pass of the keypoint detector.

        :param x: input feature maps, shape: [B, N, D]
        :param grids: list of grid shapes: B x [H, W]
        :return: tuple containing:

            - predicted class probabilities, shape: B x [ncls, H*4, W*4]
            - predicted (dy, dx) offsets, shape: B x [2, H*4, W*4]
        """
        cls_preds = []
        reg_preds = []

        # Process each element in the batch separately to handle variable grid sizes
        for i in range(x.size(0)):
            num_patches = grids[i][0] * grids[i][1]

            # [1, N, C_in] -> [1, C_in, H, W], considering only valid patches
            valid_x = x[i, :num_patches]
            valid_x = valid_x.permute(1, 0).view(1, -1, grids[i][0], grids[i][1])

            # Get feature map from the decoder [C_out, H*4, W*4]
            valid_x = self.decoder(valid_x)
            valid_x = self.drop_layer(valid_x)

            # Predict class probabilities and offsets
            cls_preds.append(self.fc_cls(valid_x))
            reg_preds.append(self.fc_reg(valid_x))

        return cls_preds, reg_preds
