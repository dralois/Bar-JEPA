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

        :param x: input feature maps from the backbone, shape: [B, C_in, H, W]
        :return: estimated heatmaps for each keypoint, shape: [B, C_out, H*4, W*4]
        """
        x = self.deconv_layers(x)
        heatmaps = self.final_layer(x)
        return heatmaps


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

        :param x: input feature maps from the backbone, shape: [B, C_in, H, W]
        :return: estimated heatmaps for each keypoint, shape: [B, C_out, H*4, W*4]
        """
        # Upsample feature maps by 4 times with bilinear interpolation
        x = F.interpolate(F.relu_(x), scale_factor=4, mode='bilinear', align_corners=False)
        heatmaps = self.final_layer(x)
        return heatmaps


class KeypointDetector(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        num_keypoints=64,
        num_classes=3,
        decoder_type='simple',
        init_std=0.02,
    ):
        """
        Combined keypoint detector with classification and regression heads.

        :param in_channels: number of input channels from the backbone
        :param num_keypoints: max. number of keypoints
        :param num_classes: number of keypoint classes (e.g., tick, bar, background)
        :param decoder_type: type of decoder ('simple' or 'classic')
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.decoder_type = decoder_type

        # ViTPose-inspired Decoder
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(in_channels, num_keypoints)
        elif decoder_type == 'classic':
            self.decoder = ClassicDecoder(in_channels, num_keypoints)
        else:
            raise ValueError(f'Unknown decoder type {decoder_type}')

        self.fc_cls = nn.Conv2d(num_keypoints, num_classes, 1, 1, 0)  # Predicts class probabilities
        self.fc_reg = nn.Conv2d(num_keypoints, 2, 1, 1, 0)  # Predicts (dx, dy) offsets

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

    def forward(self, x):
        """
        Forward pass of the keypoint detector.

        :param x: input feature maps from I-JEPA, shape: [B, N, H, W]
        :return: tuple containing:
            - predicted class probabilities, shape: [B, ncls, H*4, W*4]
            - predicted (dx, dy) offsets, shape: [B, 2, H*4, W*4]
        """
        # Get feature maps from the decoder [B, N, H, W]
        x = self.decoder(x)
        x = self.drop_layer(x)

        # Predict class probabilities and offsets
        cls_pred = self.fc_cls(x)
        reg_pred = F.tanh(self.fc_reg(x))

        return cls_pred, reg_pred
