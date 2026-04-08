# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(
        self,
        in_channels=[128, 256],
        out_channels=[256, 256],
        final_out_channels=256,
        upsample_strides=[1, 2],
        use_conv_for_no_stride=True,
    ):
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_fuser = nn.Sequential(
            nn.Conv2d(sum(out_channels), final_out_channels, 1),
            nn.BatchNorm2d(final_out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False
                )
            else:
                stride=np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False
                )

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
    
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
            out = self.conv_fuser(out)
        else:
            out = ups[0]
        return out
