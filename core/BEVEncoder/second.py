# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from torch import nn as nn

class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
    ):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i], out_channels[i], 3, stride=layer_strides[i], padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(
                        out_channels[i], out_channels[i], 3, padding=1
                    )
                )
                block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
