import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            l_conv = nn.Sequential(
                nn.Conv2d(self.in_channels[i] + (self.in_channels[i + 1] if i == len(self.in_channels) - 2 else self.out_channels), self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),   
            )
            f_conv = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(f_conv)

    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2), (H/4, W/4)]
        Returns:
            outs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2)]
        """
        laterals = inputs
        for i in range(len(inputs) - 2, -1, -1):
            x = F.interpolate(laterals[i + 1], laterals[i].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i] = torch.cat([laterals[i], x], 1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        
        outs = [laterals[i] for i in range(len(laterals) - 1)]
        return outs
    
class SwinT_tiny_Encoder(nn.Module):
    def __init__(self, output_indices, featureShape, out_channels, FPN_in_channels, FPN_out_channels):
        super(SwinT_tiny_Encoder, self).__init__()
        self.model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.output_indices = output_indices
        self.FPN = FPN(FPN_in_channels, FPN_out_channels)
        _, self.fH, self.fW = featureShape
        self.out_channels = out_channels
    
    def forward(self, x):
        """
        Args:
            x: (B, N, C, H, W), N is the number of images at the same time
        Returns:
            out: (B, N, out_channels, fH, fW), feature maps
        """
        imgs = x
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        output = self.model(imgs, output_hidden_states=True)
        ret = [output.reshaped_hidden_states[i] for i in self.output_indices]
        out = self.FPN(ret)
        return out[0].view(B, N, self.out_channels, self.fH, self.fW)