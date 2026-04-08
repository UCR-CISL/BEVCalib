import torch
import torch.nn as nn

from .second import SECOND
from .secondfpn import SECONDFPN

class BEVEncoder(nn.Module):
    def __init__(self):
        super(BEVEncoder, self).__init__()
        self.second = SECOND()
        self.fpn = SECONDFPN()

    def forward(self, x):
        x = self.second(x)
        x = self.fpn(x)
        return x