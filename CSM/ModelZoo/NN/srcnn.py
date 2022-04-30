from torch import nn as nn
from torch.nn import functional as F

class SRCNN(nn.Module):
    def __init__(self, upscale=4):
        super(SRCNN, self).__init__()
        self.upscale = upscale
        self.patch_extraction = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=2)
        self.non_linear = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.reconstruction = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=4)

    def forward(self, x):
        x_1 = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        fm_1 = F.relu(self.patch_extraction(x_1))
        fm_2 = F.relu(self.non_linear(fm_1))
        fm_3 = F.sigmoid(self.reconstruction(fm_2))
        return fm_1, fm_2, fm_3