import torch.nn as nn
import torch.nn.functional as F


class DUC(nn.Module):
    """
    INPUT: inplanes, planes, upscale_factor
    OUTPUT: (planes // 4)* ht * wd
    """
# used
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)   # 实现卷积用
        # in_channels —— 输入的channels数;
        # out_channels —— 输出的channels数
        # kernel_size ——卷积核的尺寸
        # padding ——输入边沿扩边操作
        # bias ——是否使用偏置(即out = wx+b中的b)
        self.bn = nn.BatchNorm2d(planes)    # 标准化处理，通常用在Relu前防止数据过大使网络不稳定；参数是输入通道数
        self.relu = nn.ReLU()   # 激活函数

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)    # 卷积后图片尺寸变小，使用Pixel Shuffle方法扩大回原尺寸

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
