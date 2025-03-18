import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba import VSSM
from typing import Dict

class LocalFeatureEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(LocalFeatureEnhancement, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x

class GlobalFeatureEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalFeatureEnhancement, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 全局平均池化
        x_global = self.global_avg_pool(x)
        x_global = x_global.view(b, c)
        # 全连接层
        x_global = self.fc1(x_global)
        x_global = self.relu(x_global)
        x_global = self.fc2(x_global)
        x_global = self.sigmoid(x_global)
        # 增强原始特征
        x_global = x_global.view(b, c, 1, 1)
        x = x * x_global
        return x


class DoubleConv(nn.Sequential): #定义两个卷积层
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.AvgPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    #前向传播过程
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class FeatureExtractor2(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=2,
                 mid_channel = 48,
                 depths=[2,2,2,2], 
                 depths_decoder=[2, 2, 9, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path= None,
                 deep_supervision=True
                ):
        super().__init__()    
        self.num_classes = num_classes
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
        

    def forward(self, x):
        f1, f2, f3, f4 = self.vmunet(x) #  [b c h w]
        return [f1, f2, f3, f4]

class FeatureExtractor1(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(FeatureExtractor1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]
    
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
    def forward(self, x1, x2):
        fused = torch.add(x1, x2)
        return fused
        

class BS_Mamba(nn.Module):
    def __init__(self,out_channels: int = 3,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(BS_Mamba, self).__init__()
        self.feature_extractor1 = FeatureExtractor1()
        self.feature_extractor2 = FeatureExtractor2()
        self.attention_fusion = nn.ModuleList([
            AttentionFusion(in_channels) for in_channels in [ 64, 128, 256, 512]
        ])  # 假设每层的通道数
        self.up1 = Up(base_c * 12, base_c * 4, bilinear)
        self.up2 = Up(base_c * 6, base_c * 2, bilinear)
        self.up3 = Up(base_c * 3, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.local_feature_enhancement1 = LocalFeatureEnhancement(in_channels=32, out_channels=32)
        self.local_feature_enhancement2 = LocalFeatureEnhancement(in_channels=64, out_channels=64)
        self.local_feature_enhancement3 = LocalFeatureEnhancement(in_channels=128, out_channels=128)
        self.local_feature_enhancement4 = LocalFeatureEnhancement(in_channels=256, out_channels=256)
        self.global_feature_enhancement1 = GlobalFeatureEnhancement(in_channels=32, out_channels=32)
        self.global_feature_enhancement2 = GlobalFeatureEnhancement(in_channels=64, out_channels=64)
        self.global_feature_enhancement3 = GlobalFeatureEnhancement(in_channels=128, out_channels=128)
        self.global_feature_enhancement4 = GlobalFeatureEnhancement(in_channels=256, out_channels=256)
    def forward(self, x):
        features1 = self.feature_extractor1(x)
        features2 = self.feature_extractor2(x)
        features1[0] = self.local_feature_enhancement1(features1[0])
        features1[1] = self.local_feature_enhancement2(features1[1])
        features1[2] = self.local_feature_enhancement3(features1[2])
        features1[3] = self.local_feature_enhancement4(features1[3])
        features2[0] = self.global_feature_enhancement1(features2[0])
        features2[1] = self.global_feature_enhancement2(features2[1])
        features2[2] = self.global_feature_enhancement3(features2[2])
        features2[3] = self.global_feature_enhancement4(features2[3])
        fused_features = [self.attention_fusion[i](f1, f2) for i, (f1, f2) in enumerate(zip(features1[:4], features2))]
        # fused_features = [torch.cat((f1, f2), dim=1) for f1, f2 in zip(features1, features2)]
        x = self.up1(fused_features[3], fused_features[2])
        x = self.up2(x, fused_features[1])
        x = self.up3(x, fused_features[0])
        x = self.out_conv(x)
        # output = self.decoder(fused_features)
        return x

# 测试网络
# x = torch.randn(1, 3, 256, 256)  # 示例输入
# model = gdnet()
# output = model(x)
# print(output.shape)
