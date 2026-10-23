import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_large, mobilenet_v3_small

ENCODER_TAPS = (1, 3, 6, 12, 16)
ENCODER_CHANNELS = (16, 24, 40, 112, 960)
DECODER_CHANNELS = (32, 48, 64, 96, 128)


class Block(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class RiDepthNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.lateral = nn.ModuleList(nn.Conv2d(c, d, 1) for c, d in zip(ENCODER_CHANNELS, DECODER_CHANNELS))
        self.fuse = nn.ModuleList(
            Block(DECODER_CHANNELS[i] + DECODER_CHANNELS[i + 1], DECODER_CHANNELS[i]) for i in range(len(DECODER_CHANNELS) - 1)
        )
        self.depth_head = nn.Sequential(Block(DECODER_CHANNELS[0], 32), nn.Conv2d(32, 1, 1))
        self.normal_head = nn.Sequential(Block(DECODER_CHANNELS[0], 32), nn.Conv2d(32, 3, 1))

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        hidden = x
        for index, layer in enumerate(self.encoder):
            hidden = layer(hidden)
            if index in ENCODER_TAPS:
                features.append(hidden)
                if index == ENCODER_TAPS[-1]:
                    break
        levels = [lateral(feature) for lateral, feature in zip(self.lateral, features)]
        merged = levels[-1]
        for index in range(len(levels) - 2, -1, -1):
            merged = F.interpolate(merged, size=levels[index].shape[-2:], mode="bilinear", align_corners=False)
            merged = self.fuse[index](torch.cat([levels[index], merged], dim=1))
        merged = F.interpolate(merged, size=size, mode="bilinear", align_corners=False)
        disparity = F.softplus(self.depth_head(merged)).squeeze(1)
        normals = F.normalize(self.normal_head(merged), dim=1)
        return disparity, normals


TINY_TAPS = (0, 1, 3, 8, 11)
TINY_ENCODER_CHANNELS = (16, 16, 24, 48, 96)
TINY_DECODER_CHANNELS = (16, 24, 32, 48, 64)


class Light(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class RiDepthTiny(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features[: TINY_TAPS[-1] + 1]
        self.lateral = nn.ModuleList(nn.Conv2d(c, d, 1) for c, d in zip(TINY_ENCODER_CHANNELS, TINY_DECODER_CHANNELS))
        self.fuse = nn.ModuleList(
            Light(TINY_DECODER_CHANNELS[i] + TINY_DECODER_CHANNELS[i + 1], TINY_DECODER_CHANNELS[i]) for i in range(len(TINY_DECODER_CHANNELS) - 1)
        )
        self.head = nn.Conv2d(TINY_DECODER_CHANNELS[0], 4, 3, padding=1)

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        hidden = x
        for index, layer in enumerate(self.encoder):
            hidden = layer(hidden)
            if index in TINY_TAPS:
                features.append(hidden)
        levels = [lateral(feature) for lateral, feature in zip(self.lateral, features)]
        merged = levels[-1]
        for index in range(len(levels) - 2, -1, -1):
            merged = F.interpolate(merged, size=levels[index].shape[-2:], mode="bilinear", align_corners=False)
            merged = self.fuse[index](torch.cat([levels[index], merged], dim=1))
        output = F.interpolate(self.head(merged), size=size, mode="bilinear", align_corners=False)
        disparity = F.softplus(output[:, 0])
        normals = F.normalize(output[:, 1:], dim=1)
        return disparity, normals


VARIANTS = {"base": RiDepthNet, "tiny": RiDepthTiny}
