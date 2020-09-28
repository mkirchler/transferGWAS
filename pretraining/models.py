import torch
from torch import nn
from torchvision import models


class ResNetFeatures(nn.Module):
    def __init__(self, resnet=50):
        super().__init__()
        if resnet == 18:
            self.r = models.resnet18(pretrained=False)
        elif resnet == 34:
            self.r = models.resnet34(pretrained=False)
        elif resnet == 50:
            self.r = models.resnet50(pretrained=False)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.r.conv1(x)
        x = self.r.bn1(x)
        x = self.r.relu(x)
        x = self.r.maxpool(x)

        x = self.r.layer1(x)
        x = self.r.layer2(x)
        x = self.r.layer3(x)
        x = self.r.layer4(x)
        return x


class SpatialDecoder(nn.Module):
    def __init__(
            self,
            d=512,
            features=[360, 240, 160, 80, 40],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            mode='bilinear',
            ):
        super().__init__()
        self.mean, self.std = mean, std

        sizes = [14, 28, 56, 112, 222]
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(d, features[0], 3, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizes[0], sizes[0]), mode=mode, align_corners=False),

                nn.ConvTranspose2d(features[0], features[1], 3, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizes[1], sizes[1]), mode=mode, align_corners=False),

                nn.ConvTranspose2d(features[1], features[2], 3, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizes[2], sizes[2]), mode=mode, align_corners=False),

                nn.ConvTranspose2d(features[2], features[3], 3, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizes[3], sizes[3]), mode=mode, align_corners=False),

                nn.ConvTranspose2d(features[3], features[4], 3, padding=0),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(sizes[4], sizes[4]), mode=mode, align_corners=False),

                nn.ConvTranspose2d(features[4], 3, 3, padding=0),
                nn.Sigmoid()
                )

    def _normalize(self, x):
        dtype = x.dtype
        dev = x.device
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=dev)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=dev)
        return (x - self.mean[..., None, None]) / self.std[..., None, None]

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (7, 7))
        x = self.decoder(x.view(len(x), -1, 7, 7))
        return self._normalize(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FlattenLayer(Lambda):
    def __init__(self):
        super().__init__(func=lambda x: x.view(len(x), -1))
