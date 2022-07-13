from torch import nn
import torch.nn.functional as F

class ResnetDecoder(nn.Module):
    def __init__(self):
        super(ResnetDecoder, self).__init__()

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )

    def forward(self, x):  # ,i1,i2,i3):
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear')

        return x


class DensenetDecoder(nn.Module):
    def __init__(self, batch_size=32):
        super(DensenetDecoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x
