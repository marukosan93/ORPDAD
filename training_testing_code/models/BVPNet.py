import torch.nn as nn
import torch

class BVPNet(nn.Module):
    def __init__(self, frames=256,fw=48):
        super(BVPNet, self).__init__()
        #add double conv
        self.inc = nn.Sequential(
            nn.Conv2d(3, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(2*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(2*fw),
            nn.ReLU(inplace=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(2*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(4*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(4*fw),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(4*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(8*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(8*fw),
            nn.ReLU(inplace=True),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(8*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(8*fw),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(8*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(4*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(4*fw),
            nn.ReLU(inplace=True),
        )


        self.up2 = nn.Sequential(
            nn.Conv2d(4*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(2*fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(2*fw),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(2*fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
            nn.BatchNorm2d(fw),
            nn.ReLU(inplace=True),
        )

        self.outc = nn.Sequential(
            nn.Conv2d(fw,3, [3,3],stride=1, padding=1,padding_mode="replicate"),
        )

        self.upsample = nn.Upsample(scale_factor=(2,2),mode="nearest")
        self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        # 64,256,3
        x = self.inc(x)
        x = self.avgpool(x)
        # 32,128,48
        x = self.down1(x)
        x = self.avgpool(x)
        # 16,64,96
        x = self.down2(x)
        x = self.avgpool(x)
        # 8,32,192
        x = self.down3(x)
        x = self.avgpool(x)
        # 4,16,384
        feat = self.down4(x)
        # 4,16,384

        x = self.upsample(feat)
        x = self.up1(x)
        # 8,32,192
        x = self.upsample(x)
        x = self.up2(x)
        # 16,64,96
        x = self.upsample(x)
        x = self.up3(x)
        # 32,128,48
        x = self.upsample(x)
        x = self.up4(x)
        # 64,256,48
        x = self.outc(x)
        # 64,256,3
        return x, feat
