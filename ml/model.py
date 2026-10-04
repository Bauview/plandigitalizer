"""Kompaktes U-Net für die Planerkennung (Wand / Fenster / Tür).

Klein genug, um im Browser über OpenCV-DNN (Pyodide) zu laufen. Die Ausgabe hat die
halbe Eingangsauflösung (Stride-2-Stem), hochskaliert wird ausserhalb des Netzes.
Nur Operatoren, die OpenCV-DNN sicher unterstützt: Conv, BatchNorm, Relu, Resize(nearest), Concat.
"""
from __future__ import annotations

import torch
from torch import nn

N_CLASSES = 4   # 0 Hintergrund, 1 Wand, 2 Fenster, 3 Tür


def cbr(cin, cout, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(cbr(cin + cskip, cout), cbr(cout, cout))

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


class PlanNet(nn.Module):
    def __init__(self, w=(16, 32, 48, 64, 96)):
        super().__init__()
        self.e1 = nn.Sequential(cbr(1, w[0], 2), cbr(w[0], w[0]))                 # 1/2
        self.e2 = nn.Sequential(cbr(w[0], w[1], 2), cbr(w[1], w[1]))              # 1/4
        self.e3 = nn.Sequential(cbr(w[1], w[2], 2), cbr(w[2], w[2]))              # 1/8
        self.e4 = nn.Sequential(cbr(w[2], w[3], 2), cbr(w[3], w[3]))              # 1/16
        self.e5 = nn.Sequential(cbr(w[3], w[4], 2), cbr(w[4], w[4], dil=2), cbr(w[4], w[4], dil=4))  # 1/32
        self.d4 = Up(w[4], w[3], w[3])
        self.d3 = Up(w[3], w[2], w[2])
        self.d2 = Up(w[2], w[1], w[1])
        self.d1 = Up(w[1], w[0], w[0])
        self.head = nn.Conv2d(w[0], N_CLASSES, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        y = self.d4(x5, x4)
        y = self.d3(y, x3)
        y = self.d2(y, x2)
        y = self.d1(y, x1)
        return self.head(y)


if __name__ == "__main__":
    m = PlanNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M Parameter")
    print(m(torch.zeros(1, 1, 256, 256)).shape)
