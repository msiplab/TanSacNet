import torch
import torch.nn as nn

class LsunChannelConcatenation1dLayer(nn.Module):
    """
    LSUNCHANNELCONCATENATION1DLAYER

        ２コンポーネント入力(nComponents=2のみサポート):
            nSamples x nBlks x (nChsTotal-1)
            nSamples x nBlks (x 1)

        １コンポーネント出力(nComponents=1のみサポート):
            nSamples x nBlks x nChsTotal

    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

    Copyright (c) 2026, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def __init__(self,
        name=''):
        super(LsunChannelConcatenation1dLayer, self).__init__()
        self.name = name
        self.description = "Channel concatenation"

    def forward(self,Xac,Xdc):
        if Xdc.dim() == 2:
            Xdc = Xdc.unsqueeze(dim=2)
        if Xac.dim() == 2:
            Xac = Xac.unsqueeze(dim=2)
        return torch.cat((Xdc,Xac),dim=2)
