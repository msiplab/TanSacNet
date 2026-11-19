import torch
import torch.nn as nn

class LsunChannelConcatenation3dLayer(nn.Module):
    """
    LSUNCHANNELCONCATENATION3DLAYER

        ２コンポーネント入力(nComponents=2のみサポート):
            nSamples x nRows x nCols x nLays x (nChsTotal-1)
            nSamples x nRows x nCols x nLays

        １コンポーネント出力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nLays x nChsTotal

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
        super(LsunChannelConcatenation3dLayer, self).__init__()
        self.name = name
        self.description = "Channel concatenation"

    def forward(self,Xac,Xdc):
        if Xdc.dim() == 4:
            Xdc = Xdc.unsqueeze(dim=4)
        if Xac.dim() == 4:
            Xac = Xac.unsqueeze(dim=4)
        return torch.cat((Xdc,Xac),dim=4)
