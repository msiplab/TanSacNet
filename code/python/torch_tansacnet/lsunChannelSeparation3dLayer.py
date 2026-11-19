import torch.nn as nn

class LsunChannelSeparation3dLayer(nn.Module):
    """
    LSUNCHANNELSEPARATION3DLAYER

        １コンポーネント入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nLays x nChsTotal

        ２コンポーネント出力(nComponents=2のみサポート):
            nSamples x nRows x nCols x nLays x (nChsTotal-1)
            nSamples x nRows x nCols x nLays

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
        super(LsunChannelSeparation3dLayer, self).__init__()
        self.name = name
        self.description = "Channel separation"

    def forward(self,X):
        return X[...,1:], X[...,0]
