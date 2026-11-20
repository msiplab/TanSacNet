import torch
import torch.nn as nn
from .lsunUtility import block_dct_matrix_1d

class LsunBlockIdct1dLayer(nn.Module):
    """
    LSUNBLOCKIDCT1DLAYER

        コンポーネント別に入力(nComponents):
            nSamples x nBlks x nDecs

        ベクトル配列をブロック配列にして出力:
            nSamples x nComponents x (Stride x nBlks)

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
        name='',
        stride=None,
        number_of_components=1
        ):
        super(LsunBlockIdct1dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block IDCT of size " + str(self.stride)
        self.num_inputs = number_of_components

    def forward(self,*args):
        dec = self.stride
        components = []
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nSamples = X.size(0)
            nblks = X.size(1)
            # Block IDCT matrix (the transpose of C in MATLAB)
            C = block_dct_matrix_1d(dec,dtype=X.dtype,device=X.device)
            # nSamples x nBlks x Stride -> nSamples x 1 x (Stride x nBlks)
            components.append((X @ C).reshape(nSamples,1,nblks*dec))
        return torch.cat(components,dim=1)
