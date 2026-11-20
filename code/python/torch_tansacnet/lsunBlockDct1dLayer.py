import torch
import torch.nn as nn
from .lsunUtility import block_dct_matrix_1d
from .lsunLayerExceptions import InvalidInputSize

class LsunBlockDct1dLayer(nn.Module):
    """
    LSUNBLOCKDCT1DLAYER

        ベクトル配列をブロック配列を入力:
            nSamples x nComponents x (Stride x nBlks)

        コンポーネント別に出力(nComponents):
            nSamples x nBlks x nDecs

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
        super(LsunBlockDct1dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block DCT of size " + str(self.stride)
        self.num_outputs = number_of_components

    def forward(self,X):
        nComponents = self.num_outputs
        nSamples = X.size(0)
        dec = self.stride
        if X.size(2)%dec:
            raise InvalidInputSize(
                '%d : Input size should be a multiple of the stride.'\
                % X.size(2))
        nblks = X.size(2)//dec

        # Block DCT matrix (the same as C in MATLAB lsunBlockDct1dLayer)
        C = block_dct_matrix_1d(dec,dtype=X.dtype,device=X.device)
        # nSamples x nComponents x nBlks x nDecs
        Z = X.reshape(nSamples,nComponents,nblks,dec) @ C.T

        if nComponents<2:
            return torch.squeeze(Z,dim=1)
        else:
            return tuple(torch.squeeze(x,dim=1) for x in torch.chunk(Z,nComponents,dim=1))
