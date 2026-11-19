import torch
import torch.nn as nn
from .lsunUtility import Direction, block_dct_matrix_3d

class LsunBlockIdct3dLayer(nn.Module):
    """
    LSUNBLOCKIDCT3DLAYER

        コンポーネント別に入力(nComponents):
            nSamples x nRows x nCols x nLays x nDecs

        ベクトル配列をブロック配列にして出力:
            nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)

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
        stride=[],
        number_of_components=1
        ):
        super(LsunBlockIdct3dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block IDCT of size " \
            + str(self.stride[Direction.VERTICAL]) + "x" \
            + str(self.stride[Direction.HORIZONTAL]) + "x" \
            + str(self.stride[Direction.DEPTH])
        self.num_inputs = number_of_components

    def forward(self,*args):
        decV = self.stride[Direction.VERTICAL]
        decH = self.stride[Direction.HORIZONTAL]
        decD = self.stride[Direction.DEPTH]
        components = []
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nSamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            nlays = X.size(3)
            # Block IDCT matrix (the transpose of Cvhd in MATLAB)
            Cvhd = block_dct_matrix_3d(self.stride,dtype=X.dtype,device=X.device)
            # nSamples x nrows x ncols x nlays x (decD x decH x decV)
            arrayY = X @ Cvhd
            # Place the blocks: nSamples x 1 x height x width x depth
            components.append(
                arrayY.reshape(nSamples,nrows,ncols,nlays,decD,decH,decV)\
                .permute(0,1,6,2,5,3,4)\
                .reshape(nSamples,1,nrows*decV,ncols*decH,nlays*decD))
        return torch.cat(components,dim=1)
