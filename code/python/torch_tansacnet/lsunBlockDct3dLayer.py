import torch
import torch.nn as nn
from .lsunUtility import Direction, block_dct_matrix_3d
from .lsunLayerExceptions import InvalidInputSize

class LsunBlockDct3dLayer(nn.Module):
    """
    LSUNBLOCKDCT3DLAYER

        ベクトル配列をブロック配列を入力:
            nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)

        コンポーネント別に出力(nComponents):
            nSamples x nRows x nCols x nLays x nDecs

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
        super(LsunBlockDct3dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block DCT of size " \
            + str(self.stride[Direction.VERTICAL]) + "x" \
            + str(self.stride[Direction.HORIZONTAL]) + "x" \
            + str(self.stride[Direction.DEPTH])
        self.num_outputs = number_of_components

    def forward(self,X):
        nComponents = self.num_outputs
        nSamples = X.size(0)
        decV = self.stride[Direction.VERTICAL]
        decH = self.stride[Direction.HORIZONTAL]
        decD = self.stride[Direction.DEPTH]
        if X.size(2)%decV or X.size(3)%decH or X.size(4)%decD:
            raise InvalidInputSize(
                '%d x %d x %d : Input size should be multiples of the stride.'\
                % (X.size(2), X.size(3), X.size(4)))
        nrows = X.size(2)//decV
        ncols = X.size(3)//decH
        nlays = X.size(4)//decD
        ndecs = decV*decH*decD

        # Block DCT matrix (the same as Cvhd in MATLAB lsunBlockDct3dLayer)
        Cvhd = block_dct_matrix_3d(self.stride,dtype=X.dtype,device=X.device)
        # Split into decV x decH x decD blocks, whose voxels are arranged
        # in column-major order as in MATLAB:
        # nSamples x nComponents x nrows x ncols x nlays x (decD x decH x decV)
        arrayX = X.reshape(nSamples,nComponents,nrows,decV,ncols,decH,nlays,decD)\
            .permute(0,1,2,4,6,7,5,3)\
            .reshape(nSamples,nComponents,nrows,ncols,nlays,ndecs)
        # Apply the DCT: nSamples x nComponents x nrows x ncols x nlays x ndecs
        Z = arrayX @ Cvhd.T

        if nComponents<2:
            return torch.squeeze(Z,dim=1)
        else:
            return tuple(torch.squeeze(x,dim=1) for x in torch.chunk(Z,nComponents,dim=1))
