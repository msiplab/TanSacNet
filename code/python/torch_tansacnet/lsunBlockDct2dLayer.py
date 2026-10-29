import torch
import torch.nn as nn
import math
from .lsunUtility import Direction, block_dct_matrix_2d
    
class LsunBlockDct2dLayer(nn.Module):
    """
    LSUNBLOCKDCT2DLAYER
    
       ベクトル配列をブロック配列を入力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) 
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
        
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
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
        super(LsunBlockDct2dLayer, self).__init__()
        self.stride = stride
        self.name = name
        self.description = "Block DCT of size " \
            + str(self.stride[Direction.VERTICAL]) + "x" \
            + str(self.stride[Direction.HORIZONTAL])
        #self.type = ''
        self.num_outputs = number_of_components
        #self.num_inputs = 1

    def forward(self,X):
        nComponents = self.num_outputs
        nSamples = X.size(0)
        height = X.size(2)
        width = X.size(3)
        stride = self.stride
        decV = stride[Direction.VERTICAL]
        decH = stride[Direction.HORIZONTAL]
        nrows = int(math.ceil(height/decV))
        ncols = int(math.ceil(width/decH))
        ndecs = decV*decH

        # Block DCT matrix (the same as Cvh in MATLAB lsunBlockDct2dLayer)
        Cvh = block_dct_matrix_2d(stride,dtype=X.dtype,device=X.device)
        # Split into decV x decH blocks, whose pixels are arranged in
        # column-major order as in MATLAB:
        # nSamples x nComponents x nrows x ncols x (decH x decV)
        arrayX = X.reshape(nSamples,nComponents,nrows,decV,ncols,decH)\
            .permute(0,1,2,4,5,3)\
            .reshape(nSamples,nComponents,nrows,ncols,ndecs)
        # Apply the DCT: nSamples x nComponents x nrows x ncols x ndecs
        Z = arrayX @ Cvh.T

        if nComponents<2:
            return torch.squeeze(Z,dim=1)
        else:
            return map(lambda x: torch.squeeze(x,dim=1),torch.chunk(Z,nComponents,dim=1))