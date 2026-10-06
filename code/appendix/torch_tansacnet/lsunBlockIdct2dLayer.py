import torch
import torch.nn as nn
import math
from .lsunUtility import Direction, block_dct_matrix_2d

class LsunBlockIdct2dLayer(nn.Module):
   """
   LSUNBLOCKIDCT2DLAYER
   
      コンポーネント別に入力(nComponents):
        nSamples x nRows x nCols x nDecs
   
      ベクトル配列をブロック配列にして出力:
        nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) 
   
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
      super(LsunBlockIdct2dLayer, self).__init__()
      self.stride = stride 
      self.name = name 
      self.description = "Block IDCT of size " \
         + str(self.stride[Direction.VERTICAL]) + "x" \
         + str(self.stride[Direction.HORIZONTAL])
      #self.type = ''
      self.num_inputs = number_of_components

   def forward(self,*args):
        block_size = self.stride
        decV = block_size[Direction.VERTICAL]
        decH = block_size[Direction.HORIZONTAL]
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nsamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            # Block IDCT matrix (the transpose of Cvh in MATLAB)
            Cvh = block_dct_matrix_2d(block_size,dtype=X.dtype,device=X.device)
            # nsamples x nrows x ncols x (decH x decV)
            arrayY = X @ Cvh
            # Place the blocks: nsamples x 1 x height x width
            height = nrows * decV
            width = ncols * decH
            Y = arrayY.reshape(nsamples,nrows,ncols,decH,decV)\
                .permute(0,1,4,2,3)\
                .reshape(nsamples,1,height,width)
            if iComponent<1:
                Z = Y
            else:
                Z = torch.cat((Z,Y),dim=1)
        return Z
